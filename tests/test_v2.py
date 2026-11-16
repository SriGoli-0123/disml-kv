"""Tests for the v2 fixes and upgrades."""
import random

import pytest

from aclkv.acl import ACL, Directory
from aclkv.context import ContextBuilder
from aclkv.ordering import (OrderingConfig, PopularityTracker, PrefixTree, RetrievedDoc, order_layout, order_lcr,
                            order_tree)
from aclkv.policies import POLICIES, get_policy
from aclkv.privacy import PromotionTracker, doc_prefix_key
from aclkv.scope import Barrier, decode_cache_salt, encode_cache_salt, scope_id, to_salt_regions, user_scope_id
from aclkv.simulator import PrefixCacheSimulator
from aclkv.stats import cluster_bootstrap, mcnemar_exact, sign_test
from aclkv.tokenizer import FakeTokenizer

try:
    from tests.conftest import make_workload
except (ImportError, ModuleNotFoundError):
    from conftest import make_workload

KEY = b"test-key"
USERS = ("alice", "bob", "carol", "dave", "eve")
D = Directory(USERS, ("hr",), {u: frozenset({"hr"}) for u in USERS[:4]} | {"eve": frozenset()})


def docs():
    return [
        RetrievedDoc("p1", "Pub one", "public text " * 30, ACL.parse("public"), 0, 3.0),
        RetrievedDoc("h1", "HR one", "hr text " * 35, ACL.parse("group:hr"), 1, 2.0),
        RetrievedDoc("p2", "Pub two", "more public " * 20, ACL.parse("public"), 2, 1.0),
    ]


def builder():
    return ContextBuilder(FakeTokenizer(), D, KEY, block_size=16, chat_format="plain")


def sim():
    return PrefixCacheSimulator(num_blocks=4000, directory=D, block_size=16, max_concurrency=4, output_tokens=16)


# ------------------------------------------------------------------ fix 1: private query tail
def test_go_no_go_v2_question_is_private():
    """Bob shares Alice's public+hr evidence blocks but NOT her question (v1 leaked it)."""
    b, s = builder(), sim()
    st = OrderingConfig()
    pa = b.build("a", "alice", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("ACLScope"), ordering=st)
    pb = b.build("b", "bob", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("ACLScope"), ordering=st)
    s.process(pa)
    rb = s.process(pb)
    tail_start = pb.segments[-1].start
    assert rb.cached_tokens == tail_start            # every evidence block, no question block
    assert pa.segments[-1].scope_id == user_scope_id(KEY, "alice")
    assert pb.segments[-1].scope_id == user_scope_id(KEY, "bob")


def test_legacy_alias_still_leaks_the_question():
    b, s = builder(), sim()
    st = OrderingConfig()
    s.process(b.build("a", "alice", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("B3"), ordering=st))
    rb = s.process(b.build("b", "bob", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("B3"), ordering=st))
    pb = b.build("b2", "bob", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("B3"), ordering=OrderingConfig())
    assert rb.cached_tokens > pb.segments[-1].start   # v1 behaviour kept for reproducibility


def test_own_repeat_still_hits_own_question():
    b, s = builder(), sim()
    st = OrderingConfig()
    s.process(b.build("a", "alice", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("ACLScope"), ordering=st))
    p2 = b.build("a2", "alice", "what is the severance policy for employees who leave the company before the end of their probation period and how is it computed", docs(), get_policy("ACLScope"), ordering=st)
    assert s.process(p2).cached_tokens > p2.segments[-1].start


# ------------------------------------------------------------------ fix 2: k-anonymous promotion
def test_promotion_tracker_semantics():
    t = PromotionTracker(3)
    key = ("public", doc_prefix_key(["a"]))
    assert not t.would_share(key, "u1")
    t.observe(key, "u1")
    assert not t.would_share(key, "u1") and not t.would_share(key, "u2")
    t.observe(key, "u2")
    t.observe(key, "u2")
    assert t.would_share(key, "u3") and not t.would_share(key, "u1")
    t.observe(key, "u3")
    assert t.would_share(key, "u1") and t.would_share(key, "u9")
    with pytest.raises(ValueError):
        PromotionTracker(0)


def test_kanon_blocks_sharing_until_k_distinct_users():
    pol = get_policy("acl_scope/retrieval:k=3")
    b, s = builder(), sim()
    st = OrderingConfig()
    hits = {}
    for u in ("alice", "bob", "carol", "dave"):
        p = b.build(u, u, f"question from {u}", docs(), pol, ordering=st)
        hits[u] = (s.process(p).cached_tokens, p)
    pre_end = hits["alice"][1].segments[0].end
    assert hits["bob"][0] <= pre_end            # only the system preamble (k not reached)
    assert hits["carol"][0] <= pre_end          # carol is the 3rd distinct user: computes the shared blocks
    assert hits["dave"][0] == hits["dave"][1].segments[-1].start   # dave hits all promoted evidence


def test_kanon_respects_audience_size():
    """A prefix whose audience is smaller than k can never be promoted."""
    small = Directory(("a", "b", "c", "d"), ("tiny",), {"a": frozenset({"tiny"}), "b": frozenset({"tiny"}),
                                                         "c": frozenset(), "d": frozenset()})
    b = ContextBuilder(FakeTokenizer(), small, KEY, block_size=16, chat_format="plain")
    pol = get_policy("acl_scope/retrieval:k=3")
    st = OrderingConfig()
    d2 = [RetrievedDoc("p1", "P", "x " * 40, ACL.parse("public"), 0), RetrievedDoc("t1", "T", "y " * 40, ACL.parse("group:tiny"), 1)]
    for u in ("a", "b", "a", "b"):
        p = b.build(u, u, "q", d2, pol, ordering=st)
    tiny_seg = next(sg for sg in p.segments if sg.chunk_id == "t1")
    assert not tiny_seg.shared and tiny_seg.scope_id == user_scope_id(KEY, "b")


# ------------------------------------------------------------------ new baselines
def test_divergence_shares_only_the_preamble():
    b, s = builder(), sim()
    st = OrderingConfig()
    s.process(b.build("a", "alice", "q", docs(), get_policy("Divergence"), ordering=st))
    pb = b.build("b", "bob", "q", docs(), get_policy("Divergence"), ordering=st)
    assert s.process(pb).cached_tokens == pb.segments[0].end


def test_policy_registry_and_specs():
    assert all(p.tail_scope == "private" for p in POLICIES.values() if not p.legacy)
    assert all(p.tail_scope == "inherit" for p in POLICIES.values() if p.legacy)
    p = get_policy("acl_scope/layout:k=5,tail=inherit,anchor=1")
    assert (p.kanon, p.tail_scope, p.anchor_k, p.ordering) == (5, "inherit", 1, "layout")
    assert "/" not in p.slug and ":" not in p.slug
    with pytest.raises(ValueError):
        get_policy("acl_scope/layout:tail=public")


# ------------------------------------------------------------------ orderings
def rd(cid, rank, acl="public"):
    return RetrievedDoc(cid, cid, "x", ACL.parse(acl), rank, 0.0)


def test_lcr_matches_langchain():
    out = order_lcr([rd(f"d{i}", i) for i in range(5)])
    assert [d.chunk_id for d in out] == ["d0", "d2", "d4", "d3", "d1"]


def test_prefix_tree_walk_and_window():
    t = PrefixTree(window=3)
    for _ in range(2):
        t.observe(["a", "b", "c"], "u1")
    t.observe(["x", "y"], "u2")
    out = order_tree([rd("c", 0), rd("b", 1), rd("a", 2), rd("z", 3)], t)
    assert [d.chunk_id for d in out] == ["a", "b", "c", "z"]
    for _ in range(3):
        t.observe(["x", "y"], "u3")                   # pushes a-b-c out of the window
    assert [d.chunk_id for d in order_tree([rd("c", 0), rd("b", 1), rd("a", 2)], t)] == ["c", "b", "a"]


def test_layout_keeps_relevant_evidence_at_the_edges():
    t, pop = PrefixTree(), PopularityTracker(decay=1.0)
    for u in ("u1", "u2", "u3"):
        t.observe(["h1", "h2"], u)
        pop.observe(["h1", "h2"])
    d = [rd("q0", 0), rd("q1", 1), rd("h1", 5), rd("h2", 6), rd("q2", 2), rd("q3", 3)]
    out = [x.chunk_id for x in order_layout(d, t, pop, lambda a: 100, kanon=3)]
    assert out[:2] == ["h1", "h2"]                  # shared head
    assert out[-1] == "q0" and out[-2] == "q1"      # most relevant non-shared docs next to the question
    assert out[2] == "q3"                           # least relevant in the middle


def test_layout_audience_gating():
    t, pop = PrefixTree(), PopularityTracker(decay=1.0)
    for u in ("u1", "u2", "u3"):
        t.observe(["g1", "p1"], u)
        pop.observe(["g1", "p1"])
    d = [rd("g1", 3, "group:tiny"), rd("p1", 4), rd("q0", 0)]
    sizes = {"group:tiny": 2}
    out = [x.chunk_id for x in order_layout(d, t, pop, lambda a: sizes.get(a.canonical(), 100), kanon=3)]
    assert out[0] == "p1" and "g1" not in out[:1]   # a group of 2 cannot reach k=3: not in the shared head
    out2 = [x.chunk_id for x in order_layout(d, t, pop, lambda a: sizes.get(a.canonical(), 100), kanon=3,
                                              acl_grouping=False)]
    assert out2[0] == "g1"                          # ablation follows the tree regardless of audience


# ------------------------------------------------------------------ position-independent caching
def test_pic_reuses_chunks_at_any_position():
    b = builder()
    st = OrderingConfig()
    s = PrefixCacheSimulator(4000, D, 16, 4, 16, mode="pic", pic_recompute=0.25, pic_scoping="acl")
    s.process(b.build("a", "alice", "q", docs(), get_policy("ACLScope"), ordering=st))
    d0 = docs()
    swapped = [RetrievedDoc(d.chunk_id, d.title, d.text, d.acl, r, d.score) for d, r in zip(d0, (2, 0, 1))]
    pb = b.build("b", "bob", "another question", swapped, get_policy("ACLScope"), ordering=st)
    assert pb.ordered_chunk_ids == ["h1", "p2", "p1"]              # every chunk at a new position
    rb = s.process(pb)
    doc_tokens = sum(sg.end - sg.start for sg in pb.segments if sg.name == "doc")
    pre = pb.segments[0].end
    assert rb.cached_tokens == pre + int(round(0.75 * doc_tokens))
    pe = b.build("e", "eve", "q", docs(), get_policy("ACLScope"), ordering=st)
    re_ = s.process(pe)
    assert re_.unauthorized_hits == 0


# ------------------------------------------------------------------ attacks
def test_attack_harness_orders_policies(mini_data):
    from aclkv.attacks import run_attack_trace
    from aclkv.pipeline import BuildSettings, make_builder

    w = make_workload(mini_data, share_rate=0.5, seed=2, n_requests=150, question_pool_size=6, acl_mix=(0.6, 0.4, 0.0))
    chunks, _, _ = mini_data
    for r in w.requests:   # fake-tokenizer questions are shorter than one KV block; real ones are ~28 tokens
        r.question = (r.question + " ") * 6
    settings = BuildSettings(tokenizer="fake", chat_format="plain")
    res = {}
    for name in ("B3", "ACLScope", "ACLKV", "PerUser", "Global"):
        bld = make_builder(w, settings, FakeTokenizer())
        res[name] = run_attack_trace(w, chunks, name, bld, settings, num_blocks=20000, max_concurrency=8,
                                     output_tokens=16, probe_rate=0.5, bypass_rate=0.3, seed=1)
    assert res["B3"].query_revealed > 0                  # v1 leaks the question text
    assert res["ACLScope"].query_revealed == 0           # v2 never does
    assert res["ACLKV"].query_revealed == 0
    assert res["PerUser"].revealed == 0
    assert res["ACLKV"].min_anon_observed == -1 or res["ACLKV"].min_anon_observed >= 2   # k=3 -> >= k-1 others
    assert res["ACLScope"].singled_out >= res["ACLKV"].singled_out
    for name in ("B3", "ACLScope", "ACLKV", "PerUser"):
        assert res[name].bypass_hits == 0
    assert res["Global"].bypass_hits > 0


# ------------------------------------------------------------------ misc upgrades
def test_salt_regions_adapter():
    bs = [Barrier(0, scope_id(KEY, "public")), Barrier(64, scope_id(KEY, "group:hr"))]
    regions = to_salt_regions(bs)
    assert regions == [(0, bs[0].scope_id), (64, bs[1].scope_id)]
    assert to_salt_regions(decode_cache_salt(encode_cache_salt(bs))) == regions


def test_stats():
    assert abs(mcnemar_exact(45, 29) - 0.0805) < 1e-3
    assert sign_test([1, 1, 1, -1]) == (3, 1, 0.625)
    rng = random.Random(0)
    a = [rng.random() for _ in range(600)]
    b = [x + 0.05 + rng.gauss(0, 0.05) for x in a]
    r = cluster_bootstrap(a, b, [str(i % 200) for i in range(600)], n_boot=1000)
    assert r["ci_low"] < 0.05 < r["ci_high"] and r["p_boot"] < 0.01 and r["n_clusters"] == 200


def test_sparse_bm25_matches_reference():
    import numpy as np
    from rank_bm25 import BM25Okapi

    from aclkv.retrieval import Retriever, SparseBM25, query_dropout, tokenize
    rng = random.Random(0)
    words = [f"w{i}" for i in range(200)]
    texts = [" ".join(rng.choice(words) for _ in range(rng.randint(20, 50))) for _ in range(300)]
    ids = [f"c{i}" for i in range(300)]
    ours, ref = SparseBM25(ids, texts), BM25Okapi([tokenize(t) for t in texts])
    for q in ("w1 w2 w3", "w10 w150"):
        assert np.corrcoef(ours.scores(q), ref.get_scores(tokenize(q)))[0, 1] > 0.999
    r = Retriever(ids, texts)
    hits = r.search("w1 w2 w3", 5, allowed=lambda c: int(c[1:]) % 2 == 0)
    assert len(hits) == 5 and all(int(c[1:]) % 2 == 0 for c, _ in hits)
    assert query_dropout("alpha beta gamma delta epsilon", 0.0, rng) == "alpha beta gamma delta epsilon"
    assert len(query_dropout("alpha beta gamma delta epsilon zeta", 0.5, rng).split()) == 3


def test_topic_workload(mini_data):
    from aclkv.data_prep import BM25Index
    from aclkv.retrieval import Retriever
    from aclkv.workload import Workload, WorkloadConfig, generate_topic

    chunks, questions, _ = mini_data
    ids = sorted(chunks)
    ret = Retriever(ids, [chunks[c].title + " " + chunks[c].text for c in ids])
    cfg = WorkloadConfig(name="t", scenario="pinned", n_users=12, n_groups=3, n_divisions=2, n_cross_groups=2,
                         cross_groups_per_user=1, n_requests=60, docs_per_request=6, question_pool_size=40,
                         n_topics=4, n_pinned=2, query_noise=0.2, seed=3)
    w = generate_topic(cfg, chunks, questions, ret)
    assert len(w.requests) == 60 and len(w.pinned_chunk_ids) == 2
    for r in w.requests:
        assert all(ACL.parse(d["acl"]).allows(r.user, w.directory) for d in r.docs)
        assert not set(w.pinned_chunk_ids) & {d["chunk_id"] for d in r.docs}
    assert any(g.startswith("v") for g in w.directory.groups)
    w2 = generate_topic(cfg, chunks, questions, ret)
    assert [r.docs for r in w.requests] == [r.docs for r in w2.requests]
    assert 0.0 <= w.stats["gold_recall_at_k"] <= 1.0
    _ = BM25Index, Workload
