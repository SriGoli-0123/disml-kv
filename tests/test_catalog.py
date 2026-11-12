"""Tests for the catalog branch: k-gated order, eviction baselines, pinning,
and the k-anonymous evidence catalog."""
import pytest

from aclkv.acl import ACL, Directory
from aclkv.catalog import CatalogConfig
from aclkv.context import ContextBuilder
from aclkv.ordering import OrderingConfig, PrefixTree, RetrievedDoc, order_tree
from aclkv.pipeline import BuildSettings, build_prompts, make_builder, make_ordering_state
from aclkv.policies import get_policy
from aclkv.simulator import (EVICTION_POLICIES, PrefixCacheSimulator, _ARCFreeQueue, _Block, _PriorityFreeQueue,
                             prompt_block_hashes, simulate)
from aclkv.tokenizer import FakeTokenizer

try:
    from tests.conftest import make_workload
except (ImportError, ModuleNotFoundError):
    from conftest import make_workload

KEY = b"test-key"
USERS = ("alice", "bob", "carol", "dave", "eve")
D = Directory(USERS, ("hr",), {u: frozenset({"hr"}) for u in USERS[:4]} | {"eve": frozenset()})
LONG_Q = "what is the severance policy for employees who leave before the end of their probation period"


def rd(cid, rank, acl="public", words=40):
    return RetrievedDoc(cid, f"T{cid}", f"text of {cid} " * words, ACL.parse(acl), rank)


def builder():
    return ContextBuilder(FakeTokenizer(), D, KEY, block_size=16, chat_format="plain")


# ------------------------------------------------------------------ k-gated order
def test_ktree_follows_only_prefixes_k_users_requested():
    t = PrefixTree(64)
    t.observe(["z", "a"], "bob")                      # a single colleague put z first
    docs = [rd("a", 0), rd("z", 1)]
    plain, gated = {}, {}
    assert [d.chunk_id for d in order_tree(docs, t, user="alice", audit=plain)] == ["z", "a"]
    assert plain["order_anon"] == 1                   # the plain order singles bob out
    assert [d.chunk_id for d in order_tree(docs, t, user="alice", gate_k=3, audit=gated)] == ["a", "z"]
    assert gated["order_anon"] is None                # retrieval order: nothing revealed
    t.observe(["z", "a"], "carol")                    # bob + carol + the requester = 3
    out = order_tree(docs, t, user="alice", gate_k=3, audit=gated)
    assert [d.chunk_id for d in out] == ["z", "a"] and gated["order_anon"] == 2
    t2 = PrefixTree(64)
    t2.observe(["z", "a"], "bob")
    t2.observe(["z", "a"], "alice")
    # bob + alice = 2 distinct users even counting alice: still below k=3
    assert [d.chunk_id for d in order_tree(docs, t2, user="alice", gate_k=3)] == ["a", "z"]


def test_aclkv_uses_the_gated_walk_and_records_the_audit():
    assert get_policy("ACLKV").ordering == "ktree"
    assert get_policy("ACLKV-ungated").ordering == "tree"
    b, st = builder(), OrderingConfig()
    p = b.build("r1", "alice", LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLKV"), ordering=st)
    assert p.meta["order_anon"] is None
    q = b.build("r2", "alice", LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLScope"), ordering=st)
    assert q.meta["order_anon"] is None               # retrieval order is audited and reveals nothing
    r = b.build("r3", "alice", LONG_Q, [rd("a", 0)], get_policy("acl_scope/reuse_aware"), ordering=st)
    assert "order_anon" not in r.meta                 # popularity orders are not audited


# ------------------------------------------------------------------ eviction baselines
def _blocks(n):
    out = [_Block(i) for i in range(n)]
    for b in out:
        b.hash = bytes([b.bid + 1]) * 16
    return out


def test_lfu_evicts_least_used_then_earliest_freed():
    q = _PriorityFreeQueue([], "lfu")
    for bid in range(4):
        q.on_insert(bid)
    q.on_hit(0); q.on_hit(0); q.on_hit(2)
    for bid in range(4):
        q.append(bid)
    assert [q.popleft() for _ in range(4)] == [1, 3, 2, 0]


def test_empty_blocks_are_used_before_any_cached_block():
    for q in (_PriorityFreeQueue([7], "lfu"), _ARCFreeQueue([7], _blocks(8), 8)):
        q.on_insert(0)
        q.append(0)
        assert q.popleft() == 7 and q.popleft() == 0


def test_gdsf_ages_with_the_clock():
    q = _PriorityFreeQueue([], "gdsf")
    q.on_insert(0); q.on_hit(0); q.on_hit(0)          # priority 3
    q.on_insert(1)                                    # priority 1
    q.append(0); q.append(1)
    assert q.popleft() == 1 and q.clock == 1.0
    q.on_insert(2)                                    # priority clock + 1 = 2 < 3
    q.append(2)
    assert q.popleft() == 2 and q.popleft() == 0 and q.clock == 3.0


def test_arc_prefers_t1_victims_and_adapts_on_ghost_hits():
    blocks = _blocks(4)
    q = _ARCFreeQueue([], blocks, capacity=4)
    for bid in range(4):
        q.on_insert(bid)
    q.on_hit(0)                                       # 0 moves to T2
    for bid in range(4):
        q.append(bid)
    assert q.popleft() == 1                           # |T1| = 3 > p = 0: evict T1's LRU
    h1 = blocks[1].hash
    assert h1 in q._ghost[1]
    q.on_insert(1)                                    # recomputing a B1 ghost grows p, enters T2
    assert q.p == 1.0 and q._list[1] == 2
    q.append(1)
    assert q.popleft() == 2                           # |T1| = 2 > p = 1


@pytest.mark.parametrize("eviction", EVICTION_POLICIES)
def test_every_eviction_policy_is_safe_and_lru_is_the_default(mini_data, eviction):
    w = make_workload(mini_data, share_rate=0.5, seed=4, n_requests=120, question_pool_size=10, acl_mix=(0.5, 0.5, 0.0))
    chunks, _, _ = mini_data
    settings = BuildSettings(tokenizer="fake", chat_format="plain")
    prompts = build_prompts(w, chunks, "ACLKV", make_builder(w, settings, FakeTokenizer()), settings)
    _, s = simulate(prompts, 300, w.directory, 16, 4, 16, eviction=eviction)
    assert s.unauthorized_hits == 0 and s.eviction == eviction and s.evictions > 0
    if eviction == "lru":
        _, s0 = simulate(prompts, 300, w.directory, 16, 4, 16)
        assert (s0.cached_tokens, s0.evictions, s0.stalls) == (s.cached_tokens, s.evictions, s.stalls)


def test_inflight_counts_requests_that_fit():
    b = builder()
    ps = [b.build(f"r{i}", "alice", LONG_Q + str(i), [rd("a", 0)], get_policy("NoCache")) for i in range(6)]
    big = PrefixCacheSimulator(4000, D, 16, 4, 16)
    small = PrefixCacheSimulator(2 * (ps[0].num_tokens // 16 + 3), D, 16, 4, 16)
    for p in ps:
        big.process(p)
        small.process(p)
    assert big.summary().mean_inflight > small.summary().mean_inflight


# ------------------------------------------------------------------ pinning
def test_hard_pins_survive_pressure_and_unpinned_blocks_become_evictable():
    b = builder()
    pa = b.build("a", "alice", LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLScope"))
    hs = prompt_block_hashes(pa, 16)
    n_doc = pa.segments[-1].start // 16
    sim = PrefixCacheSimulator(60, D, 16, 2, 16)
    assert sim.set_pinned([(h, pa.acl_at(i * 16 + 15)) for i, h in enumerate(hs[:n_doc])]) == n_doc
    for i in range(20):                               # flood with unrelated requests
        sim.process(b.build(f"x{i}", "bob", LONG_Q + str(i), [rd(f"x{i}", 0)], get_policy("ACLScope")))
    rb = sim.process(b.build("b", "bob", "other " + LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLScope")))
    assert rb.hit_blocks == n_doc and rb.unauthorized_hits == 0
    assert sim.publish_tokens == 16 * n_doc
    sim.set_pinned([])
    sim.drain()
    assert all(blk.ref_cnt == 0 for blk in sim.blocks) and len(sim.free) == sim.num_blocks


def test_soft_pins_are_evicted_last_but_never_reserved():
    b = builder()
    pa = b.build("a", "alice", LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLScope"))
    n_doc = pa.segments[-1].start // 16
    hs = prompt_block_hashes(pa, 16)[:n_doc]
    sim = PrefixCacheSimulator(3 * (pa.num_tokens // 16 + 2), D, 16, 1, 16)
    sim.process(pa)
    sim.apply_publish({"mode": "soft", "blocks": [(h, None) for h in hs]})
    x = b.build("x", "bob", LONG_Q, [rd("x", 0), rd("y", 1)], get_policy("ACLScope"))
    sim.process(x)                                     # fits by evicting unprotected blocks only
    rb = sim.process(b.build("b", "carol", "other " + LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLScope")))
    assert rb.hit_blocks == n_doc
    small = PrefixCacheSimulator(pa.num_tokens // 16 + 3, D, 16, 1, 16)
    small.process(pa)
    small.apply_publish({"mode": "soft", "blocks": [(h, None) for h in hs]})
    small.process(x)                                   # protected blocks give way rather than stall
    assert small.records[-1].stalls <= 1


# ------------------------------------------------------------------ catalog
def test_catalog_config_parse_roundtrip_and_validation():
    c = CatalogConfig.parse("pin=hard,budget=0.25,period=100,window=400,order=ret,min_count=3,k=4")
    assert (c.pin, c.budget_frac, c.period, c.window, c.order, c.min_count, c.k) == ("hard", 0.25, 100, 400, "ret", 3, 4)
    assert CatalogConfig.parse(c.tag() + ",k=4") == c            # tag() covers every field but k
    assert CatalogConfig.parse(c.tag()).k is None
    with pytest.raises(ValueError):
        CatalogConfig(pin="sticky")
    with pytest.raises(ValueError):
        CatalogConfig.parse("bogus=1")


def _catalog_run(requests, cfg, num_blocks=4000):
    """requests: [(user, [doc ids])] -> (prompts, simulator)"""
    b = builder()
    pol = get_policy("Catalog")
    settings = BuildSettings(tokenizer="fake", chat_format="plain")
    st = make_ordering_state(pol, settings, (), cfg, num_blocks, 16)
    sim = PrefixCacheSimulator(num_blocks, D, 16, 2, 16)
    out = []
    for i, (u, ids) in enumerate(requests):
        ev = st.catalog.maybe_publish(b, pol, st, ())
        if ev is not None:
            sim.apply_publish(ev)
        p = b.build(f"r{i}", u, f"{LONG_Q} {i}", [rd(c, j) for j, c in enumerate(ids)], pol, ordering=st)
        sim.process(p)
        out.append(p)
    return out, sim, st.catalog


def test_catalog_shares_only_published_prefixes():
    cfg = CatalogConfig(period=6, window=6, min_count=2, pin="hard", budget_frac=1.0)
    reqs = [("alice", ["a", "b"]), ("bob", ["a", "b"]), ("carol", ["a", "b"]),
            ("alice", ["c", "d"]), ("bob", ["c", "d"]), ("alice", ["c", "d"]),     # c,d: only 2 users
            ("dave", ["b", "a"]), ("eve", ["p", "q"])]
    ps, sim, cat = _catalog_run(reqs, cfg)
    # before the first publish nothing is shared across users
    assert sim.records[1].cached_tokens < ps[1].segments[1].end
    assert ("a",) in cat.paths and ("a", "b") in cat.paths
    assert not any(p[0] == "c" for p in cat.paths)                                  # k = 3 not reached
    d = ps[6]
    assert d.ordered_chunk_ids == ["a", "b"]                                        # order follows the catalog
    assert d.meta["order_anon"] == 3                                                # alice, bob, carol
    docs_end = d.segments[-1].start
    assert sim.records[6].cached_tokens == docs_end                                 # pinned path fully hit
    assert all(r.unauthorized_hits == 0 for r in sim.records)
    # users' copies were private, so the publisher computes the shared doc blocks (the preamble was shared)
    n_pre = ps[0].segments[0].end // 16
    assert sim.publish_tokens == 16 * (cat.history[-1]["blocks"] - n_pre) > 0


def test_catalog_respects_its_block_budget_and_parents_come_first():
    reqs = [(u, ["a", "b", "c"]) for u in ("alice", "bob", "carol")] * 2
    full, sim_full, cat_full = _catalog_run(reqs + [("dave", ["a"])], CatalogConfig(period=6, window=6, budget_frac=1.0,
                                                                                    pin="hard"))
    n_all = cat_full.history[-1]["blocks"]
    cap = n_all - 1
    _, sim_cap, cat = _catalog_run(reqs + [("dave", ["a"])],
                                   CatalogConfig(period=6, window=6, budget_frac=cap / 4000, pin="hard"))
    assert cat.history[-1]["blocks"] <= cap and len(cat.paths) < len(cat_full.paths)
    for p in cat.paths:
        assert len(p) == 1 or p[:-1] in cat.paths


def test_catalog_none_and_soft_modes_emit_no_reservation(mini_data):
    w = make_workload(mini_data, share_rate=0.6, seed=5, n_requests=200, question_pool_size=8, acl_mix=(0.5, 0.5, 0.0))
    chunks, _, _ = mini_data
    settings = BuildSettings(tokenizer="fake", chat_format="plain")
    res = {}
    for pin in ("none", "soft", "hard"):
        cfg = CatalogConfig(period=50, window=100, pin=pin, budget_frac=0.3)
        ps = build_prompts(w, chunks, "Catalog", make_builder(w, settings, FakeTokenizer()), settings,
                           catalog=cfg, num_blocks=1500, output_tokens=16)
        events = [p.meta["publish"] for p in ps if "publish" in p.meta]
        _, s = simulate(ps, 1500, w.directory, 16, 4, 16)
        assert s.unauthorized_hits == 0
        res[pin] = (events, s)
    assert res["none"][0] == []
    assert all(e["mode"] == "soft" for e in res["soft"][0]) and res["soft"][1].publish_tokens == 0
    assert res["hard"][1].publishes == len(res["hard"][0]) > 0
    assert all(len(e["blocks"]) <= int(0.3 * 1500) for e in res["hard"][0])


def test_catalog_attack_trace_never_singles_out(mini_data):
    from aclkv.attacks import run_attack_trace

    w = make_workload(mini_data, share_rate=0.5, seed=2, n_requests=200, question_pool_size=6, acl_mix=(0.6, 0.4, 0.0))
    chunks, _, _ = mini_data
    for r in w.requests:
        r.question = (r.question + " ") * 6
    settings = BuildSettings(tokenizer="fake", chat_format="plain")
    out = {}
    for name in ("Catalog", "ACLKV", "ACLKV-ungated"):
        out[name] = run_attack_trace(w, chunks, name, make_builder(w, settings, FakeTokenizer()), settings,
                                     num_blocks=20000, max_concurrency=8, output_tokens=16, probe_rate=0.5,
                                     bypass_rate=0.2, seed=1, catalog=CatalogConfig(period=40, window=80, pin="hard"))
    for name in ("Catalog", "ACLKV"):
        assert out[name].query_revealed == 0 and out[name].bypass_hits == 0
        assert out[name].min_anon_observed in (-1,) or out[name].min_anon_observed >= 2
        assert out[name].order_singled_out == 0
        assert out[name].order_min_anon == -1 or out[name].order_min_anon >= 2
    assert out["ACLKV-ungated"].order_revealed >= out["ACLKV"].order_revealed


def test_soft_and_unpinned_catalogs_also_need_k_served_requesters():
    """Users were served (b, a); the catalog publishes (a, b).  With hard pins
    dave hits the published blocks at once; with soft/no pins the reordered
    prefix must first be served to k distinct users."""
    reqs = [(u, ["b", "a"]) for u in ("alice", "bob", "carol")] * 2 + [("dave", ["b", "a"])]
    shared = {}
    for pin in ("hard", "soft", "none"):
        ps, sim, cat = _catalog_run(reqs, CatalogConfig(period=6, window=6, pin=pin, budget_frac=1.0))
        assert ("a", "b") in cat.paths and ps[-1].ordered_chunk_ids == ["a", "b"]
        shared[pin] = [s.shared for s in ps[-1].segments if s.name == "doc"]
    assert shared["hard"] == [True, True]
    assert shared["soft"] == shared["none"] == [False, False]


def test_hard_pins_give_way_when_a_request_cannot_fit_otherwise():
    b = builder()
    pa = b.build("a", "alice", LONG_Q, [rd("a", 0), rd("b", 1)], get_policy("ACLScope"))
    n_doc = pa.segments[-1].start // 16
    hs = prompt_block_hashes(pa, 16)[:n_doc]
    big = b.build("x", "bob", LONG_Q, [rd(f"x{i}", i, words=60) for i in range(6)], get_policy("ACLScope"))
    need = -(-(big.num_tokens + 16) // 16)
    sim = PrefixCacheSimulator(need + n_doc - 6, D, 16, 2, 16)   # < need free blocks even after the preamble hits
    sim.set_pinned([(h, pa.acl_at(i * 16 + 15)) for i, h in enumerate(hs)])
    r = sim.process(big)                        # would raise without releasing pins
    assert r.new_blocks + r.hit_blocks == need and 0 < sim.pin_breaks <= n_doc   # shares the preamble
    assert sim.summary().pin_breaks == sim.pin_breaks
    assert all(sim.blocks[bid].ref_cnt >= 1 for bid in sim.pinned.values())
