"""Simulator invariants — including the proposal's Week-4 go/no-go checkpoint:
two authorized users must share public/group blocks while an unauthorized
user cannot hit those restricted blocks."""

import pytest

from aclkv.acl import ACL, Directory
from aclkv.context import ContextBuilder
from aclkv.ordering import RetrievedDoc
from aclkv.pipeline import BuildSettings, build_prompts, make_builder
from aclkv.policies import POLICIES, get_policy
from aclkv.simulator import PrefixCacheSimulator, simulate
from aclkv.tokenizer import FakeTokenizer
from tests.conftest import make_workload

KEY = b"test-key"
D = Directory(("alice", "bob", "eve"), ("hr",), {"alice": frozenset({"hr"}), "bob": frozenset({"hr"}), "eve": frozenset()})


def docs():
    return [
        RetrievedDoc("p1", "Pub one", "public text " * 30, ACL.parse("public"), 0, 3.0),
        RetrievedDoc("h1", "HR one", "hr text " * 35, ACL.parse("group:hr"), 1, 2.0),
        RetrievedDoc("p2", "Pub two", "more public " * 20, ACL.parse("public"), 2, 1.0),
    ]


def builder(**kw):
    return ContextBuilder(FakeTokenizer(), D, KEY, block_size=16, chat_format="plain", **kw)


@pytest.mark.parametrize("policy", ["B3", "B4"])
def test_go_no_go_checkpoint(policy):
    """alice and bob (both in hr) share public+hr blocks; eve only shares the public prefix."""
    b = builder()
    pol = get_policy(policy)
    sim = PrefixCacheSimulator(num_blocks=2000, directory=D, block_size=16, max_concurrency=4, output_tokens=16)
    pa = b.build("a", "alice", "same question", docs(), pol)
    pb = b.build("b", "bob", "same question", docs(), pol)
    ra = sim.process(pa)
    rb = sim.process(pb)
    assert ra.cached_tokens == 0
    # bob hits everything but the (uncacheable) partial tail block, since the tail text is identical
    n_full = pb.num_tokens // 16
    assert rb.hit_blocks == min(n_full, (pb.num_tokens - 1) // 16)
    assert rb.unauthorized_hits == 0
    # eve: not in hr -> the filter removes h1; she must not hit any hr-scoped block
    pe = b.build("e", "eve", "same question", docs(), pol)
    re_ = sim.process(pe)
    assert re_.unauthorized_hits == 0
    hr_start = next(s.start for s in pa.segments if s.acl.canonical() == "group:hr")
    assert 0 < re_.cached_tokens <= hr_start           # only the public prefix (preamble [+ p1 (+ p2 under B4)])


@pytest.mark.parametrize("mode", ["no_salt", "forged"])
def test_adversarial_probe(mode):
    """eve replays alice's exact token sequence straight at the engine (bypassing the
    middleware).  Under UB she hits hr-scoped blocks (timing side channel); under
    every scoped policy she hits nothing restricted."""
    b = builder()
    for name, pol in POLICIES.items():
        sim = PrefixCacheSimulator(num_blocks=2000, directory=D, block_size=16, max_concurrency=4, output_tokens=16)
        pa = b.build("a", "alice", "q", docs(), pol)
        sim.process(pa)
        r = sim.probe(pa, "eve", mode)
        if pol.secure:
            assert r.unauthorized_hits == 0, name
            assert r.cached_tokens == 0, name          # not even the public prefix: no scope id
        else:
            assert r.unauthorized_hits > 0, name
            assert r.cached_tokens == ((pa.num_tokens - 1) // 16) * 16


def test_per_user_salt_isolates_even_identical_public_prompts():
    b = builder()
    sim = PrefixCacheSimulator(num_blocks=2000, directory=D, block_size=16, max_concurrency=4, output_tokens=16)
    pa = b.build("a", "alice", "q", docs()[:1], get_policy("B1"))
    pb = b.build("b", "bob", "q", docs()[:1], get_policy("B1"))
    sim.process(pa)
    assert sim.process(pb).cached_tokens == 0
    pa2 = b.build("a2", "alice", "q", docs()[:1], get_policy("B1"))
    assert sim.process(pa2).cached_tokens > 0


def test_b0_never_hits():
    b = builder()
    sim = PrefixCacheSimulator(num_blocks=2000, directory=D, block_size=16, max_concurrency=4, output_tokens=16)
    for i in range(3):
        assert sim.process(b.build(f"r{i}", "alice", "q", docs(), get_policy("B0"))).cached_tokens == 0


def test_lru_eviction_and_capacity():
    b = builder()
    p = b.build("r", "alice", "q", docs(), get_policy("B3"))
    need = (p.num_tokens + 16) // 16 + 1
    sim = PrefixCacheSimulator(num_blocks=need + 2, directory=D, block_size=16, max_concurrency=1, output_tokens=16)
    sim.process(p)
    # a different user with different docs forces evictions once the tiny cache is full
    other = [RetrievedDoc("x1", "X", "other words " * 40, ACL.parse("public"), 0, 1.0)]
    r2 = sim.process(b.build("r2", "bob", "q2", other, get_policy("B3")))
    r3 = sim.process(b.build("r3", "bob", "q3", docs(), get_policy("B3")))
    assert r2.evictions + r3.evictions > 0
    with pytest.raises(RuntimeError):
        PrefixCacheSimulator(num_blocks=3, directory=D, block_size=16, max_concurrency=1).process(p)


def test_matrix_invariants(mini_data):
    """Across a synthetic workload: secure policies have zero unauthorized hits,
    global sharing has some, B0 caches nothing, and ordering helps."""
    w = make_workload(mini_data, share_rate=0.75, seed=5, n_requests=120, acl_mix=(0.4, 0.4, 0.2))
    chunks, _, _ = mini_data
    settings = BuildSettings(tokenizer="fake", chat_format="plain", block_size=16)
    bld = make_builder(w, settings, FakeTokenizer())
    res = {}
    for name in ["B0", "B1", "B2", "B3", "B4", "B5", "UB", "UB-order"]:
        prompts = build_prompts(w, chunks, name, bld, settings)
        assert all(p.n_dropped_by_filter == 0 for p in prompts)
        _, s = simulate(prompts, num_blocks=20000, directory=w.directory, block_size=16, max_concurrency=8,
                        output_tokens=16, probe_rate=0.5, seed=1)
        res[name] = s
    assert res["B0"].cached_tokens == 0
    # honest replay: the filter guarantees every hit is authorized under *every* policy
    for name in res:
        assert res[name].unauthorized_hits == 0, name
    # adversarial probes: only the insecure upper bound leaks
    for name in ["B1", "B2", "B3", "B4", "B5"]:
        assert res[name].probe_requests > 0 and res[name].probe_unauthorized_hits == 0, name
    assert res["UB"].probe_unauthorized_hits > 0
    # efficiency ordering (B4 vs B3 is workload dependent — see README)
    assert res["B3"].cached_tokens >= res["B2"].cached_tokens >= res["B1"].cached_tokens
    assert res["B3"].cached_tokens > res["B1"].cached_tokens
    # ACL scopes never split namespaces between honest, jointly-authorized users:
    # B3 matches the insecure upper bound exactly (same ordering)
    assert res["B3"].cached_tokens == res["UB"].cached_tokens
    assert res["B4"].cached_tokens == res["UB-order"].cached_tokens
    assert res["B5"].cached_tokens >= res["B4"].cached_tokens * 0.9
