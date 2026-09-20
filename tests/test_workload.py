from aclkv.acl import ACL
from tests.conftest import make_workload


def test_requests_are_authorized_and_sized(mini_data):
    w = make_workload(mini_data, share_rate=0.5, seed=3)
    assert len(w.requests) == 80
    for r in w.requests:
        assert len(r.docs) == 10
        assert [d["rank"] for d in r.docs] == list(range(10))
        for d in r.docs:
            assert ACL.parse(d["acl"]).allows(r.user, w.directory)
        assert sum(d["is_gold"] for d in r.docs) == 2
    assert w.stats["hot_docs_frac"] > 0.3


def test_share_rate_controls_overlap(mini_data):
    lo = make_workload(mini_data, share_rate=0.0, seed=3)
    hi = make_workload(mini_data, share_rate=0.75, seed=3)
    assert lo.stats["hot_docs_frac"] == 0.0
    assert hi.stats["hot_docs_frac"] > lo.stats["hot_docs_frac"]


def test_determinism(mini_data):
    a = make_workload(mini_data, share_rate=0.25, seed=7)
    b = make_workload(mini_data, share_rate=0.25, seed=7)
    assert [r.docs for r in a.requests] == [r.docs for r in b.requests]
    assert a.chunk_acl == b.chunk_acl


def test_roundtrip(tmp_path, mini_data):
    w = make_workload(mini_data, share_rate=0.25, seed=1, n_requests=10)
    p = tmp_path / "w.json"
    w.save(p)
    from aclkv.workload import Workload
    w2 = Workload.load(p)
    assert w2.config == w.config and [r.req_id for r in w2.requests] == [r.req_id for r in w.requests]
    assert w2.directory.membership == w.directory.membership
