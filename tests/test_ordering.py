from aclkv.acl import ACL
from aclkv.ordering import PopularityTracker, RetrievedDoc, order_acl_aware, order_retrieval, order_reuse_aware


def docs(acls):
    return [RetrievedDoc(f"c{i}", f"t{i}", "x", ACL.parse(a), i, 10.0 - i) for i, a in enumerate(acls)]


def test_acl_aware_broad_to_narrow_stable_within_class():
    d = docs(["user:a", "group:g", "public", "public", "group:g", "user:a"])
    out = order_acl_aware(d)
    assert [x.chunk_id for x in out] == ["c2", "c3", "c1", "c4", "c0", "c5"]
    assert [x.chunk_id for x in order_retrieval(out)] == [f"c{i}" for i in range(6)]


def test_max_shift_bounds_displacement():
    d = docs(["user:a", "user:a", "user:a", "public", "public", "public"])
    for shift in (0, 1, 2, 3):
        out = order_acl_aware(d, max_shift=shift)
        for pos, x in enumerate(out):
            assert pos <= x.rank + shift, (shift, [y.chunk_id for y in out])
    assert [x.chunk_id for x in order_acl_aware(d, max_shift=0)] == [f"c{i}" for i in range(6)]
    assert [x.chunk_id for x in order_acl_aware(d, max_shift=1)] == ["c3", "c0", "c1", "c2", "c4", "c5"]


def test_groups_kept_adjacent():
    d = docs(["group:a", "group:b", "group:a", "group:b"])
    out = order_acl_aware(d)
    labels = [x.acl.canonical() for x in out]
    assert labels == ["group:a", "group:a", "group:b", "group:b"]


def test_reuse_aware_cold_equals_acl_aware_and_hot_is_canonical():
    d = docs(["public", "public", "public", "group:g"])
    pop = PopularityTracker(decay=1.0)
    assert [x.chunk_id for x in order_reuse_aware(d, pop)] == [x.chunk_id for x in order_acl_aware(d)]
    for _ in range(20):
        pop.observe(["c2"])
    pop.observe(["c1"])
    out = order_reuse_aware(d, pop)
    # hot chunks first (c2 more popular than c1), cold c0 by rank, group last
    assert [x.chunk_id for x in out] == ["c2", "c1", "c0", "c3"]
    # canonical: the same hot chunks with *different* retrieval ranks come out in the same order
    d2 = [RetrievedDoc(c, c, "x", ACL.parse("public"), r, 0.0) for c, r in (("c1", 0), ("c0", 1), ("c2", 2))]
    assert [x.chunk_id for x in order_reuse_aware(d2, pop)] == ["c2", "c1", "c0"]
