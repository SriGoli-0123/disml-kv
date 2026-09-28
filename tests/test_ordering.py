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


def test_relevance_anchor_pins_best_doc_last():
    from aclkv.ordering import apply_ordering, apply_relevance_anchor
    d = docs(["public", "public", "group:g", "user:a", "public"])
    out = apply_relevance_anchor(d, anchor_k=2)
    # the two best-ranked docs (c0, c1) move to the end, best (c0) immediately before the question
    assert [x.chunk_id for x in out] == ["c2", "c3", "c4", "c1", "c0"]
    # the rest keep their relative order
    assert [x.chunk_id for x in apply_relevance_anchor(d, anchor_k=0)] == [x.chunk_id for x in d]
    # never anchors the whole prompt (something must stay in front to be cacheable)
    assert len(apply_relevance_anchor(d, anchor_k=99)) == len(d)
    assert apply_relevance_anchor(d, anchor_k=99)[0].chunk_id == "c4"
    # composes with every ordering
    for name in ("retrieval", "acl_aware", "reuse_aware"):
        assert apply_ordering(name, d, anchor_k=2)[-1].chunk_id == "c0"


def test_anchor_takes_the_question_specific_docs_out_of_the_shared_front():
    from aclkv.ordering import PopularityTracker, apply_ordering
    pop = PopularityTracker(decay=1.0)
    for _ in range(10):
        pop.observe(["h1", "h2"])
    shared = [RetrievedDoc("h1", "h", "x", ACL.parse("public"), 3, 0.0),
              RetrievedDoc("h2", "h", "x", ACL.parse("public"), 4, 0.0)]
    # two requests whose own top-ranked docs differ but which share the hot pair
    a = [RetrievedDoc("a1", "a", "x", ACL.parse("public"), 0, 0.0), RetrievedDoc("a2", "a", "x", ACL.parse("public"), 1, 0.0)] + shared
    b = [RetrievedDoc("b1", "b", "x", ACL.parse("public"), 0, 0.0), RetrievedDoc("b2", "b", "x", ACL.parse("public"), 1, 0.0)] + shared
    oa = [x.chunk_id for x in apply_ordering("reuse_aware", a, pop, anchor_k=2)]
    ob = [x.chunk_id for x in apply_ordering("reuse_aware", b, pop, anchor_k=2)]
    assert oa[:2] == ob[:2] == ["h1", "h2"]        # shared prefix at the front
    assert oa[-1] == "a1" and ob[-1] == "b1"       # each request's best doc next to its question
