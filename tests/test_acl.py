from aclkv.acl import ACL, EMPTY, PUBLIC, Directory, authorization_filter, effective_prefix_acls


def dir3():
    return Directory(("alice", "bob", "carol"), ("hr", "eng"),
                     {"alice": frozenset({"hr"}), "bob": frozenset({"hr", "eng"}), "carol": frozenset({"eng"})})


def test_parse_and_canonical():
    assert ACL.parse("public").is_public
    assert ACL.parse("group:hr").canonical() == "group:hr"
    assert ACL.parse("group:eng&group:hr").canonical() == "group:eng&group:hr"
    assert ACL.parse("public&group:hr").canonical() == "group:hr"     # public is the identity
    assert ACL.parse("user:alice&group:hr").canonical() == "user:alice"


def test_intersection_narrows_monotonically():
    d = dir3()
    seq = [ACL.parse(x) for x in ["public", "group:hr", "public", "group:eng", "user:bob"]]
    eff = effective_prefix_acls(seq)
    assert [e.canonical() for e in eff] == ["public", "group:hr", "group:hr", "group:eng&group:hr", "user:bob"]
    # audiences shrink (C1: public placed after private evidence is still restricted)
    auds = [e.audience(d) for e in eff]
    for a, b in zip(auds, auds[1:]):
        assert b <= a
    assert auds[1] == {"alice", "bob"}
    assert auds[3] == {"bob"}


def test_allows_and_empty():
    d = dir3()
    assert PUBLIC.allows("carol", d)
    assert ACL.parse("group:hr").allows("alice", d) and not ACL.parse("group:hr").allows("carol", d)
    assert ACL.parse("user:alice").allows("alice", d) and not ACL.parse("user:alice").allows("bob", d)
    e = ACL.parse("user:alice").intersect(ACL.parse("user:bob"))
    assert e is EMPTY and not e.allows("alice", d)
    assert e.canonical() == "empty"


def test_breadth_order():
    assert PUBLIC.breadth < ACL.parse("group:hr").breadth < ACL.parse("user:alice").breadth < EMPTY.breadth


def test_authorization_filter():
    d = dir3()

    class D:
        def __init__(self, a):
            self.acl = ACL.parse(a)

    kept, dropped = authorization_filter([D("public"), D("group:eng"), D("user:alice")], "alice", d)
    assert [x.acl.canonical() for x in kept] == ["public", "user:alice"]
    assert [x.acl.canonical() for x in dropped] == ["group:eng"]
