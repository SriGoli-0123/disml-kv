from aclkv.acl import ACL, Directory
from aclkv.context import ContextBuilder
from aclkv.ordering import RetrievedDoc
from aclkv.policies import get_policy
from aclkv.scope import decode_cache_salt, scope_id
from aclkv.tokenizer import FakeTokenizer

KEY = b"test-key"
D = Directory(("alice", "bob", "eve"), ("hr",), {"alice": frozenset({"hr"}), "bob": frozenset({"hr"}), "eve": frozenset()})


def docs():
    return [
        RetrievedDoc("p1", "Pub one", "public text " * 20, ACL.parse("public"), 0, 3.0),
        RetrievedDoc("h1", "HR one", "hr text " * 25, ACL.parse("group:hr"), 1, 2.0),
        RetrievedDoc("p2", "Pub two", "more public " * 15, ACL.parse("public"), 2, 1.0),
    ]


def builder(**kw):
    return ContextBuilder(FakeTokenizer(), D, KEY, block_size=16, chat_format="plain", **kw)


def test_alignment_and_barriers_hash_mode():
    b = builder()
    p = b.build("r1", "alice", "what?", docs(), get_policy("B3"))
    # every non-final segment ends on a block boundary
    for s in p.segments[:-1]:
        assert s.end % 16 == 0
    names = [s.name for s in p.segments]
    assert names == ["preamble", "doc", "doc", "doc", "tail"]
    # retrieval order keeps p1, h1, p2; effective ACL: public, public, hr, hr, hr
    assert [s.acl.canonical() for s in p.segments] == ["public", "public", "group:hr", "group:hr", "group:hr"]
    # one barrier at 0 (public) and one where the HR doc starts
    assert [(x.token_offset, x.acl_canonical) for x in p.barriers] == [(0, "public"), (p.segments[2].start, "group:hr")]
    dec = decode_cache_salt(p.cache_salt)
    assert dec == ((0, scope_id(KEY, "public")), (p.segments[2].start, scope_id(KEY, "group:hr")))


def test_acl_aware_ordering_moves_public_first():
    b = builder()
    p = b.build("r1", "alice", "what?", docs(), get_policy("B4"))
    assert p.ordered_chunk_ids == ["p1", "p2", "h1"]
    assert [s.acl.canonical() for s in p.segments] == ["public", "public", "public", "group:hr", "group:hr"]


def test_policy_salts():
    b = builder()
    d = docs()
    b0a = b.build("r1", "alice", "q", d, get_policy("B0"))
    b0b = b.build("r2", "alice", "q", d, get_policy("B0"))
    assert b0a.cache_salt != b0b.cache_salt and len(b0a.barriers) == 1
    b1a = b.build("r3", "alice", "q", d, get_policy("B1"))
    b1b = b.build("r4", "alice", "q", d, get_policy("B1"))
    b1c = b.build("r5", "bob", "q", d, get_policy("B1"))
    assert b1a.cache_salt == b1b.cache_salt != b1c.cache_salt
    ub = b.build("r6", "alice", "q", d, get_policy("UB"))
    assert ub.cache_salt is None and ub.barriers == []
    b2 = b.build("r7", "alice", "q", d, get_policy("B2"))
    assert [x.acl_canonical for x in b2.barriers] == ["public", "group:hr"]
    assert decode_cache_salt(b2.cache_salt)[1][1] == scope_id(KEY, "user:alice")


def test_marker_mode_uses_no_salt():
    b = builder(scope_mode="marker")
    p = b.build("r1", "alice", "q", docs(), get_policy("B3"))
    assert p.cache_salt is None and p.barriers == []
    assert [s.name for s in p.segments] == ["marker", "preamble", "doc", "marker", "doc", "doc", "tail"]


def test_filter_drops_unauthorized():
    b = builder()
    p = b.build("r1", "eve", "q", docs(), get_policy("B3"))
    assert p.n_dropped_by_filter == 1 and p.ordered_chunk_ids == ["p1", "p2"]


def test_no_align_option():
    b = builder(align_blocks=False)
    p = b.build("r1", "alice", "q", docs(), get_policy("B3"))
    assert all(s.n_pad == 0 for s in p.segments)
