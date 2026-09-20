import pytest

from aclkv.scope import (Barrier, MAX_BARRIERS, SALT_PREFIX, SCOPE_HEX_LEN, decode_cache_salt,
                         encode_cache_salt, nonce_scope_id, scope_id, scope_keys_for_block, user_scope_id)

KEY = b"k"


def test_scope_id_is_keyed_and_fixed_size():
    a = scope_id(KEY, "public")
    assert len(a) == SCOPE_HEX_LEN and a == scope_id(KEY, "public")
    assert a != scope_id(b"other", "public")
    assert a != scope_id(KEY, "group:hr")
    assert user_scope_id(KEY, "alice") == scope_id(KEY, "user:alice")
    assert len(nonce_scope_id()) == SCOPE_HEX_LEN and nonce_scope_id() != nonce_scope_id()


def test_roundtrip():
    bs = [Barrier(0, scope_id(KEY, "public")), Barrier(160, scope_id(KEY, "group:hr")), Barrier(400, scope_id(KEY, "user:a"))]
    salt = encode_cache_salt(bs)
    assert salt.startswith(SALT_PREFIX) and len(salt) < 1024
    assert decode_cache_salt(salt) == tuple((b.token_offset, b.scope_id) for b in bs)


@pytest.mark.parametrize("bad", ["", "aclkv1:", "aclkv1:zz@0", "aclkv1:" + "a" * 16 + "@x",
                                 "aclkv1:" + "a" * 16 + "@10," + "b" * 16 + "@5", "other-salt"])
def test_malformed_fails_closed(bad):
    assert decode_cache_salt(bad) is None


def test_limits():
    with pytest.raises(ValueError):
        encode_cache_salt([Barrier(0, "abc")])
    with pytest.raises(ValueError):
        encode_cache_salt([Barrier(i, scope_id(KEY, str(i))) for i in range(MAX_BARRIERS + 1)])
    with pytest.raises(ValueError):
        encode_cache_salt([Barrier(10, scope_id(KEY, "a")), Barrier(5, scope_id(KEY, "b"))])


def test_scope_keys_for_block_only_where_barrier_falls():
    bs = ((0, "a" * 16), (20, "b" * 16), (21, "c" * 16))
    assert scope_keys_for_block(bs, 0, 16) == ["a" * 16]
    assert scope_keys_for_block(bs, 16, 32) == ["b" * 16, "c" * 16]
    assert scope_keys_for_block(bs, 32, 48) == []
