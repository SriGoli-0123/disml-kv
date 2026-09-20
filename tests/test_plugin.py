"""Unit tests for the vLLM plugin logic with a fake kv_cache_utils module."""
import types

from aclkv.scope import Barrier, encode_cache_salt, scope_id
from aclkv.vllm_plugin import build_patched, scoped_extra_keys

KEY = b"k"


class FakeReq:
    def __init__(self, salt, lora=None):
        self.cache_salt = salt
        self.lora_request = lora
        self.mm_features = []
        self.prompt_embeds = None


def fake_module():
    m = types.ModuleType("kcu")

    def orig(request, s, e, mm):
        keys = ([request.cache_salt] if (s == 0 and request.cache_salt) else [])
        return (tuple(keys) if keys else None), mm

    m.generate_block_hash_extra_keys = orig
    m._gen_mm_extra_hash_keys = lambda r, s, e, mm: ([], mm)
    m._gen_lora_extra_hash_keys = lambda r: ([r.lora_request] if r.lora_request else [])
    m._gen_prompt_embeds_extra_hash_keys = lambda r, s, e: []
    return m


def test_scoped_extra_keys_places_scope_in_containing_block_only():
    salt = encode_cache_salt([Barrier(0, scope_id(KEY, "public")), Barrier(40, scope_id(KEY, "group:hr"))])
    assert scoped_extra_keys([], salt, 0, 16) == (scope_id(KEY, "public"),)
    assert scoped_extra_keys([], salt, 16, 32) is None
    assert scoped_extra_keys([], salt, 32, 48) == (scope_id(KEY, "group:hr"),)
    assert scoped_extra_keys(["lora-x"], salt, 48, 64) == ("lora-x",)


def test_malformed_salt_fails_closed():
    bad = "aclkv1:not-a-scope"
    assert scoped_extra_keys([], bad, 0, 16) == (bad,)
    assert scoped_extra_keys([], bad, 16, 32) is None


def test_patched_function_delegates_and_scopes():
    m = fake_module()
    patched = build_patched(m)
    # stock salts keep vLLM behaviour
    r = FakeReq("plain-salt")
    assert patched(r, 0, 16, 0) == (("plain-salt",), 0)
    assert patched(r, 16, 32, 0) == (None, 0)
    # scoped salts
    salt = encode_cache_salt([Barrier(0, scope_id(KEY, "public")), Barrier(20, scope_id(KEY, "user:a"))])
    r = FakeReq(salt, lora="L")
    assert patched(r, 0, 16, 0) == (("L", scope_id(KEY, "public")), 0)
    assert patched(r, 16, 32, 0) == (("L", scope_id(KEY, "user:a")), 0)
    assert patched(r, 32, 48, 0) == (("L",), 0)
    assert patched(FakeReq(None), 0, 16, 0) == (None, 0)
    assert getattr(patched, "_aclkv_scoped_hash_patch") is True
