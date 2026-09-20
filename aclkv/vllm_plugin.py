"""vLLM general plugin: multi-scope (multi-barrier) prefix-cache hashing.

vLLM (v0.29.0) folds a request's ``cache_salt`` into the extra keys of the
**first** block only; because every block hash chains through its parent's
hash, one salt isolates the whole request.  That is the coarse, request-level
trust scope the proposal criticises.

This plugin keeps vLLM's plumbing (``cache_salt`` travels API server →
EngineCore unchanged) but re-interprets salts of the form::

    aclkv1:<sid>@<off>,<sid>@<off>,...

Each ``(sid, off)`` barrier keys ``sid`` into the block that contains token
``off``; the parent-hash chain then propagates it to every later block.  A
request may therefore switch scope several times: e.g. ``public`` for the
preamble and public evidence, ``group:hr`` once HR evidence appears, and
``user:alice`` once private evidence appears.  Two users whose prompts share
an identical public/group prefix *and the same server-derived scope ids*
hash to the same blocks; everyone else lands in a disjoint namespace.

Salts that do not start with ``aclkv1:`` keep vLLM's stock behaviour.  Salts
that start with the prefix but are malformed **fail closed**: the whole
string is used as an opaque first-block salt (full isolation).

Patched function: ``vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys``
which ``get_request_block_hasher`` looks up through the module globals at call
time, inside the EngineCore process where general plugins are loaded.
"""

from __future__ import annotations

import logging
from typing import Any

from .scope import SALT_PREFIX, decode_cache_salt, scope_keys_for_block

logger = logging.getLogger("aclkv.vllm_plugin")

_PATCH_ATTR = "_aclkv_scoped_hash_patch"


def scoped_extra_keys(base_keys: list[Any], cache_salt: str, start_token_idx: int, end_token_idx: int) -> tuple[Any, ...] | None:
    """Pure logic (unit-testable without vLLM).

    ``base_keys`` are vLLM's non-salt extra keys for the block (LoRA name,
    multimodal hashes, prompt-embeds hash).  Returns the tuple vLLM should
    hash for the block, or ``None`` when there is nothing extra.
    """
    barriers = decode_cache_salt(cache_salt)
    keys = list(base_keys)
    if barriers is None:
        # fail closed: opaque salt on the first block => request-level isolation
        if start_token_idx == 0:
            keys.append(cache_salt)
    else:
        keys.extend(scope_keys_for_block(barriers, start_token_idx, end_token_idx))
    return tuple(keys) if keys else None


def build_patched(kcu_module):
    """Create the replacement for ``generate_block_hash_extra_keys``."""
    original = kcu_module.generate_block_hash_extra_keys
    gen_mm = kcu_module._gen_mm_extra_hash_keys
    gen_lora = kcu_module._gen_lora_extra_hash_keys
    gen_pe = getattr(kcu_module, "_gen_prompt_embeds_extra_hash_keys", None)

    def generate_block_hash_extra_keys(request, start_token_idx: int, end_token_idx: int, start_mm_idx: int):
        salt = getattr(request, "cache_salt", None)
        if not salt or not salt.startswith(SALT_PREFIX):
            return original(request, start_token_idx, end_token_idx, start_mm_idx)
        mm_keys, new_mm_idx = gen_mm(request, start_token_idx, end_token_idx, start_mm_idx)
        base = list(gen_lora(request)) + list(mm_keys)
        if gen_pe is not None:
            base += list(gen_pe(request, start_token_idx, end_token_idx))
        return scoped_extra_keys(base, salt, start_token_idx, end_token_idx), new_mm_idx

    setattr(generate_block_hash_extra_keys, _PATCH_ATTR, True)
    generate_block_hash_extra_keys.__wrapped__ = original  # type: ignore[attr-defined]
    return generate_block_hash_extra_keys


def register() -> None:
    """Entry point registered under ``vllm.general_plugins``."""
    try:
        import vllm.v1.core.kv_cache_utils as kcu
    except Exception as e:  # pragma: no cover - only when vLLM is absent/broken
        logger.warning("aclkv: could not import vllm.v1.core.kv_cache_utils (%s); plugin inactive", e)
        return
    if getattr(kcu.generate_block_hash_extra_keys, _PATCH_ATTR, False):
        return
    for name in ("_gen_mm_extra_hash_keys", "_gen_lora_extra_hash_keys"):
        if not hasattr(kcu, name):
            logger.error("aclkv: vLLM API drift — %s missing; scoped hashing NOT installed", name)
            return
    kcu.generate_block_hash_extra_keys = build_patched(kcu)
    logger.info("aclkv: scoped prefix-cache hashing installed (salt prefix %r)", SALT_PREFIX)


def is_installed() -> bool:
    try:
        import vllm.v1.core.kv_cache_utils as kcu
    except Exception:
        return False
    return bool(getattr(kcu.generate_block_hash_extra_keys, _PATCH_ATTR, False))
