"""Server-derived cache scopes and their wire encoding.

Design constraints taken from the proposal (and GHSA-wpww-v874-ph2p):

* scope identifiers are **server-derived** — an HMAC over the canonical
  effective ACL using a key the client never sees;
* they are **fixed-size** (``SCOPE_HEX_LEN`` hex chars);
* the number of barriers per request is bounded (``MAX_BARRIERS``) and the
  encoded ``cache_salt`` is far below vLLM's 1024-char limit.

The encoding piggybacks on vLLM's existing per-request ``cache_salt`` field so
that the *only* engine change is how block-hash extra keys are generated
(see ``vllm_plugin.py``)::

    aclkv1:<scope_id>@<token_offset>,<scope_id>@<token_offset>,...

A barrier ``(sid, off)`` means: the block that contains token ``off`` (and,
through vLLM's parent-hash chain, every later block) is hashed with ``sid``
as an extra key.  Later blocks therefore *inherit* the restriction.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass
from functools import lru_cache

SALT_PREFIX = "aclkv1:"
SCOPE_HEX_LEN = 16          # 64-bit scope ids
MAX_BARRIERS = 32
MAX_SALT_LEN = 1024         # vLLM CompletionRequest.cache_salt max_length

_SCOPE_RE = re.compile(rf"^[0-9a-f]{{{SCOPE_HEX_LEN}}}$")
_DEFAULT_DEV_KEY = b"aclkv-dev-key-do-not-use-in-production"


def load_scope_key() -> bytes:
    """Read the HMAC key from ``ACLKV_SCOPE_KEY`` (hex) or fall back to a dev key."""
    raw = os.environ.get("ACLKV_SCOPE_KEY")
    if raw:
        return bytes.fromhex(raw)
    return _DEFAULT_DEV_KEY


def scope_id(key: bytes, canonical_acl: str) -> str:
    """Fixed-size keyed identifier for an effective ACL."""
    mac = hmac.new(key, canonical_acl.encode("utf-8"), hashlib.sha256).hexdigest()
    return mac[:SCOPE_HEX_LEN]


def user_scope_id(key: bytes, user: str) -> str:
    return scope_id(key, f"user:{user}")


def nonce_scope_id() -> str:
    """Unique per-request scope: emulates *no* prefix caching (policy B0)."""
    return secrets.token_hex(SCOPE_HEX_LEN // 2)


@dataclass(frozen=True)
class Barrier:
    token_offset: int
    scope_id: str
    acl_canonical: str = ""     # ground truth for audits; never sent to the engine


def encode_cache_salt(barriers: list[Barrier]) -> str:
    if not barriers:
        raise ValueError("at least one barrier is required")
    if len(barriers) > MAX_BARRIERS:
        raise ValueError(f"too many barriers: {len(barriers)} > {MAX_BARRIERS}")
    prev = -1
    for b in barriers:
        if not _SCOPE_RE.match(b.scope_id):
            raise ValueError(f"scope id must be {SCOPE_HEX_LEN} lowercase hex chars: {b.scope_id!r}")
        if b.token_offset < 0 or b.token_offset < prev:
            raise ValueError("barrier offsets must be non-negative and non-decreasing")
        prev = b.token_offset
    salt = SALT_PREFIX + ",".join(f"{b.scope_id}@{b.token_offset}" for b in barriers)
    if len(salt) > MAX_SALT_LEN:
        raise ValueError("encoded cache_salt exceeds vLLM limit")
    return salt


@lru_cache(maxsize=4096)
def decode_cache_salt(salt: str) -> tuple[tuple[int, str], ...] | None:
    """Parse an encoded salt.  Returns ``None`` when the salt is not ours or
    is malformed — callers must then *fail closed* (treat it as an opaque
    per-request salt, i.e. full isolation)."""
    if not isinstance(salt, str) or not salt.startswith(SALT_PREFIX) or len(salt) > MAX_SALT_LEN:
        return None
    body = salt[len(SALT_PREFIX):]
    if not body:
        return None
    out: list[tuple[int, str]] = []
    prev = -1
    for item in body.split(","):
        if "@" not in item:
            return None
        sid, off_s = item.split("@", 1)
        if not _SCOPE_RE.match(sid) or not off_s.isdigit():
            return None
        off = int(off_s)
        if off < prev:
            return None
        prev = off
        out.append((off, sid))
        if len(out) > MAX_BARRIERS:
            return None
    return tuple(out)


def scope_keys_for_block(barriers, start_token_idx: int, end_token_idx: int) -> list[str]:
    """Scope ids that apply to block ``[start, end)``: every barrier whose
    offset falls inside the block.  Because vLLM chains block hashes through
    the parent hash, a barrier only needs to be keyed into the block that
    contains it — all subsequent blocks inherit it."""
    return [sid for off, sid in barriers if start_token_idx <= off < end_token_idx]
