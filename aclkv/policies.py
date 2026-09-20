"""Caching policies: the baselines/variants of the evaluation plan.

    B0  none        no prefix caching (unique nonce per request)
    B1  per_user    safe per-user request salt (vLLM's documented mitigation)
    B2  two_level   one shared/private boundary
    B3  acl_scope   ACL-derived multi-scope caching, normal retrieval order
    B4  acl_scope   + access-aware ordering
    B5  acl_scope   + reuse-aware ordering (stretch)
    UB  global      unrestricted sharing — efficiency upper bound, NOT secure

``plan_scopes`` turns an ordered list of documents into one scope label per
prompt segment (preamble, each document, query tail).  Segments with the same
label as their predecessor need no new barrier; a barrier is emitted wherever
the label changes.  Ground-truth effective ACLs are computed for *every*
policy so the simulator can audit authorized vs. unauthorized hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .acl import ACL, PUBLIC, effective_prefix_acls
from .ordering import RetrievedDoc
from .scope import nonce_scope_id, scope_id, user_scope_id

CACHING_MODES = ("none", "per_user", "two_level", "acl_scope", "global")


@dataclass(frozen=True)
class Policy:
    name: str
    caching: str
    ordering: str
    description: str
    secure: bool = True


POLICIES: dict[str, Policy] = {
    "B0": Policy("B0", "none", "retrieval", "no prefix caching (unique nonce salt per request)"),
    "B1": Policy("B1", "per_user", "retrieval", "safe per-user request salt"),
    "B2": Policy("B2", "two_level", "retrieval", "one shared(public)/private(user) boundary"),
    "B3": Policy("B3", "acl_scope", "retrieval", "ACL-derived multi-scope caching, retrieval order"),
    "B4": Policy("B4", "acl_scope", "acl_aware", "B3 + access-aware (broad-to-narrow) ordering"),
    "B5": Policy("B5", "acl_scope", "reuse_aware", "B3 + reuse-aware ordering (stretch goal)"),
    "UB": Policy("UB", "global", "retrieval", "INSECURE global sharing (upper bound)", secure=False),
    "UB-order": Policy("UB-order", "global", "acl_aware", "INSECURE global sharing + ACL-aware order", secure=False),
    "UB-reuse": Policy("UB-reuse", "global", "reuse_aware", "INSECURE global sharing + reuse-aware order", secure=False),
}


def get_policy(name: str) -> Policy:
    if name in POLICIES:
        return POLICIES[name]
    # ad-hoc "caching/ordering" spec, e.g. "two_level/acl_aware"
    if "/" in name:
        caching, ordering = name.split("/", 1)
        if caching not in CACHING_MODES:
            raise ValueError(f"unknown caching mode {caching!r}")
        return Policy(name, caching, ordering, "custom", secure=(caching != "global"))
    raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")


@dataclass(frozen=True)
class SegmentScope:
    """Scope assignment for one prompt segment."""
    scope_id: str | None        # None == no salt at all (global sharing)
    acl: ACL                    # ground-truth effective ACL of the prefix ending here


def plan_scopes(caching: str, user: str, ordered_docs: Sequence[RetrievedDoc], key: bytes) -> list[SegmentScope]:
    """One ``SegmentScope`` for: preamble, each doc (in order), query tail."""
    doc_acls = [d.acl for d in ordered_docs]
    eff = effective_prefix_acls(doc_acls, start=PUBLIC)        # per doc
    truth = [PUBLIC] + eff + [eff[-1] if eff else PUBLIC]       # preamble, docs, tail

    if caching == "global":
        return [SegmentScope(None, a) for a in truth]

    if caching == "none":
        sid = nonce_scope_id()
        return [SegmentScope(sid, a) for a in truth]

    if caching == "per_user":
        sid = user_scope_id(key, user)
        return [SegmentScope(sid, a) for a in truth]

    if caching == "two_level":
        pub = scope_id(key, PUBLIC.canonical())
        priv = user_scope_id(key, user)
        out: list[SegmentScope] = []
        private = False
        for a in truth:
            if not a.is_public:               # first non-public doc == the boundary
                private = True
            out.append(SegmentScope(priv if private else pub, a))
        return out

    if caching == "acl_scope":
        return [SegmentScope(scope_id(key, a.canonical()), a) for a in truth]

    raise ValueError(f"unknown caching mode {caching!r}")
