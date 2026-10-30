"""Caching policies.

v2 policy set (paper names)::

    NoCache        no prefix reuse (fresh nonce salt per request)
    PerUser        one salt per user — vLLM's documented mitigation, KVGov's
                   per-principal salt
    Divergence     shared system preamble, per-user from the first document
                   (KVGov's divergence-boundary variant)
    Binary         one public/private boundary with oracle sensitivity labels
                   (SafeKV-style)
    ACLScope       ACL-derived multi-scope hashing, retrieval order
    ACLScope+Tree  ACLScope + cache-aware ordering (CacheWeaver-style tree)
    ACLKV          ours: ACLScope + private query + k-anonymous promotion,
                   with cache-aware (CacheWeaver-style) ordering
    Global         unrestricted sharing — insecure efficiency upper bound
    Global+Tree    CacheWeaver as published (insecure)

All secure v2 policies scope the **question tail to the requesting user**.
v1 names (B0..B6, UB*) are kept as exact aliases of the original
implementation, in which the tail *inherited* the documents' scope, so the
committed v1 results remain reproducible.

Ad-hoc ablations use a spec string::

    "<caching>/<ordering>[:k=3,tail=private,anchor=2]"
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from .acl import ACL, PUBLIC, effective_prefix_acls
from .ordering import RetrievedDoc
from .privacy import PromotionTracker, doc_prefix_key
from .scope import nonce_scope_id, scope_id, user_scope_id

CACHING_MODES = ("none", "per_user", "divergence", "two_level", "acl_scope", "global")
SYSTEM_SCOPE = "system-preamble"
DEFAULT_KANON = 3


@dataclass(frozen=True)
class Policy:
    name: str
    caching: str
    ordering: str
    description: str
    secure: bool = True
    anchor_k: int = 0                # documents pinned immediately before the question (v1 B6)
    tail_scope: str = "private"      # "private" (v2) | "inherit" (v1 behaviour, leaks the query)
    kanon: int = 1                   # k-anonymous promotion threshold (1 = off)
    legacy: bool = False

    @property
    def slug(self) -> str:
        return "".join(ch if ch.isalnum() or ch in "+-._" else "_" for ch in self.name)


_V1 = dict(tail_scope="inherit", legacy=True)

POLICIES: dict[str, Policy] = {
    # ---------------------------------------------------------------- v2
    "NoCache": Policy("NoCache", "none", "retrieval", "no prefix caching (unique nonce salt per request)"),
    "PerUser": Policy("PerUser", "per_user", "retrieval", "per-user salt (vLLM cache_salt / KVGov per-principal)"),
    "Divergence": Policy("Divergence", "divergence", "retrieval", "shared system preamble, per-user after (KVGov divergence)"),
    "Binary": Policy("Binary", "two_level", "retrieval", "one public/private boundary, oracle labels (SafeKV-style)"),
    "ACLScope": Policy("ACLScope", "acl_scope", "retrieval", "ACL-derived multi-scope hashing, retrieval order"),
    "ACLScope+Tree": Policy("ACLScope+Tree", "acl_scope", "tree", "ACLScope + cache-aware ordering (CacheWeaver-style)"),
    "ACLKV": Policy("ACLKV", "acl_scope", "tree",
                    "ours: ACL scopes + private query + k-anonymous promotion + cache-aware order",
                    kanon=DEFAULT_KANON),
    "ACLKV-k1": Policy("ACLKV-k1", "acl_scope", "tree", "ACLKV without k-anonymous promotion (= ACLScope+Tree)"),
    "Global": Policy("Global", "global", "retrieval", "INSECURE unrestricted sharing (upper bound)", secure=False),
    "Global+Tree": Policy("Global+Tree", "global", "tree", "INSECURE CacheWeaver-style ordering, global cache", secure=False),
    # ------------------------------------------------- v1 (exact aliases)
    "B0": Policy("B0", "none", "retrieval", "v1: no prefix caching", **_V1),
    "B1": Policy("B1", "per_user", "retrieval", "v1: per-user salt", **_V1),
    "B2": Policy("B2", "two_level", "retrieval", "v1: one public/private boundary", **_V1),
    "B3": Policy("B3", "acl_scope", "retrieval", "v1: ACL scopes, retrieval order", **_V1),
    "B4": Policy("B4", "acl_scope", "acl_aware", "v1: + broad-to-narrow order", **_V1),
    "B5": Policy("B5", "acl_scope", "reuse_aware", "v1: + reuse-aware canonical order", **_V1),
    "B6": Policy("B6", "acl_scope", "reuse_aware", "v1: B5 + relevance anchor (top-2 last)", anchor_k=2, **_V1),
    "UB": Policy("UB", "global", "retrieval", "v1: INSECURE global sharing", secure=False, **_V1),
    "UB-order": Policy("UB-order", "global", "acl_aware", "v1: INSECURE + ACL-aware order", secure=False, **_V1),
    "UB-reuse": Policy("UB-reuse", "global", "reuse_aware", "v1: INSECURE + reuse-aware order", secure=False, **_V1),
    "UB-anchor": Policy("UB-anchor", "global", "reuse_aware", "v1: INSECURE + reuse order + anchor", secure=False,
                        anchor_k=2, **_V1),
}


def get_policy(name: str) -> Policy:
    if name in POLICIES:
        return POLICIES[name]
    if "/" in name:   # spec: caching/ordering[:k=..,tail=..,anchor=..]
        base, _, opts = name.partition(":")
        caching, ordering = base.split("/", 1)
        if caching not in CACHING_MODES:
            raise ValueError(f"unknown caching mode {caching!r}")
        kw = {}
        for item in filter(None, opts.split(",")):
            k, v = item.split("=", 1)
            if k == "k":
                kw["kanon"] = int(v)
            elif k == "tail":
                if v not in ("private", "inherit"):
                    raise ValueError(f"tail must be private|inherit, got {v!r}")
                kw["tail_scope"] = v
            elif k == "anchor":
                kw["anchor_k"] = int(v)
            else:
                raise ValueError(f"unknown policy option {k!r}")
        return Policy(name, caching, ordering, "custom", secure=(caching != "global"), **kw)
    raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")


def with_options(policy: Policy, **kw) -> Policy:
    return replace(policy, **kw)


@dataclass(frozen=True)
class SegmentScope:
    """Scope assignment for one prompt segment."""
    scope_id: str | None        # None == no salt at all (global sharing)
    acl: ACL                    # ground-truth effective ACL of the prefix ending here
    shared: bool = True         # True if the scope can be shared across users
    prefix_key: str = ""        # digest of the evidence prefix ending at this segment


def plan_scopes(
    caching: str,
    user: str,
    ordered_docs: Sequence[RetrievedDoc],
    key: bytes,
    tail_scope: str = "private",
    kanon: int = 1,
    promotion: PromotionTracker | None = None,
    audience_size: Callable[[ACL], int] | None = None,
) -> list[SegmentScope]:
    """One ``SegmentScope`` per prompt segment: preamble, each doc (in order), question tail."""
    doc_acls = [d.acl for d in ordered_docs]
    eff = effective_prefix_acls(doc_acls, start=PUBLIC)
    truth = [PUBLIC] + eff + [eff[-1] if eff else PUBLIC]          # preamble, docs, tail
    ids = [d.chunk_id for d in ordered_docs]
    pkeys = [doc_prefix_key(ids[:i]) for i in range(len(ids) + 1)]  # preamble, after doc 1, ...
    pkeys.append(doc_prefix_key(ids + ["<tail>"]))
    n = len(truth)
    usid = user_scope_id(key, user)

    if caching == "global":
        out = [SegmentScope(None, a, True, pk) for a, pk in zip(truth, pkeys)]
    elif caching == "none":
        sid = nonce_scope_id()
        return [SegmentScope(sid, a, False, pk) for a, pk in zip(truth, pkeys)]
    elif caching == "per_user":
        return [SegmentScope(usid, a, False, pk) for a, pk in zip(truth, pkeys)]
    elif caching == "divergence":
        out = [SegmentScope(scope_id(key, SYSTEM_SCOPE), truth[0], True, pkeys[0])]
        out += [SegmentScope(usid, a, False, pk) for a, pk in zip(truth[1:], pkeys[1:])]
        return out
    elif caching == "two_level":
        pub = scope_id(key, PUBLIC.canonical())
        out, private = [], False
        for i, (a, pk) in enumerate(zip(truth, pkeys)):
            if not a.is_public:
                private = True
            out.append(SegmentScope(usid if private else pub, a, not private, pk))
    elif caching == "acl_scope":
        out = [SegmentScope(scope_id(key, a.canonical()), a, True, pk) for a, pk in zip(truth, pkeys)]
    else:
        raise ValueError(f"unknown caching mode {caching!r}")

    # question tail: private to the requester (v2) or inherited from the documents (v1)
    if tail_scope == "private" and caching != "global":
        out[-1] = SegmentScope(usid, truth[-1], False, pkeys[-1])
    elif tail_scope not in ("private", "inherit"):
        raise ValueError(f"unknown tail_scope {tail_scope!r}")

    # k-anonymous promotion of shared evidence prefixes
    if kanon > 1 and promotion is not None:
        demoted = False
        for i in range(n - 1):                      # the tail is handled above
            sc = out[i]
            if not sc.shared:
                demoted = True
            pk_key = (truth[i].canonical(), pkeys[i])
            if not demoted:
                aud = audience_size(truth[i]) if audience_size else 10 ** 9
                if aud < kanon or not promotion.would_share(pk_key, user):
                    demoted = True
            if demoted and sc.shared:
                out[i] = SegmentScope(usid, truth[i], False, pkeys[i])
        for i in range(n - 1):
            promotion.observe((truth[i].canonical(), pkeys[i]), user)
        if out[-1].shared and caching == "global":
            pass
    return out
