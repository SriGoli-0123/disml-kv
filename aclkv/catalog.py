"""k-anonymous evidence catalog (the ``Catalog`` policy).

Problem it addresses.  Under ACLKV, shared cache state is created *by user
requests*: a prefix enters the shared scope the moment its k-th distinct
requester arrives, and the cache-aware order is computed from a tree of other
users' recent requests.  Both make what one user sees (a cache hit, the order
of the sources) depend on what colleagues asked recently.

The catalog separates the two.  Users never write to the shared namespace
directly.  Every ``period`` requests a publisher looks back over the last
``window`` requests and *publishes* the evidence prefixes (ordered document
sequences) that at least ``k`` distinct users requested at least
``min_count`` times, within a KV block budget.  Then:

* only published prefixes are hashed into their shared ACL scope; everything
  else (and always the question) stays in the requester's private scope, where
  it is cached with ordinary LRU;
* the evidence order follows only the published catalog (a greedy walk over
  published paths), never other users' individual requests;
* publication happens in batches, so neither channel reveals *when* a
  colleague asked.

``pin`` decides what publication does to the KV cache:

* ``"hard"``: the publisher prefills missing catalog blocks and the engine
  keeps them resident (they reduce the memory left for running requests);
* ``"soft"``: catalog blocks are evicted only after every other free block,
  but they are never reserved and never prefilled;
* ``"none"``: publication only changes the shared allow-list and the order
  (runs on unmodified vLLM with the aclkv plugin).

Privacy.  With hard pins every published block is resident from publication
until the next publish, so a hit on a shared block is a deterministic
function of the published catalog, which at least ``k`` users' requests in
the window produced; it says nothing about who asked since.  With soft or no
pins a published block can be evicted and later recomputed by a single user,
and a colleague's hit would then point at that user.  These modes therefore
also require k-anonymous promotion of the exact served prefix (as ACLKV does)
before it is hashed into the shared scope.

Whether this beats simply evicting with LFU/GDSF/ARC is an empirical question;
see ``docs/CATALOG.md`` for the pre-registered test.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Sequence

from .ordering import RetrievedDoc
from .privacy import PromotionTracker, doc_prefix_key

PIN_MODES = ("hard", "soft", "none")


@dataclass(frozen=True)
class CatalogConfig:
    k: int | None = None          # distinct users per published prefix (None: the policy's k)
    window: int = 1000            # requests the publisher looks back on
    period: int = 500             # publish every ``period`` requests
    budget_frac: float = 0.5      # share of the KV blocks the catalog may cover (>= 1: no limit)
    order: str = "fp"             # "fp": documents sorted by window frequency; "ret": retrieval order
    min_count: int = 2            # requests per published prefix
    pin: str = "soft"

    def __post_init__(self):
        if self.pin not in PIN_MODES:
            raise ValueError(f"pin must be one of {PIN_MODES}, got {self.pin!r}")
        if self.order not in ("fp", "ret"):
            raise ValueError(f"order must be 'fp' or 'ret', got {self.order!r}")
        if self.period < 1 or self.window < 1:
            raise ValueError("period and window must be >= 1")

    @classmethod
    def parse(cls, spec: str) -> "CatalogConfig":
        """``"pin=soft,budget=0.5,period=500,window=1000,order=fp,min_count=2,k=3"``"""
        kw: dict = {}
        names = {"budget": "budget_frac", "budget_frac": "budget_frac", "pin": "pin", "order": "order",
                 "period": "period", "window": "window", "min_count": "min_count", "k": "k"}
        for item in filter(None, spec.split(",")):
            key, val = item.split("=", 1)
            if key not in names:
                raise ValueError(f"unknown catalog option {key!r}")
            f = names[key]
            kw[f] = val if f in ("pin", "order") else (float(val) if f == "budget_frac" else int(val))
        return cls(**kw)

    def tag(self) -> str:
        return (f"pin={self.pin},budget={self.budget_frac:g},period={self.period},window={self.window},"
                f"order={self.order},min_count={self.min_count}")


def _rerank(docs: Sequence[RetrievedDoc]) -> list[RetrievedDoc]:
    return [replace(d, rank=i) for i, d in enumerate(docs)]


class EvidenceCatalog:
    """Publisher state kept by the trusted middleware.

    It also plays the role of the promotion tracker in ``plan_scopes``
    (``would_share`` / ``observe``): a prefix is shared iff it is published.
    """

    def __init__(self, cfg: CatalogConfig, k: int, num_blocks: int | None = None, block_size: int = 16,
                 output_tokens: int = 48):
        self.cfg = cfg
        self.k = cfg.k if cfg.k is not None else k
        if self.k < 2:
            raise ValueError("the catalog needs k >= 2")
        self.num_blocks = num_blocks
        self.bs = block_size
        self.output_tokens = output_tokens
        self.hist: deque = deque(maxlen=cfg.window)      # (user, docs in retrieval order)
        self.paths: dict[tuple[str, ...], int] = {}      # published path -> requests in its window
        self.path_users: dict[tuple[str, ...], frozenset] = {}
        self._shared_keys: set[str] = set()              # prefix digests of published prefixes
        self._base_ids: tuple[str, ...] = ()             # always-included (pinned) documents
        self._base_keys: set[str] = {doc_prefix_key([])}
        self.publish_mode = False
        # soft / no pins: a published prefix is shared only after k distinct users were served it
        self.promotion = None if cfg.pin == "hard" else PromotionTracker(self.k)
        self.n_seen = 0
        self._last_publish = 0
        self.max_prompt_tokens = 0
        self.history: list[dict] = []                    # one entry per publish (diagnostics)

    # ------------------------------------------------ promotion interface
    def set_base_docs(self, chunk_ids: Sequence[str]) -> None:
        """Documents every request starts with (workload-pinned).  Their
        prefixes are shared unconditionally: they reveal nothing about a user."""
        self._base_ids = tuple(chunk_ids)
        self._base_keys = {doc_prefix_key(self._base_ids[:i]) for i in range(len(self._base_ids) + 1)}

    def would_share(self, key: tuple[str, str], user: str) -> bool:
        if self.publish_mode or key[1] in self._base_keys:
            return True
        if key[1] not in self._shared_keys:
            return False
        return self.promotion is None or self.promotion.would_share(key, user)

    def observe(self, key: tuple[str, str], user: str) -> None:
        if self.promotion is not None:
            self.promotion.observe(key, user)

    def published_anonymity(self) -> dict[str, frozenset]:
        """Prefix digest of every published prefix -> the users whose window
        requests got it published (the anonymity set of a hard-pinned hit)."""
        return {doc_prefix_key(self._base_ids + p): u for p, u in self.path_users.items()}

    # ------------------------------------------------------ request side
    def record(self, user: str, docs: Sequence[RetrievedDoc]) -> None:
        self.hist.append((user, sorted(docs, key=lambda d: d.rank)))
        self.n_seen += 1

    def note_prompt(self, num_tokens: int) -> None:
        self.max_prompt_tokens = max(self.max_prompt_tokens, num_tokens)

    def order(self, docs: Sequence[RetrievedDoc], user: str | None = None, audit: dict | None = None) -> list[RetrievedDoc]:
        """Follow the published paths (most requested first), then retrieval order."""
        rem = sorted(docs, key=lambda d: d.rank)
        path: tuple[str, ...] = ()
        head: list[RetrievedDoc] = []
        trail: list[tuple[str, ...]] = []
        while rem:
            opts = [d for d in rem if path + (d.chunk_id,) in self.paths]
            if not opts:
                break
            d = max(opts, key=lambda x: (self.paths[path + (x.chunk_id,)], -x.rank))
            head.append(d)
            rem.remove(d)
            path += (d.chunk_id,)
            trail.append(path)
        out = head + rem
        if audit is not None:
            ranked = sorted(out, key=lambda d: d.rank)
            diff = [i for i, (a, b) in enumerate(zip(out, ranked)) if a.chunk_id != b.chunk_id]
            if diff and head:
                p = trail[min(diff[-1], len(head) - 1)]
                audit["order_anon"] = len(self.path_users[p] - {user})
            else:
                audit["order_anon"] = None
        return out

    # ---------------------------------------------------- publisher side
    def due(self) -> bool:
        return self.n_seen - self._last_publish >= self.cfg.period

    def budget_blocks(self) -> int | None:
        if self.num_blocks is None or self.cfg.budget_frac >= 1:
            limit = None
        else:
            limit = int(self.cfg.budget_frac * self.num_blocks)
        if self.num_blocks is not None and self.cfg.pin == "hard":
            # hard pins must leave room for the largest request seen so far
            reserve = math.ceil((self.max_prompt_tokens + self.output_tokens) / self.bs) + 1
            room = max(0, self.num_blocks - reserve)
            limit = room if limit is None else min(limit, room)
        return limit

    def _candidates(self):
        recent = list(self.hist)
        freq = Counter(d.chunk_id for _, ds in recent for d in ds)
        count: Counter = Counter()
        users: dict[tuple[str, ...], set] = {}
        docmap: dict[str, RetrievedDoc] = {}
        for u, ds in recent:
            seq = sorted(ds, key=lambda d: (-freq[d.chunk_id], d.chunk_id)) if self.cfg.order == "fp" else ds
            path: tuple[str, ...] = ()
            for d in seq:
                path += (d.chunk_id,)
                count[path] += 1
                users.setdefault(path, set()).add(u)
                docmap[d.chunk_id] = d
        cands = [(p, c) for p, c in count.items() if c >= self.cfg.min_count and len(users[p]) >= self.k]
        # parents always have count >= their children, so this visits a parent before its children
        cands.sort(key=lambda pc: (-pc[1], len(pc[0]), pc[0]))
        return cands, users, docmap

    def _path_blocks(self, path, users, docmap, builder, policy, ordering, pinned):
        """Blocks a request starting with ``path`` computes for the preamble,
        the pinned documents and the path (same scopes as a user's request)."""
        pub_policy = replace(policy, ordering="retrieval")
        self.publish_mode = True
        try:
            p = builder.build("catalog-publish", min(users[path]), "", _rerank([docmap[c] for c in path]),
                              pub_policy, ordering=ordering, pinned=pinned, observe=False)
        finally:
            self.publish_mode = False
        from .simulator import prompt_block_hashes

        end = max(sg.end for sg in p.segments if sg.name in ("doc", "pinned"))
        hashes = prompt_block_hashes(p, self.bs)[: end // self.bs]
        return [(h, p.acl_at(i * self.bs + self.bs - 1)) for i, h in enumerate(hashes)]

    def maybe_publish(self, builder, policy, ordering, pinned: Sequence[RetrievedDoc] = ()) -> dict | None:
        """Publish a new catalog if one is due.  Returns the cache event for
        the simulator (``{"mode": "hard"|"soft", "blocks": [(hash, acl), ...]}``)
        or ``None``."""
        if not self.due():
            return None
        self._last_publish = self.n_seen
        cands, users, docmap = self._candidates()
        budget = self.budget_blocks()
        chosen: dict[tuple[str, ...], int] = {}
        blocks: dict[bytes, object] = {}          # insertion order: parents before children
        for path, c in cands:
            if len(path) > 1 and path[:-1] not in chosen:
                continue
            pb = self._path_blocks(path, users, docmap, builder, policy, ordering, pinned)
            extra = [(h, a) for h, a in pb if h not in blocks]
            if budget is not None and len(blocks) + len(extra) > budget:
                continue
            for h, a in extra:
                blocks[h] = a
            chosen[path] = c
        self.paths = chosen
        self.path_users = {p: frozenset(users[p]) for p in chosen}
        self._shared_keys = {doc_prefix_key(self._base_ids + p) for p in chosen}
        self.history.append({"at": self.n_seen, "paths": len(chosen), "blocks": len(blocks),
                             "budget": budget, "depth": max((len(p) for p in chosen), default=0)})
        if self.cfg.pin == "none":
            return None
        return {"mode": self.cfg.pin, "blocks": list(blocks.items())}
