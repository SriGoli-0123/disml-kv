"""Evidence ordering policies (Ideas B and C; challenge C3).

All orderings take the *retrieval-ranked* list of authorized documents and
return a permutation.  ``max_shift`` limits how far a document may move
**later** than its retrieval rank, which is the proposal's mitigation for
"ordering hurts QA quality".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .acl import ACL


@dataclass(frozen=True)
class RetrievedDoc:
    chunk_id: str
    title: str
    text: str
    acl: ACL
    rank: int                 # 0 == most relevant
    score: float = 0.0        # retrieval score (BM25); higher == better
    is_gold: bool = False


class PopularityTracker:
    """Exponentially-decayed frequency of chunk appearances.

    The middleware sees every request, so it can estimate which chunks are
    repeatedly retrieved (an estimate of *authorized reuse*).  ``observe`` is
    called after each request is built, in arrival order, so the estimate is
    deterministic and identical between the simulator and the live bench.
    Decay is applied lazily (value * decay^(steps since last update)).
    """

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self._v: dict[str, tuple[float, int]] = {}
        self._steps = 0

    def observe(self, chunk_ids: Sequence[str]) -> None:
        self._steps += 1
        for c in chunk_ids:
            self._v[c] = (self.score(c) + 1.0, self._steps)

    def score(self, chunk_id: str) -> float:
        v = self._v.get(chunk_id)
        if v is None:
            return 0.0
        val, t = v
        return val * (self.decay ** (self._steps - t)) if self.decay < 1.0 else val

    @property
    def counts(self) -> dict[str, float]:
        return {c: self.score(c) for c in self._v}

    def to_json(self) -> dict:
        return {"decay": self.decay, "counts": self.counts}


class PrefixTree:
    """Trie over a sliding window of recently served evidence sequences.

    This is the data structure behind cache-aware evidence ordering
    (CacheWeaver-style): a greedy walk places first the documents whose prefix
    was served most often, so related requests converge on shared prefixes.
    Each node also tracks *which users* traversed it, which the ACL-aware
    layout uses to prefer prefixes that already satisfy k-anonymous promotion.
    """

    __slots__ = ("window", "root", "_recent")

    class _Node:
        __slots__ = ("children", "n", "users")

        def __init__(self):
            self.children: dict[str, "PrefixTree._Node"] = {}
            self.n = 0
            self.users: dict[str, int] = {}

    def __init__(self, window: int = 512):
        from collections import deque

        self.window = window
        self.root = PrefixTree._Node()
        self._recent: deque = deque()

    def _update(self, seq: Sequence[str], user: str, delta: int) -> None:
        node = self.root
        for c in seq:
            child = node.children.get(c)
            if child is None:
                if delta < 0:
                    return
                child = PrefixTree._Node()
                node.children[c] = child
            child.n += delta
            child.users[user] = child.users.get(user, 0) + delta
            if child.users[user] <= 0:
                del child.users[user]
            if child.n <= 0:
                del node.children[c]
                return
            node = child

    def observe(self, seq: Sequence[str], user: str = "") -> None:
        seq = tuple(seq)
        self._update(seq, user, +1)
        self._recent.append((seq, user))
        if len(self._recent) > self.window:
            old, u = self._recent.popleft()
            self._update(old, u, -1)

    def walk(self, docs: Sequence["RetrievedDoc"], score, eligible=None) -> tuple[list["RetrievedDoc"], list["RetrievedDoc"]]:
        """Greedy walk: repeatedly take the child (among ``docs``) with the
        highest ``score(node, doc)``; stop when no child matches.  Returns
        (head, remaining)."""
        node = self.root
        remaining = list(docs)
        head: list[RetrievedDoc] = []
        while remaining:
            cands = [(d, node.children[d.chunk_id]) for d in remaining
                     if d.chunk_id in node.children and (eligible is None or eligible(head, d))]
            if not cands:
                break
            d, child = max(cands, key=lambda dc: (score(dc[1], dc[0]), -dc[0].rank))
            head.append(d)
            remaining.remove(d)
            node = child
        return head, remaining


def order_retrieval(docs: Sequence[RetrievedDoc]) -> list[RetrievedDoc]:
    """Normal RAG: relevance order."""
    return sorted(docs, key=lambda d: d.rank)


def _greedy_with_deadline(docs: Sequence[RetrievedDoc], key, max_shift: int | None) -> list[RetrievedDoc]:
    """Place documents greedily by ``key`` while guaranteeing that no document
    ends up more than ``max_shift`` positions later than its retrieval rank.

    Ranks are distinct, so at any output position at most one document is
    "due" (rank + max_shift == position); it is placed immediately.
    """
    remaining = sorted(docs, key=lambda d: d.rank)
    out: list[RetrievedDoc] = []
    n = len(remaining)
    shift = n if max_shift is None else max(0, max_shift)
    for pos in range(n):
        due = [d for d in remaining if d.rank + shift <= pos]
        if due:
            pick = min(due, key=lambda d: d.rank)
        else:
            pick = min(remaining, key=key)
        out.append(pick)
        remaining.remove(pick)
    return out


def order_acl_aware(docs: Sequence[RetrievedDoc], max_shift: int | None = None) -> list[RetrievedDoc]:
    """Idea B: broad-to-narrow (public → group → user), retrieval rank within class.

    A prefix stays reusable by a larger audience until narrower evidence is
    introduced, so this delays the point at which sharing becomes restricted.
    """
    return _greedy_with_deadline(docs, key=lambda d: (d.acl.breadth, d.acl.canonical() if d.acl.breadth == 1 else "", d.rank), max_shift=max_shift)


def order_reuse_aware(
    docs: Sequence[RetrievedDoc],
    popularity: PopularityTracker,
    min_pop: float = 1.0,
    max_shift: int | None = None,
) -> list[RetrievedDoc]:
    """Idea C (stretch): reuse-aware ordering.

    Prefix reuse needs *identical* token sequences, so two requests that hold
    the same public chunks in a different order share nothing past the first
    difference.  Within each breadth class we therefore put the chunks that
    are estimated to be reusable ("hot": decayed popularity >= ``min_pop``)
    first, in a **canonical** order that does not depend on the request
    (popularity bucket, then chunk id), and keep the remaining "cold" chunks
    in relevance order — they will not be shared anyway, so their order only
    matters for answer quality.  When overlap is low nothing is hot and the
    order degenerates to ``order_acl_aware``; ``max_shift`` still bounds how
    far any chunk may be delayed.
    """

    def bucket(cid: str) -> int:   # coarse so that small count changes do not reorder
        return int(math.floor(math.log2(1.0 + popularity.score(cid))))

    def key(d: RetrievedDoc):
        grp = d.acl.canonical() if d.acl.breadth == 1 else ""
        if popularity.score(d.chunk_id) >= min_pop:
            return (d.acl.breadth, grp, 0, -bucket(d.chunk_id), d.chunk_id, 0)
        return (d.acl.breadth, grp, 1, 0, "", d.rank)

    return _greedy_with_deadline(docs, key=key, max_shift=max_shift)


def apply_relevance_anchor(docs: Sequence[RetrievedDoc], anchor_k: int) -> list[RetrievedDoc]:
    """Move the ``anchor_k`` most relevant documents to the **end** of the prompt,
    least-relevant-of-them first, so the single best-ranked document sits
    immediately before the question.

    Rationale (challenge C3): the two objectives want opposite ends of the
    prompt.  A prefix-cache match runs left to right and dies at the first
    question-specific document, so the cache only cares about the *front*.
    Language models attend most strongly to the start and the end of a context
    and least to the middle, and the answer is generated right after the
    question, so answer quality cares most about the *back*.  Anchoring the
    question-specific evidence at the back therefore costs the cache nothing —
    those documents were the source of divergence wherever they sat — while
    moving it out of the weak middle.
    """
    if anchor_k <= 0 or len(docs) <= 1:
        return list(docs)
    k = min(anchor_k, len(docs) - 1)          # never anchor the whole prompt
    anchored = sorted(docs, key=lambda d: d.rank)[:k]
    ids = {d.chunk_id for d in anchored}
    head = [d for d in docs if d.chunk_id not in ids]
    return head + sorted(anchored, key=lambda d: -d.rank)


def order_tree(docs: Sequence[RetrievedDoc], tree: PrefixTree, kanon: int = 1, audience_size=None,
               privacy_aware: bool = False) -> list[RetrievedDoc]:
    """Cache-aware ordering (CacheWeaver-style): follow the most frequently
    served prefix, then retrieval order.

    ``privacy_aware=True`` is the ablation in which the walk prefers prefixes
    already requested by >= ``kanon`` distinct users and never extends the
    prefix to an effective ACL whose audience is smaller than ``kanon`` (such
    a prefix can never be promoted).  It did not change reuse on our
    development workload, so ACLKV uses the plain walk.
    """
    if not privacy_aware:
        head, rest = tree.walk(docs, score=lambda node, d: node.n)
        return head + sorted(rest, key=lambda d: d.rank)
    from .acl import PUBLIC

    def eligible(head, d):
        if kanon <= 1 or audience_size is None:
            return True
        a = PUBLIC
        for x in head:
            a = a.intersect(x.acl)
        return audience_size(a.intersect(d.acl)) >= kanon

    head, rest = tree.walk(docs, score=lambda node, d: (len(node.users) >= kanon, len(node.users), node.n),
                           eligible=eligible)
    return head + sorted(rest, key=lambda d: d.rank)


def order_lcr(docs: Sequence[RetrievedDoc]) -> list[RetrievedDoc]:
    """LongContextReorder (LangChain): most relevant documents at the two
    edges of the context, least relevant in the middle.  Exact port of the
    published algorithm, applied to the relevance-sorted list."""
    ranked = sorted(docs, key=lambda d: d.rank)
    ranked.reverse()
    out: list[RetrievedDoc] = []
    for i, d in enumerate(ranked):
        if i % 2 == 1:
            out.append(d)
        else:
            out.insert(0, d)
    return out


def order_layout(
    docs: Sequence[RetrievedDoc],
    tree: PrefixTree,
    popularity: PopularityTracker,
    audience_size,
    kanon: int = 1,
    min_pop: float = 1.0,
    acl_grouping: bool = True,
) -> list[RetrievedDoc]:
    """Edge-placement layout (ablation; v1's "relevance anchor" generalised).

    Evaluated and *rejected* on the realistic workloads: with real retrieval
    the most relevant documents are also the most stable across paraphrased
    queries, so moving them away from the front shortens shared prefixes
    (see the paper's ordering ablation).  Kept for reproducibility.

    The prompt is split into a **shared head** and a **private tail of
    evidence**, and relevant evidence is kept at the two edges:

    1. *Shared head.*  Greedy walk on the prefix tree, preferring prefixes
       already requested by >= ``kanon`` distinct users (so they are, or will
       soon be, promoted to the shared scope), then by frequency, then by
       audience breadth.  A document may extend the head only while the
       effective ACL of the extended prefix still has an audience of at least
       ``kanon`` users — a prefix narrower than that can never be shared under
       k-anonymous promotion, so placing it early only truncates sharing.
    2. *Canonical continuation.*  Remaining popular documents (decayed
       popularity >= ``min_pop``) that keep the audience >= ``kanon`` follow in
       a request-independent order — breadth class, then popularity bucket,
       then chunk id — so new combinations of popular evidence converge on one
       layout that future walks will follow.
    3. *Edge placement of relevance.*  Everything else is appended in
       *ascending* relevance, so the most relevant non-shared document sits
       immediately before the question.  Relevant documents that are part of
       the shared head stay at the front.  Either way, relevant evidence lands
       at an edge of the context, never in the middle.

    ``acl_grouping=False`` is the ablation that drops the ACL-dependent rules
    (audience gating and breadth-first canonical order).
    """
    from .acl import PUBLIC

    def eff_after(head: Sequence[RetrievedDoc], d: RetrievedDoc):
        a = PUBLIC
        for x in head:
            a = a.intersect(x.acl)
        return a.intersect(d.acl)

    def eligible(head, d) -> bool:
        if not acl_grouping or kanon <= 1:
            return True
        return audience_size(eff_after(head, d)) >= kanon

    def node_score(node, d):
        distinct = len(node.users)
        return (distinct >= kanon, distinct, node.n, -d.acl.breadth if acl_grouping else 0)

    head, remaining = tree.walk(docs, score=node_score, eligible=eligible)

    def bucket(cid: str) -> int:
        return int(math.floor(math.log2(1.0 + popularity.score(cid))))

    hot = [d for d in remaining if popularity.score(d.chunk_id) >= min_pop]
    if acl_grouping:
        hot.sort(key=lambda d: (d.acl.breadth, d.acl.canonical() if d.acl.breadth == 1 else "", -bucket(d.chunk_id), d.chunk_id))
    else:
        hot.sort(key=lambda d: (-bucket(d.chunk_id), d.chunk_id))
    for d in hot:
        if eligible(head, d):
            head.append(d)
    placed = {d.chunk_id for d in head}
    cold = sorted((d for d in docs if d.chunk_id not in placed), key=lambda d: -d.rank)
    return head + cold


ORDERINGS = ("retrieval", "acl_aware", "reuse_aware", "tree", "tree_k", "lcr", "layout", "layout_noacl")


def apply_ordering(
    name: str,
    docs: Sequence[RetrievedDoc],
    popularity: PopularityTracker | None = None,
    min_pop: float = 1.0,
    max_shift: int | None = None,
    anchor_k: int = 0,
    tree: PrefixTree | None = None,
    audience_size=None,
    kanon: int = 1,
) -> list[RetrievedDoc]:
    popularity = popularity if popularity is not None else PopularityTracker()
    tree = tree if tree is not None else PrefixTree()
    if name == "retrieval":
        out = order_retrieval(docs)
    elif name == "acl_aware":
        out = order_acl_aware(docs, max_shift=max_shift)
    elif name == "reuse_aware":
        out = order_reuse_aware(docs, popularity, min_pop=min_pop, max_shift=max_shift)
    elif name == "tree":
        out = order_tree(docs, tree)
    elif name == "tree_k":
        out = order_tree(docs, tree, kanon=kanon, audience_size=audience_size, privacy_aware=True)
    elif name == "lcr":
        out = order_lcr(docs)
    elif name in ("layout", "layout_noacl"):
        out = order_layout(docs, tree, popularity, audience_size or (lambda a: 10 ** 9), kanon=kanon,
                           min_pop=min_pop, acl_grouping=(name == "layout"))
    else:
        raise ValueError(f"unknown ordering {name!r}")
    return apply_relevance_anchor(out, anchor_k)


@dataclass
class OrderingConfig:
    name: str = "retrieval"
    max_shift: int | None = None
    min_pop: float = 1.0
    anchor_k: int = 0
    popularity: PopularityTracker = field(default_factory=PopularityTracker)
    tree: PrefixTree = field(default_factory=PrefixTree)
    promotion: object = None          # privacy.PromotionTracker, created per run when kanon > 1
