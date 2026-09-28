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
    """

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.counts: dict[str, float] = {}
        self._steps = 0

    def observe(self, chunk_ids: Sequence[str]) -> None:
        self._steps += 1
        if self.decay < 1.0:
            for k in list(self.counts):
                self.counts[k] *= self.decay
                if self.counts[k] < 1e-6:
                    del self.counts[k]
        for c in chunk_ids:
            self.counts[c] = self.counts.get(c, 0.0) + 1.0

    def score(self, chunk_id: str) -> float:
        return self.counts.get(chunk_id, 0.0)

    def to_json(self) -> dict:
        return {"decay": self.decay, "counts": self.counts}


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


ORDERINGS = ("retrieval", "acl_aware", "reuse_aware")


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


def apply_ordering(
    name: str,
    docs: Sequence[RetrievedDoc],
    popularity: PopularityTracker | None = None,
    min_pop: float = 1.0,
    max_shift: int | None = None,
    anchor_k: int = 0,
) -> list[RetrievedDoc]:
    if name == "retrieval":
        out = order_retrieval(docs)
    elif name == "acl_aware":
        out = order_acl_aware(docs, max_shift=max_shift)
    elif name == "reuse_aware":
        if popularity is None:
            popularity = PopularityTracker()
        out = order_reuse_aware(docs, popularity, min_pop=min_pop, max_shift=max_shift)
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
