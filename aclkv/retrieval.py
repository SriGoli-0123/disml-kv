"""Corpus-level retrieval for the workload generator.

The v1 workload re-ranked HotpotQA's fixed 10 paragraphs per question, and
manufactured cross-user overlap by injecting random "hot" paragraphs.  v2
retrieves top-k from the *whole* corpus, so documents are shared across
requests only when the retriever genuinely returns them for related queries.

* ``SparseBM25``   Okapi BM25 over a scipy CSC term-document matrix (exact,
                   ~ms per query on 10^5 passages).
* ``DenseIndex``   optional sentence-transformers bi-encoder (e.g. bge-small).
* ``rrf``          reciprocal-rank fusion for hybrid retrieval.

Retrieval is *ACL-filtered*: ``search(..., allowed=fn)`` only returns
passages the requester may read, which is the proposal's assumption that the
retriever already removes inaccessible evidence.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is are was were be been being which who whom whose what when "
    "where why how that this these those it its as into than then there their they he she his her him do does did "
    "has have had not no yes".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseBM25:
    """Okapi BM25 (k1, b) with precomputed per-(doc, term) weights."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        from scipy import sparse

        self.ids = list(ids)
        self.pos = {c: i for i, c in enumerate(self.ids)}
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        dls = np.zeros(len(texts), dtype=np.float64)
        for i, t in enumerate(texts):
            cnt = Counter(tokenize(t))
            dls[i] = sum(cnt.values())
            for w, tf in cnt.items():
                j = vocab.setdefault(w, len(vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(tf)
        self.vocab = vocab
        n = len(texts)
        tf = np.asarray(tfs, dtype=np.float64)
        rows_a = np.asarray(rows)
        df = np.bincount(np.asarray(cols), minlength=len(vocab)).astype(np.float64)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))          # Lucene/BM25+ style, always >= 0
        avgdl = dls.mean() if n else 1.0
        norm = k1 * (1.0 - b + b * dls[rows_a] / avgdl)
        w = idf[np.asarray(cols)] * tf * (k1 + 1.0) / (tf + norm)
        self._m = sparse.csc_matrix((w, (rows_a, np.asarray(cols))), shape=(n, len(vocab)))

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[w] for w in set(tokenize(query)) if w in self.vocab]
        if not cols:
            return np.zeros(len(self.ids))
        return np.asarray(self._m[:, cols].sum(axis=1)).ravel()

    def score_ids(self, query: str, ids: Sequence[str]) -> dict[str, float]:
        s = self.scores(query)
        return {c: float(s[self.pos[c]]) for c in ids}


class DenseIndex:
    """Bi-encoder index; embeddings are cached to ``cache_path`` (.npy)."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], model: str = "BAAI/bge-small-en-v1.5",
                 cache_path: str | Path | None = None, batch_size: int = 128,
                 query_prefix: str = "Represent this sentence for searching relevant passages: "):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.ids = list(ids)
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model)
        emb = None
        if cache_path and Path(cache_path).exists():
            emb = np.load(cache_path)
            if emb.shape[0] != len(self.ids):
                emb = None
        if emb is None:
            emb = self._model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            if cache_path:
                np.save(cache_path, emb)
        self._emb = emb

    def scores(self, query: str) -> np.ndarray:
        q = self._model.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self._emb @ q.astype(np.float32)


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion of several ranked index lists."""
    out: dict[int, float] = {}
    for r in rankings:
        for pos, idx in enumerate(r):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return out


def query_dropout(query: str, rate: float, rng) -> str:
    """Drop a fraction of non-stopword terms: a cheap proxy for paraphrase-induced
    retrieval variance (the *prompt* still contains the original question)."""
    if rate <= 0:
        return query
    toks = tokenize(query)
    content = [t for t in toks if t not in STOPWORDS]
    if len(content) <= 2:
        return query
    n_drop = min(len(content) - 2, int(round(rate * len(content))))
    drop = set(rng.sample(range(len(content)), n_drop))
    kept = [t for i, t in enumerate(content) if i not in drop]
    return " ".join(kept)


class Retriever:
    """BM25, dense, or hybrid (RRF) top-k with an authorization predicate."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], mode: str = "bm25",
                 dense_model: str = "BAAI/bge-small-en-v1.5", dense_cache: str | Path | None = None):
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(mode)
        self.mode = mode
        self.ids = list(ids)
        self.bm25 = SparseBM25(ids, texts)
        self.dense = DenseIndex(ids, texts, dense_model, dense_cache) if mode in ("dense", "hybrid") else None

    def _ranked(self, query: str, depth: int) -> list[tuple[int, float]]:
        if self.mode == "bm25":
            s = self.bm25.scores(query)
        elif self.mode == "dense":
            s = self.dense.scores(query)
        else:
            sb, sd = self.bm25.scores(query), self.dense.scores(query)
            kb = np.argpartition(-sb, min(depth, len(sb) - 1))[:depth]
            kd = np.argpartition(-sd, min(depth, len(sd) - 1))[:depth]
            fused = rrf([sorted(kb, key=lambda i: -sb[i]), sorted(kd, key=lambda i: -sd[i])])
            return sorted(fused.items(), key=lambda kv: -kv[1])
        depth = min(depth, len(s))
        top = np.argpartition(-s, depth - 1)[:depth]
        return sorted(((int(i), float(s[i])) for i in top), key=lambda kv: -kv[1])

    def search(self, query: str, k: int, allowed: Callable[[str], bool] | None = None,
               depth: int | None = None) -> list[tuple[str, float]]:
        depth = depth or max(50, 8 * k)
        while True:
            out = []
            for i, sc in self._ranked(query, depth):
                if sc <= 0:
                    continue
                cid = self.ids[i]
                if allowed is None or allowed(cid):
                    out.append((cid, sc))
                    if len(out) == k:
                        return out
            if depth >= len(self.ids):
                return out
            depth = min(len(self.ids), depth * 4)


def idf_weighted_overlap(a: Sequence[str], b: Sequence[str]) -> float:  # pragma: no cover - diagnostics
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def ndcg_like_gold_recall(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    g = set(gold)
    return sum(1 for c in retrieved if c in g) / max(1, len(g))


__all__ = ["SparseBM25", "DenseIndex", "Retriever", "rrf", "query_dropout", "tokenize", "ndcg_like_gold_recall"]

_ = math  # keep import for downstream numeric helpers
