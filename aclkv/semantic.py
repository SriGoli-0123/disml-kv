"""Semantic answer similarity: a second QA-quality metric next to EM/F1.

EM/F1 count a correct paraphrase ("the Pentagon" vs "Pentagon building") as
wrong. This module scores answers by embedding cosine similarity instead.

Two encoders are provided:

* ``SentenceTransformerEncoder`` - the real metric (neural sentence embeddings).
  Needs ``pip install -e ".[semantic]"`` and a one-off model download.
* ``CharNgramEncoder`` - a dependency-free stand-in (hashed character n-grams).
  It is NOT a neural embedding; it exists so the pipeline and tests run offline.
  Always label results produced with it as such.

Yes/no answers are handled like the HotpotQA F1 convention: embeddings place
"yes" and "no" close together, so a yes/no mismatch is forced to similarity 0.
"""

from __future__ import annotations

import zlib
from typing import Iterable, Protocol, Sequence

import numpy as np

from .qa_eval import clean_prediction, exact_match, normalize_answer

_YESNO = ("yes", "no", "noanswer")


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # rows L2-normalised
        ...


def _normalise_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class CharNgramEncoder:
    """Hashed character n-gram vectors (offline stand-in, not neural)."""

    name = "char-ngram"

    def __init__(self, dim: int = 2048, ngrams: tuple[int, ...] = (2, 3, 4)):
        self.dim, self.ngrams = dim, ngrams

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f" {normalize_answer(t)} "
            for n in self.ngrams:
                for j in range(len(s) - n + 1):
                    out[i, zlib.crc32(s[j : j + n].encode()) % self.dim] += 1.0
        return _normalise_rows(out)


class SentenceTransformerEncoder:
    """Neural sentence embeddings via the sentence-transformers package."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise SystemExit(
                'sentence-transformers is not installed. Run: pip install -e ".[semantic]"\n'
                "(or pass --backend char for the offline stand-in)"
            ) from e
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def _is_yesno(s: str) -> bool:
    toks = normalize_answer(s).split()
    return bool(toks) and toks[0] in _YESNO


class SemanticScorer:
    """Cosine similarity between a cleaned prediction and the gold answer."""

    def __init__(self, encoder: Encoder, threshold: float = 0.8):
        self.encoder, self.threshold = encoder, threshold
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: Iterable[str]) -> None:
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            for t, v in zip(missing, self.encoder.encode(missing)):
                self._cache[t] = v

    def similarity(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """``pairs``: (raw prediction text, gold answer). Returns values in [0, 1]."""
        cleaned = [(clean_prediction(p), g) for p, g in pairs]
        self._embed([c for c, _ in cleaned])
        self._embed([g for _, g in cleaned])
        sims = []
        for pred, gold in cleaned:
            if not pred or not normalize_answer(pred):
                sims.append(0.0)
            elif (_is_yesno(pred) or _is_yesno(gold)) and normalize_answer(pred) != normalize_answer(gold):
                sims.append(0.0)
            else:
                sims.append(float(np.clip(np.dot(self._cache[pred], self._cache[gold]), 0.0, 1.0)))
        return sims

    def evaluate(self, pairs: Sequence[tuple[str, str]]) -> dict:
        """Mean similarity, share counted correct at ``threshold``, and the share
        that EM misses but the semantic check accepts ("rescued")."""
        pairs = list(pairs)
        if not pairs:
            return {"n": 0}
        sims = self.similarity(pairs)
        match = [s >= self.threshold for s in sims]
        rescued = [m and not exact_match(clean_prediction(p), g) for m, (p, g) in zip(match, pairs)]
        n = len(pairs)
        return {
            "n": n,
            "sem_sim": sum(sims) / n,
            "sem_match": sum(match) / n,
            "sem_rescued": sum(rescued) / n,
        }
