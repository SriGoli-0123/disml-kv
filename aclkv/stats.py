"""Paired significance tests for answer-quality comparisons.

Requests repeat questions (Zipf popularity), so per-request outcomes are not
independent.  ``cluster_bootstrap`` resamples *questions* (clusters) with
replacement and averages per-question mean differences, which gives honest
confidence intervals.  ``mcnemar_exact`` and ``sign_test`` are exact two-sided
binomial tests on discordant pairs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_exact(b01: int, b10: int) -> float:
    """Exact McNemar p-value from discordant counts (a wrong/b right, a right/b wrong)."""
    return _binom_two_sided(min(b01, b10), b01 + b10)


def sign_test(diffs: Sequence[float], eps: float = 1e-9) -> tuple[int, int, float]:
    up = sum(1 for d in diffs if d > eps)
    dn = sum(1 for d in diffs if d < -eps)
    return up, dn, _binom_two_sided(min(up, dn), up + dn)


def cluster_bootstrap(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], n_boot: int = 4000,
                      seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of (b - a) with a cluster-bootstrap CI; clusters = question ids."""
    by = defaultdict(list)
    for x, y, c in zip(a, b, clusters):
        by[c].append(y - x)
    keys = sorted(by)
    means = np.array([np.mean(by[k]) for k in keys])
    sizes = np.array([len(by[k]) for k in keys], dtype=float)
    point = float(np.sum(means * sizes) / np.sum(sizes))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = (means[idx] * sizes[idx]).sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value for H0: mean difference == 0
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi), "p_boot": float(min(1.0, p)),
            "n_items": len(a), "n_clusters": len(keys)}


def paired_report(a_scores: Sequence[dict], b_scores: Sequence[dict], clusters: Sequence[str], seed: int = 0) -> dict:
    """``*_scores``: per-request dicts with keys em, f1 (same request order)."""
    out = {}
    for m in ("em", "f1"):
        a = [s[m] for s in a_scores]
        b = [s[m] for s in b_scores]
        out[m] = cluster_bootstrap(a, b, clusters, seed=seed)
        up, dn, p = sign_test([y - x for x, y in zip(a, b)])
        out[m].update({"b_better": up, "a_better": dn, "p_sign": p})
    b01 = sum(1 for x, y in zip(a_scores, b_scores) if x["em"] < y["em"])
    b10 = sum(1 for x, y in zip(a_scores, b_scores) if x["em"] > y["em"])
    out["em"]["p_mcnemar"] = mcnemar_exact(b01, b10)
    return out
