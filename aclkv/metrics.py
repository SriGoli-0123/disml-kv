"""Aggregation helpers and Prometheus scraping for the vLLM server."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

_PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE naNinf]+)$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)


def parse_prometheus(text: str) -> dict:
    """Return ``{"counters": {name: summed value}, "cache_config": {label: value}}``."""
    counters: dict[str, float] = {}
    cache_cfg: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if name == "vllm:cache_config_info":
            for k, lv in _LABEL.findall(labels):
                cache_cfg[k] = lv
            continue
        counters[name] = counters.get(name, 0.0) + v
    return {"counters": counters, "cache_config": cache_cfg}


def counter_delta(before: dict, after: dict) -> dict[str, float]:
    b, a = before.get("counters", {}), after.get("counters", {})
    out = {}
    for k in set(a) | set(b):
        if k.endswith("_total") or k in COUNTERS:
            out[k] = a.get(k, 0.0) - b.get(k, 0.0)
    return out


def pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def latency_summary(values: Iterable[float]) -> dict:
    v = [x for x in values if x is not None and x == x]
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "mean": float(np.mean(v)),
        "p50": pct(v, 50),
        "p90": pct(v, 90),
        "p95": pct(v, 95),
        "p99": pct(v, 99),
        "max": float(np.max(v)),
    }
