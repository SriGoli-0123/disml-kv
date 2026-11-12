"""Aggregate simulator sweeps across seeds / user counts (consistency check)."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict

_NAME = re.compile(r"w_share([\d.]+)_mix-(.+?)_u(\d+)_s(\d+)$")


def parse_workload_name(name: str) -> dict:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"unrecognised workload name: {name!r}")
    share, mix, users, seed = m.groups()
    return {"share": float(share), "mix": mix, "users": int(users), "seed": int(seed)}


def aggregate(rows, metrics=("cached_pct", "evictions")) -> list[dict]:
    """Group simulator rows by (users, mix, share, policy, kv, concurrency) and
    summarise each metric over seeds. Security counters are summed."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        w = parse_workload_name(r["workload"])
        key = (w["users"], w["mix"], w["share"], r["policy"], float(r["kv_gib"]), int(r["concurrency"]))
        groups[key].append({**r, "_seed": w["seed"]})
    out = []
    for key, rs in sorted(groups.items()):
        users, mix, share, policy, kv, conc = key
        rec = {"users": users, "mix": mix, "share": share, "policy": policy, "kv_gib": kv,
               "concurrency": conc, "n_seeds": len({r["_seed"] for r in rs})}
        for m in metrics:
            vals = [float(r[m]) for r in rs]
            rec[f"{m}_mean"] = statistics.fmean(vals)
            rec[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            rec[f"{m}_min"], rec[f"{m}_max"] = min(vals), max(vals)
        rec["unauthorized_hits"] = sum(int(float(r["unauthorized_hits"])) for r in rs)
        rec["probe_unauthorized_hits"] = sum(int(float(r["probe_unauthorized_hits"])) for r in rs)
        out.append(rec)
    return out
