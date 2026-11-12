"""Summarise a simulator sweep that covers several seeds and/or user counts.

    python scripts/aggregate_seeds.py --sim-csv results/sim/extra/summary.csv

Writes results/sim_seed_consistency.csv and .md: cached-token % as mean +/- std
over seeds for each policy and share rate, plus a security check (every policy
except the insecure upper bound UB must show zero unauthorized hits).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.aggregate import aggregate  # noqa: E402
from aclkv.policies import POLICIES  # noqa: E402

POLICY_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "UB", "UB-order", "UB-reuse", "UB-anchor"]


def is_secure(policy: str) -> bool:
    """Use the repo's own flag; the UB-* variants are deliberately insecure."""
    return POLICIES[policy].secure if policy in POLICIES else True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", default="results/sim/extra/summary.csv")
    ap.add_argument("--concurrency", type=int, default=16, help="concurrency shown in the markdown tables")
    ap.add_argument("--out-prefix", default="results/sim_seed_consistency")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.sim_csv)))
    if not rows:
        sys.exit(f"{args.sim_csv} is empty")
    agg = aggregate(rows)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)

    leaks = sum(a["unauthorized_hits"] + a["probe_unauthorized_hits"] for a in agg if is_secure(a["policy"]))
    lines = ["# Consistency across seeds and user counts (simulator)", "",
             f"{len(rows)} simulator rows; seeds per cell: {sorted({a['n_seeds'] for a in agg})}.",
             f"Unauthorized hits for all secure policies (everything except the insecure UB* variants): **{leaks}** (must be 0).", ""]
    cells = defaultdict(dict)
    for a in agg:
        if a["concurrency"] == args.concurrency:
            cells[(a["users"], a["mix"], a["kv_gib"])][(a["policy"], a["share"])] = a
    for (users, mix, kv), d in sorted(cells.items()):
        shares = sorted({s for _, s in d})
        pols = [p for p in POLICY_ORDER if any(k[0] == p for k in d)]
        lines += [f"## {users} users, mix {mix}, KV {kv:g} GiB, concurrency {args.concurrency} - cached prompt tokens (%), mean +/- std over seeds", "",
                  "| policy | " + " | ".join(f"share {s:g}" for s in shares) + " |", "|---|" + "---|" * len(shares)]
        for p in pols:
            cols = []
            for s in shares:
                a = d.get((p, s))
                cols.append(f"{a['cached_pct_mean']:.1f} +/- {a['cached_pct_std']:.1f} (n={a['n_seeds']})" if a else "-")
            lines.append(f"| {p} | " + " | ".join(cols) + " |")
        lines.append("")
    Path(f"{prefix}.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"\nwrote {prefix}.csv and {prefix}.md")
    if leaks:
        sys.exit("SECURITY CHECK FAILED: a secure policy has unauthorized hits")


if __name__ == "__main__":
    main()
