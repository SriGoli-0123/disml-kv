#!/usr/bin/env python
"""Figures for the report (matplotlib).  Reads the simulator CSV and, when
present, the live benchmark JSONs.

    python scripts/plot_results.py --out-dir results/plots
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICIES = ["B0", "B1", "B2", "B3", "B4", "B5", "UB"]
STYLE = {"B0": ("#999999", "x"), "B1": ("#d62728", "s"), "B2": ("#ff7f0e", "^"), "B3": ("#1f77b4", "o"),
         "B4": ("#2ca02c", "D"), "B5": ("#9467bd", "*"), "UB": ("#000000", "+"), "UB-order": ("#444444", "1"), "UB-reuse": ("#777777", "2")}


def plot_sim(csv_path: Path, out_dir: Path):
    rows = list(csv.DictReader(open(csv_path)))
    mixes = sorted({r["acl_mix"] for r in rows})
    kvs = sorted({float(r["kv_gib"]) for r in rows})
    concs = sorted({int(r["concurrency"]) for r in rows})
    for mix in mixes:
        for c in concs:
            fig, axes = plt.subplots(1, len(kvs), figsize=(4.2 * len(kvs), 3.6), sharey=True)
            axes = [axes] if len(kvs) == 1 else list(axes)
            for ax, kv in zip(axes, kvs):
                for p in sorted({r["policy"] for r in rows}, key=lambda p: POLICIES.index(p) if p in POLICIES else 99):
                    sel = sorted([r for r in rows if r["acl_mix"] == mix and int(r["concurrency"]) == c
                                  and float(r["kv_gib"]) == kv and r["policy"] == p], key=lambda r: float(r["share_rate"]))
                    if not sel:
                        continue
                    col, mk = STYLE.get(p, ("#333", "."))
                    ax.plot([float(r["share_rate"]) for r in sel], [float(r["cached_pct"]) for r in sel],
                            marker=mk, color=col, label=p, linestyle="--" if p.startswith("UB") else "-")
                ax.set_title(f"KV = {kv:g} GiB")
                ax.set_xlabel("shared-document rate")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("cached prompt tokens (%)")
            axes[-1].legend(fontsize=8, ncol=2)
            fig.suptitle(f"Simulator: authorized cache reuse — ACL mix {mix}, concurrency {c}")
            fig.tight_layout()
            fig.savefig(out_dir / f"sim_cached_mix{mix.replace('/', '-')}_c{c}.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 3.6))
            for p in ["B1", "B3", "B5", "UB"]:
                ys = []
                for kv in kvs:
                    sel = [r for r in rows if r["acl_mix"] == mix and int(r["concurrency"]) == c
                           and float(r["kv_gib"]) == kv and r["policy"] == p and float(r["share_rate"]) == 0.5]
                    ys.append(int(sel[0]["evictions"]) if sel else None)
                if any(y is not None for y in ys):
                    col, mk = STYLE[p]
                    ax.plot(kvs, ys, marker=mk, color=col, label=p)
            ax.set_xlabel("KV budget (GiB)")
            ax.set_ylabel("evictions (blocks), share = 0.5")
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"sim_evictions_mix{mix.replace('/', '-')}_c{c}.png", dpi=150)
            plt.close(fig)


def plot_bench(bench_dir: Path, out_dir: Path):
    runs = []
    for f in sorted(bench_dir.glob("*.json")):
        d = json.load(open(f))
        if "summary" not in d or d.get("args", {}).get("tag"):
            continue
        runs.append((d["policy"]["name"], d["workload"]["config"]["share_rate"], float(d["args"].get("kv_gib") or 0),
                     d["args"]["concurrency"], d["summary"]))
    if not runs:
        return
    keys = sorted({(kv, c) for _, _, kv, c, _ in runs})
    for kv, c in keys:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for p in sorted({r[0] for r in runs}, key=lambda p: POLICIES.index(p) if p in POLICIES else 99):
            sel = sorted([r for r in runs if r[0] == p and r[2] == kv and r[3] == c], key=lambda r: r[1])
            if not sel:
                continue
            col, mk = STYLE.get(p, ("#333", "."))
            xs = [r[1] for r in sel]
            axes[0].plot(xs, [r[4]["ttft"]["p50"] * 1000 for r in sel], marker=mk, color=col, label=p)
            axes[1].plot(xs, [r[4]["req_per_s"] for r in sel], marker=mk, color=col, label=p)
            axes[2].plot(xs, [r[4]["qa"].get("f1", 0) for r in sel], marker=mk, color=col, label=p)
        for ax, yl in zip(axes, ["TTFT p50 (ms)", "requests / s", "QA F1"]):
            ax.set_xlabel("shared-document rate")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"Live vLLM: KV = {kv:g} GiB, concurrency {c}")
        fig.tight_layout()
        fig.savefig(out_dir / f"bench_kv{kv:g}_c{c}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", default="results/sim/matrix/summary.csv")
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--out-dir", default="results/plots")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.sim_csv).exists():
        plot_sim(Path(args.sim_csv), out)
    if Path(args.bench_dir).exists():
        plot_bench(Path(args.bench_dir), out)
    print("plots in", out)


if __name__ == "__main__":
    main()
