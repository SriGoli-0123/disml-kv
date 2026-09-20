#!/usr/bin/env python
"""Turn simulator CSVs and live-bench JSONs into Markdown tables.

    python scripts/summarize.py                                  # results/summary.md
    python scripts/summarize.py --sim-csv results/sim/matrix/summary.csv --bench-dir results/bench
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

POLICY_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "UB", "UB-order", "UB-reuse"]


def pkey(p):
    return POLICY_ORDER.index(p) if p in POLICY_ORDER else 99


def sim_tables(csv_path: Path) -> str:
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return ""
    out = ["## Simulator (deterministic cache model, no GPU)\n"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["acl_mix"], r["kv_gib"], r["concurrency"])].append(r)
    for (mix, kv, c), rs in sorted(groups.items()):
        shares = sorted({float(r["share_rate"]) for r in rs})
        pols = sorted({r["policy"] for r in rs}, key=pkey)
        out.append(f"### ACL mix {mix} (public/group/private), KV = {kv} GiB, concurrency = {c}\n")
        out.append("Cached prompt tokens (%) by shared-document rate:\n")
        out.append("| policy | " + " | ".join(f"share {s:.2f}" for s in shares) + " | probe leaks |")
        out.append("|---|" + "---|" * (len(shares) + 1))
        for p in pols:
            cells, leaks = [], 0
            for s in shares:
                m = [r for r in rs if r["policy"] == p and float(r["share_rate"]) == s]
                cells.append(f"{float(m[0]['cached_pct']):.1f}" if m else "-")
                leaks += sum(int(r["probe_unauthorized_hits"]) for r in m)
            out.append(f"| {p} | " + " | ".join(cells) + f" | {leaks} |")
        out.append("\nEvictions (blocks) by shared-document rate:\n")
        out.append("| policy | " + " | ".join(f"share {s:.2f}" for s in shares) + " |")
        out.append("|---|" + "---|" * len(shares))
        for p in pols:
            cells = []
            for s in shares:
                m = [r for r in rs if r["policy"] == p and float(r["share_rate"]) == s]
                cells.append(m[0]["evictions"] if m else "-")
            out.append(f"| {p} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def bench_tables(bench_dir: Path) -> str:
    files = sorted(bench_dir.glob("*.json"))
    if not files:
        return ""
    runs = []
    for f in files:
        d = json.load(open(f))
        if "summary" not in d or "policy" not in d:
            continue
        s = d["summary"]
        sim = d.get("simulator") or {}
        runs.append({
            "file": f.name, "policy": d["policy"]["name"], "workload": Path(d["workload"]["path"]).stem,
            "share": d["workload"]["config"]["share_rate"], "kv": d["args"].get("kv_gib"),
            "c": d["args"]["concurrency"], "tag": d["args"].get("tag", ""),
            "cached_pct": s["cached_pct"], "ttft_p50": s["ttft"].get("p50"), "ttft_p95": s["ttft"].get("p95"),
            "e2e_p50": s["e2e"].get("p50"), "rps": s["req_per_s"], "ptps": s["prompt_tok_per_s"],
            "em": s["qa"].get("em"), "f1": s["qa"].get("f1"), "errors": s["errors"],
            "evictions": sim.get("evictions"), "unauth": sim.get("unauthorized_hits"),
            "preempt": s.get("prom_delta", {}).get("vllm:num_preemptions_total"),
        })
    out = ["## Live vLLM benchmark (A100)\n"]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["workload"], r["kv"], r["c"])].append(r)
    for (w, kv, c), rs in sorted(groups.items(), key=lambda kv_: (kv_[0][0], float(kv_[0][1] or 0), kv_[0][2])):
        out.append(f"### {w}, KV = {kv} GiB, concurrency = {c}\n")
        out.append("| policy | cached % | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | req/s | prompt tok/s | EM | F1 | sim evictions | sim unauth | preemptions | errors |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda r: (pkey(r["policy"]), r["tag"])):
            name = r["policy"] + (f" ({r['tag']})" if r["tag"] else "")
            fmt = lambda v, m=1.0, d=1: ("-" if v is None else f"{v * m:.{d}f}")
            out.append(f"| {name} | {fmt(r['cached_pct'])} | {fmt(r['ttft_p50'], 1000, 0)} | {fmt(r['ttft_p95'], 1000, 0)} | "
                       f"{fmt(r['e2e_p50'], 1000, 0)} | {fmt(r['rps'], 1, 2)} | {fmt(r['ptps'], 1, 0)} | {fmt(r['em'], 1, 3)} | "
                       f"{fmt(r['f1'], 1, 3)} | {r['evictions'] if r['evictions'] is not None else '-'} | "
                       f"{r['unauth'] if r['unauth'] is not None else '-'} | {fmt(r['preempt'], 1, 0)} | {r['errors']} |")
        out.append("")
    return "\n".join(out)


def probe_tables(results_dir: Path) -> str:
    files = sorted(results_dir.glob("security_probe*.json"))
    if not files:
        return ""
    out = ["## Security probe (attacker replays a victim's tokens without the keyed scope ids)\n",
           "| file | policy | victim cold TTFT p50 (ms) | victim warm TTFT p50 (ms) | attacker no_salt hits | no_salt cached tokens | no_salt TTFT p50 (ms) | forged hits | stolen-salt hits (control) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        d = json.load(open(f))
        for pol, r in d["policies"].items():
            s = r["summary"]
            g = lambda k, kk="attacker_hits": (s.get(k, {}) or {}).get(kk, "-")
            ns = s.get("no_salt", {})
            out.append(f"| {f.name} | {pol} | {s['victim_cold_ttft'].get('p50', 0) * 1000:.0f} | {s['victim_warm_ttft'].get('p50', 0) * 1000:.0f} | "
                       f"{g('no_salt')}/{ns.get('n', '-')} | {ns.get('cached_tokens_mean', 0):.0f} | {ns.get('ttft', {}).get('p50', 0) * 1000:.0f} | "
                       f"{g('forged')} | {g('stolen')} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", default="results/sim/matrix/summary.csv")
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()
    parts = ["# Results summary\n"]
    if Path(args.sim_csv).exists():
        parts.append(sim_tables(Path(args.sim_csv)))
    if Path(args.bench_dir).exists():
        parts.append(bench_tables(Path(args.bench_dir)))
    parts.append(probe_tables(Path(args.results_dir)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(p for p in parts if p)
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
