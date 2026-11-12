#!/usr/bin/env python
"""Summarise the v2 live runs (results/v2_live/<model>_kv<gib>/{closed,open,quality}).

* closed loop: TTFT p50/p95, req/s, cached % per (model, KV, workload, policy)
* open loop:   TTFT vs arrival rate, SLO attainment and goodput
* quality:     paired, question-clustered comparisons against ACLScope
               (retrieval order) on EM / F1 / LLM-judge accuracy

    python scripts/analyze_live_v2.py --out results/v2_live/summary.json
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.qa_eval import score_one  # noqa: E402
from aclkv.stats import cluster_bootstrap, paired_report  # noqa: E402


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            print("skip (corrupt)", f)
            continue
        if "summary" in d:
            d["_file"] = f
            out.append(d)
    return out


def cell(d):
    return (d["server"]["model"].split("/")[-1], d["args"]["kv_gib"], Path(d["workload"]["path"]).stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/v2_live")
    ap.add_argument("--reference", default="ACLScope")
    ap.add_argument("--out", default="results/v2_live/summary.json")
    args = ap.parse_args()
    out = {"closed": [], "open": [], "quality": []}
    for d in load(f"{args.root}/*/closed/*.json"):
        s = d["summary"]
        m, kv, wl = cell(d)
        out["closed"].append({"model": m, "kv": kv, "workload": wl, "policy": d["policy"]["name"],
                              "cached_pct": s["cached_pct"], "ttft_p50_ms": s["ttft"]["p50"] * 1000,
                              "ttft_p95_ms": s["ttft"]["p95"] * 1000, "req_per_s": s["req_per_s"],
                              "em": s["qa"].get("em"), "f1": s["qa"].get("f1"), "errors": s["errors"]})
    for d in load(f"{args.root}/*/open/*.json"):
        s = d["summary"]
        m, kv, wl = cell(d)
        gp = s.get("goodput") or {}
        out["open"].append({"model": m, "kv": kv, "workload": wl, "policy": d["policy"]["name"], "rate": d["args"]["rate"],
                            "ttft_p50_ms": s["ttft"]["p50"] * 1000, "ttft_p95_ms": s["ttft"]["p95"] * 1000,
                            "slo_attained": gp.get("attained"), "goodput_req_per_s": gp.get("goodput_req_per_s"),
                            "cached_pct": s["cached_pct"]})
    q = defaultdict(dict)
    for d in load(f"{args.root}/*/quality/*.json"):
        q[cell(d)][d["policy"]["name"]] = d
    for key, pols in sorted(q.items()):
        if args.reference not in pols:
            continue
        ref = {r["req_id"]: r for r in pols[args.reference]["records"] if r.get("ok")}
        for name, d in pols.items():
            recs = {r["req_id"]: r for r in d["records"] if r.get("ok")}
            ids = sorted(set(ref) & set(recs))
            gold = lambda r: r.get("answers") or r["answer"]
            a = [score_one(ref[i]["text"], gold(ref[i])) for i in ids]
            b = [score_one(recs[i]["text"], gold(recs[i])) for i in ids]
            row = {"model": key[0], "kv": key[1], "workload": key[2], "policy": name, "n": len(ids),
                   "em": sum(x["em"] for x in b) / max(1, len(b)), "f1": sum(x["f1"] for x in b) / max(1, len(b))}
            if name != args.reference:
                row["vs_ref"] = paired_report(a, b, [ref[i]["qid"] for i in ids])
            jr, jd = pols[args.reference]["summary"].get("judge"), d["summary"].get("judge")
            if jd and jd.get("per_request"):
                row["judge_acc"] = jd["accuracy"]
                if jr and jr.get("per_request") and name != args.reference:
                    pa = [float(bool(jr["per_request"].get(i))) for i in ids]
                    pb = [float(bool(jd["per_request"].get(i))) for i in ids]
                    row["judge_vs_ref"] = cluster_bootstrap(pa, pb, [ref[i]["qid"] for i in ids])
            out["quality"].append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"closed={len(out['closed'])} open={len(out['open'])} quality={len(out['quality'])} -> {args.out}")


if __name__ == "__main__":
    main()
