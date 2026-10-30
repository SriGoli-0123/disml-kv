#!/usr/bin/env python
"""Analyses of the committed v1 live A100 results (results/bench, 192 runs).

1. Simulator fidelity: rebuild every run's prompts locally (checking that the
   evidence order is identical to what was sent), replay them in the
   simulator with the live completion lengths, compare cached-token %.
2. TTFT model: how much of live TTFT is explained by recomputed prompt tokens.
3. Answer quality: paired, question-clustered comparisons between orderings.
4. Effect of the v2 query-tail fix on the v1 workload (simulated).

    python scripts/analyze_live_v1.py --out results/v2/analysis_live_v1.json
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.pipeline import BuildSettings, build_prompts, load_workload_and_corpus, make_builder  # noqa: E402
from aclkv.policies import get_policy, with_options  # noqa: E402
from aclkv.qa_eval import score_one  # noqa: E402
from aclkv.simulator import simulate  # noqa: E402
from aclkv.stats import paired_report  # noqa: E402
from aclkv.tokenizer import load_tokenizer  # noqa: E402

POLS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "UB"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--workload-dir", default="data/workloads")
    ap.add_argument("--corpus", default="data/prepared/corpus.jsonl")
    ap.add_argument("--out", default="results/v2/analysis_live_v1.json")
    args = ap.parse_args()

    runs = []
    for f in sorted(glob.glob(f"{args.bench_dir}/*.json")):
        d = json.load(open(f))
        runs.append(d)
    print(f"{len(runs)} live runs")
    tok = load_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    settings = BuildSettings()
    cache_prompts: dict[tuple, list] = {}
    fidelity = []
    order_mismatch = 0
    for d in runs:
        pol = d["policy"]["name"]
        wname = Path(d["workload"]["path"]).stem
        key = (wname, pol)
        if key not in cache_prompts:
            w, corpus = load_workload_and_corpus(Path(args.workload_dir) / f"{wname}.json", args.corpus)
            b = make_builder(w, settings, tok)
            cache_prompts[key] = (w, build_prompts(w, corpus, pol, b, settings))
        w, prompts = cache_prompts[key]
        recs = {r["req_id"]: r for r in d["records"]}
        order_mismatch += sum(1 for p in prompts if p.ordered_chunk_ids != recs[p.req_id]["ordered_chunk_ids"])
        out_len = {p.req_id: (recs[p.req_id]["completion_tokens"] or 48) for p in prompts}
        _, s = simulate(prompts, d["server"]["num_gpu_blocks"], w.directory, 16,
                        max_concurrency=d["args"]["concurrency"], output_tokens=48, output_lengths=out_len)
        live = d["summary"]["cached_pct"]
        fidelity.append({"workload": wname, "policy": pol, "kv": d["args"]["kv_gib"], "c": d["args"]["concurrency"],
                         "live_cached_pct": live, "sim_cached_pct": s.cached_pct, "abs_err": abs(live - s.cached_pct)})
    errs = np.array([x["abs_err"] for x in fidelity])
    fid = {"n_runs": len(fidelity), "mean_abs_err_pp": float(errs.mean()), "max_abs_err_pp": float(errs.max()),
           "p95_abs_err_pp": float(np.percentile(errs, 95)), "evidence_order_mismatches": order_mismatch,
           "pearson_r": float(np.corrcoef([x["live_cached_pct"] for x in fidelity],
                                          [x["sim_cached_pct"] for x in fidelity])[0, 1])}
    print("fidelity:", json.dumps(fid))

    # ---- TTFT model: TTFT p50 vs mean recomputed tokens per request, per (kv, c)
    ttft = []
    for d in runs:
        s = d["summary"]
        ttft.append({"kv": d["args"]["kv_gib"], "c": d["args"]["concurrency"], "policy": d["policy"]["name"],
                     "share": d["workload"]["config"]["share_rate"],
                     "recomp_per_req": s["recomputed_tokens"] / s["ok"], "ttft_p50_ms": s["ttft"]["p50"] * 1000,
                     "ttft_mean_ms": s["ttft"]["mean"] * 1000, "rps": s["req_per_s"], "cached_pct": s["cached_pct"]})
    fits = {}
    for c in sorted({x["c"] for x in ttft}):
        sel = [x for x in ttft if x["c"] == c]
        xs = np.array([x["recomp_per_req"] for x in sel])
        for ycol in ("ttft_mean_ms", "ttft_p50_ms"):
            ys = np.array([x[ycol] for x in sel])
            A = np.vstack([xs, np.ones_like(xs)]).T
            coef, res, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            r2 = 1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)
            fits[f"c{c}_{ycol}"] = {"ms_per_1k_recomputed_tokens": float(coef[0] * 1000), "intercept_ms": float(coef[1]),
                                    "r2": float(r2), "n": len(sel)}
    print("ttft fits:", json.dumps(fits))

    # headline live numbers at 2 GiB, C=16
    head = defaultdict(dict)
    for x in ttft:
        if x["kv"] == 2.0 and x["c"] == 16:
            head[x["share"]][x["policy"]] = {k: x[k] for k in ("ttft_p50_ms", "rps", "cached_pct")}

    # ---- paired quality: per share (2 GiB, C=16 runs), B3 reference
    quality = {}
    by = {(Path(d["workload"]["path"]).stem, d["policy"]["name"], d["args"]["kv_gib"], d["args"]["concurrency"]): d for d in runs}
    for wname in sorted({k[0] for k in by}):
        ref = by[(wname, "B3", 2.0, 16)]
        rrec = {r["req_id"]: r for r in ref["records"]}
        ids = sorted(rrec)
        quality[wname] = {"B3_em": ref["summary"]["qa"]["em"], "B3_f1": ref["summary"]["qa"]["f1"]}
        for pol in ("B4", "B5", "B6"):
            o = by[(wname, pol, 2.0, 16)]
            orec = {r["req_id"]: r for r in o["records"]}
            a = [score_one(rrec[i]["text"], rrec[i]["answer"]) for i in ids]
            bb = [score_one(orec[i]["text"], orec[i]["answer"]) for i in ids]
            quality[wname][pol] = paired_report(a, bb, [rrec[i]["qid"] for i in ids])
        # numeric noise floor: B0 and B3 send identical token sequences
        b0 = {r["req_id"]: r for r in by[(wname, "B0", 2.0, 16)]["records"]}
        quality[wname]["noise_floor_answers_differing_B0_vs_B3"] = sum(1 for i in ids if b0[i]["text"] != rrec[i]["text"])
        q6 = quality[wname]["B6"]["f1"]
        print(f"{wname}: B6 vs B3 F1 {q6['diff']:+.4f} [{q6['ci_low']:+.4f},{q6['ci_high']:+.4f}] p_boot={q6['p_boot']:.3f} "
              f"clusters={q6['n_clusters']}")

    # ---- what the v2 tail fix costs on the v1 workload (simulated, 2 GiB, C=16)
    tailfix = {}
    for wname in sorted({k[0] for k in by}):
        w, corpus = load_workload_and_corpus(Path(args.workload_dir) / f"{wname}.json", args.corpus)
        b = make_builder(w, settings, tok)
        for pol in ("B3", "B6"):
            v1 = get_policy(pol)
            v2 = with_options(v1, name=pol + "-privtail", tail_scope="private", legacy=False)
            r = {}
            for tag, P in (("v1", v1), ("v2", v2)):
                _, s = simulate(build_prompts(w, corpus, P, b, settings), 2340, w.directory, 16, 16, 48)
                r[tag] = s.cached_pct
            tailfix.setdefault(wname, {})[pol] = r
    print("tail fix:", json.dumps(tailfix))

    out = {"fidelity": fid, "fidelity_rows": fidelity, "ttft_fits": fits, "ttft_rows": ttft,
           "headline_live_kv2_c16": head, "quality": quality, "tail_fix_cost": tailfix}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
