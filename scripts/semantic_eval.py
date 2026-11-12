"""Score every saved bench run with the semantic-similarity metric.

Reads results/bench/*.json (the per-request records already contain the model's
text and the gold answer), so it needs no GPU and no server.

    pip install -e ".[semantic]"
    python scripts/semantic_eval.py                     # neural embeddings (all-MiniLM-L6-v2)
    python scripts/semantic_eval.py --backend char      # offline stand-in, NOT neural
    python scripts/semantic_eval.py --bench-glob "results/v2_live/*/quality/*.json"   # v2 runs

Runs are paired by setting = (workload, load, KV budget, model), so v2 workloads
that share a config value are never mixed.  Gold answer aliases (MuSiQue) are
used when the records carry them; the best-matching alias counts.  The paired
comparison uses ``--baseline`` (default: B3 for v1 runs, ACLScope for v2 runs).

Writes results/semantic_qa_<backend>.csv (one row per run), .md (policy table +
paired comparison against B3) and _examples.csv (answers EM rejects but the
semantic check accepts, plus borderline cases) so the threshold can be eyeballed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.qa_eval import clean_prediction, evaluate, exact_match  # noqa: E402
from aclkv.semantic import CharNgramEncoder, SemanticScorer, SentenceTransformerEncoder  # noqa: E402

POLICY_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "UB",
                "NoCache", "PerUser", "Divergence", "Binary", "ACLScope", "ACLScope+Tree", "ACLKV", "Global", "Global+Tree"]


def paired_ci(diffs_by_qid: dict, n_boot: int = 2000, seed: int = 0):
    """Mean paired difference with a 95% bootstrap CI, resampling whole questions."""
    sums = np.array([sum(v) for v in diffs_by_qid.values()], dtype=float)
    cnts = np.array([len(v) for v in diffs_by_qid.values()], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return sums.sum() / cnts.sum(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--backend", choices=["st", "char"], default="st")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--baseline", default="auto",
                    help="policy for the paired comparison; auto = B3 (v1) or ACLScope (v2), i.e. plain retrieval order")
    ap.add_argument("--bench-glob", default=None, help="glob for bench JSONs (overrides --bench-dir)")
    ap.add_argument("--examples", type=int, default=40)
    ap.add_argument("--out-prefix", default=None, help="default: results/semantic_qa_<backend>")
    args = ap.parse_args()

    import glob as _glob

    files = sorted(Path(f) for f in _glob.glob(args.bench_glob)) if args.bench_glob else sorted(Path(args.bench_dir).glob("*.json"))
    if not files:
        sys.exit(f"no bench JSONs in {args.bench_glob or args.bench_dir}")
    runs = [json.load(open(f)) for f in files]
    present = {d["policy"]["name"] for d in runs}
    if args.baseline == "auto":
        args.baseline = "B3" if "B3" in present else ("ACLScope" if "ACLScope" in present else sorted(present)[0])
    encoder = CharNgramEncoder() if args.backend == "char" else SentenceTransformerEncoder(args.model)
    scorer = SemanticScorer(encoder, args.threshold)
    prefix = Path(args.out_prefix or f"results/semantic_qa_{args.backend}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    def setting_of(d):
        """(workload, load, KV budget, model): runs are only ever paired within one setting."""
        a = d["args"]
        load = f"r{a['rate']:g}" if a.get("arrival") == "poisson" else f"c{a['concurrency']}"
        return (Path(d["workload"]["path"]).stem, load, a["kv_gib"], d.get("server", {}).get("model", ""))

    # example answers for threshold sanity-checking: the original v1 setting (share 0.5, C=4, 1 GiB)
    # with B6 when present, otherwise the first setting that contains the baseline
    legacy = [d for d in runs if (d["workload"]["config"].get("share_rate"), d["args"]["concurrency"], d["args"]["kv_gib"]) == (0.5, 4, 1.0)]
    if legacy and any(d["policy"]["name"] == "B6" for d in legacy):
        ex_setting, ex_other = setting_of(legacy[0]), "B6"
    else:
        with_base = sorted(setting_of(d) for d in runs if d["policy"]["name"] == args.baseline)
        ex_setting = with_base[0] if with_base else None
        others = sorted({d["policy"]["name"] for d in runs if setting_of(d) == ex_setting} - {args.baseline})
        ex_other = others[0] if others else None

    rows, per_req, ex_rescued, ex_border = [], {}, [], []
    for d in runs:
        pol = d["policy"]["name"]
        setting = setting_of(d)
        share = d["workload"]["config"].get("share_rate")
        conc, kv = d["args"]["concurrency"], d["args"]["kv_gib"]
        recs = [r for r in d["records"] if r["ok"]]
        preds = [r["text"] for r in recs]
        golds = [r.get("answers") or [r["answer"]] for r in recs]
        sem = scorer.evaluate_multi(preds, golds)
        base = evaluate(list(zip(preds, golds)))
        rows.append({"policy": pol, "share": share, "conc": conc, "kv_gib": kv, "n": sem["n"],
                     "em": base["em"], "f1": base["f1"], "sem_sim": sem["sem_sim"],
                     "sem_match": sem["sem_match"], "sem_rescued": sem["sem_rescued"],
                     "workload": setting[0], "load": setting[1], "model": setting[3]})
        sims = scorer.similarity_multi(preds, golds)
        ems = [any(exact_match(clean_prediction(t), g) for g in gs) for t, gs in zip(preds, golds)]
        per_req[(setting, pol)] = {r["req_id"]: (r["qid"], s, em) for r, s, em in zip(recs, sims, ems)}
        if setting == ex_setting and pol in (args.baseline, ex_other):
            for txt, gs, s, em in zip(preds, golds, sims, ems):
                row = {"policy": pol, "prediction": clean_prediction(txt), "gold": " | ".join(gs), "similarity": round(s, 3), "em": int(em)}
                if s >= args.threshold and not em:
                    ex_rescued.append(row)
                elif abs(s - args.threshold) <= 0.1:
                    ex_border.append(row)

    with open(f"{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    random.seed(0)
    examples = (random.sample(ex_rescued, min(args.examples, len(ex_rescued)))
                + random.sample(ex_border, min(args.examples, len(ex_border))))
    with open(f"{prefix}_examples.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["policy", "prediction", "gold", "similarity", "em"])
        w.writeheader()
        w.writerows(examples)

    by_pol = defaultdict(list)
    for r in rows:
        by_pol[r["policy"]].append(r)
    pols = [p for p in POLICY_ORDER if p in by_pol] + [p for p in by_pol if p not in POLICY_ORDER]
    avg = lambda p, k: sum(r[k] for r in by_pol[p]) / len(by_pol[p])  # noqa: E731
    lines = [f"# Semantic QA quality ({encoder.name}, threshold {args.threshold})", "",
             f"{len(files)} bench runs, {sum(r['n'] for r in rows)} requests. Means over runs.", "",
             "| policy | EM | F1 | sem_sim | sem_match | rescued by semantic |", "|---|---|---|---|---|---|"]
    for p in pols:
        lines.append(f"| {p} | {avg(p,'em'):.3f} | {avg(p,'f1'):.3f} | {avg(p,'sem_sim'):.3f} | {avg(p,'sem_match'):.3f} | {avg(p,'sem_rescued'):.3f} |")
    lines += ["", f"## Paired difference vs {args.baseline} (same workload and request; 95% bootstrap CI over questions)", "",
              "| policy | d sem_sim | 95% CI | d EM | 95% CI |", "|---|---|---|---|---|"]
    for p in pols:
        if p == args.baseline:
            continue
        ds, de = defaultdict(list), defaultdict(list)
        for (setting, pol), reqs in per_req.items():
            if pol != p or (setting, args.baseline) not in per_req:
                continue
            ref = per_req[(setting, args.baseline)]
            for rid, (qid, s, em) in reqs.items():
                if rid in ref:
                    ds[qid].append(s - ref[rid][1])
                    de[qid].append(float(em) - float(ref[rid][2]))
        if ds:
            m1, l1, h1 = paired_ci(ds)
            m2, l2, h2 = paired_ci(de)
            lines.append(f"| {p} | {m1:+.4f} | [{l1:+.4f}, {h1:+.4f}] | {m2:+.4f} | [{l2:+.4f}, {h2:+.4f}] |")
    Path(f"{prefix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {prefix}.csv, {prefix}.md, {prefix}_examples.csv")


if __name__ == "__main__":
    main()
