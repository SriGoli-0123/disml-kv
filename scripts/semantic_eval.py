"""Score every saved bench run with the semantic-similarity metric.

Reads results/bench/*.json (the per-request records already contain the model's
text and the gold answer), so it needs no GPU and no server.

    pip install -e ".[semantic]"
    python scripts/semantic_eval.py                     # neural embeddings (all-MiniLM-L6-v2)
    python scripts/semantic_eval.py --backend char      # offline stand-in, NOT neural

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

from aclkv.qa_eval import clean_prediction, evaluate  # noqa: E402
from aclkv.semantic import CharNgramEncoder, SemanticScorer, SentenceTransformerEncoder  # noqa: E402

POLICY_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "UB"]


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
    ap.add_argument("--baseline", default="B3", help="policy for the paired comparison (B3 = plain retrieval order)")
    ap.add_argument("--examples", type=int, default=40)
    ap.add_argument("--out-prefix", default=None, help="default: results/semantic_qa_<backend>")
    args = ap.parse_args()

    files = sorted(Path(args.bench_dir).glob("*.json"))
    if not files:
        sys.exit(f"no bench JSONs in {args.bench_dir}")
    encoder = CharNgramEncoder() if args.backend == "char" else SentenceTransformerEncoder(args.model)
    scorer = SemanticScorer(encoder, args.threshold)
    prefix = Path(args.out_prefix or f"results/semantic_qa_{args.backend}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows, per_req, ex_rescued, ex_border = [], {}, [], []
    for f in files:
        d = json.load(open(f))
        pol = d["policy"]["name"]
        share = d["workload"]["config"]["share_rate"]
        conc, kv = d["args"]["concurrency"], d["args"]["kv_gib"]
        recs = [r for r in d["records"] if r["ok"]]
        pairs = [(r["text"], r["answer"]) for r in recs]
        sem = scorer.evaluate(pairs)
        base = evaluate(pairs)
        rows.append({"policy": pol, "share": share, "conc": conc, "kv_gib": kv, "n": sem["n"],
                     "em": base["em"], "f1": base["f1"], "sem_sim": sem["sem_sim"],
                     "sem_match": sem["sem_match"], "sem_rescued": sem["sem_rescued"]})
        sims = scorer.similarity(pairs)
        ems = _em_list(pairs)
        per_req[(share, conc, kv, pol)] = {r["req_id"]: (r["qid"], s, em) for r, s, em in zip(recs, sims, ems)}
        # collect example answers from one representative setting for threshold sanity-checking
        if (share, conc, kv) == (0.5, 4, 1.0) and pol in (args.baseline, "B6"):
            for (txt, gold), s, em in zip(pairs, sims, ems):
                row = {"policy": pol, "prediction": clean_prediction(txt), "gold": gold, "similarity": round(s, 3), "em": int(em)}
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
        for key, reqs in per_req.items():
            if key[3] != p or (*key[:3], args.baseline) not in per_req:
                continue
            ref = per_req[(*key[:3], args.baseline)]
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


def _em_list(pairs):
    from aclkv.qa_eval import exact_match
    return [exact_match(clean_prediction(t), g) for t, g in pairs]


if __name__ == "__main__":
    main()
