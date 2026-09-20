#!/usr/bin/env python
"""Generate the workload matrix (share rate x ACL mix) from the prepared data.

    python scripts/gen_workloads.py                      # full matrix from configs/matrix.json
    python scripts/gen_workloads.py --quick              # smaller workloads
    python scripts/gen_workloads.py --share-rates 0.5 --acl-mix default --n-requests 100
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.data_prep import BM25Index, load_corpus, load_questions  # noqa: E402
from aclkv.workload import WorkloadConfig, generate  # noqa: E402


def workload_name(share: float, mix_name: str, n_users: int, seed: int) -> str:
    return f"w_share{share:.2f}_mix-{mix_name}_u{n_users}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/matrix.json")
    ap.add_argument("--prepared", default="data/prepared")
    ap.add_argument("--out-dir", default="data/workloads")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--share-rates", type=float, nargs="*", default=None)
    ap.add_argument("--acl-mix", nargs="*", default=None, help="names from configs/matrix.json acl_mixes")
    ap.add_argument("--n-requests", type=int, default=None)
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    base = dict(cfg["workload"])
    shares = args.share_rates or (cfg["quick"]["share_rates"] if args.quick else cfg["share_rates"])
    mixes = args.acl_mix or (["default"] if args.quick else list(cfg["acl_mixes"]))
    if args.n_requests:
        base["n_requests"] = args.n_requests
    elif args.quick:
        base["n_requests"] = cfg["quick"]["n_requests"]
    if args.n_users:
        base["n_users"] = args.n_users
    seeds = args.seeds or [base["seed"]]

    corpus = load_corpus(Path(args.prepared) / "corpus.jsonl")
    questions = load_questions(Path(args.prepared) / "questions.jsonl")
    print(f"corpus: {len(corpus)} chunks, {len(questions)} questions; building BM25 ...")
    bm25 = BM25Index(list(corpus.values()))
    out_dir = Path(args.out_dir)
    for seed in seeds:
        for mix_name in mixes:
            for share in shares:
                name = workload_name(share, mix_name, base["n_users"], seed)
                wcfg = WorkloadConfig(**{**base, "name": name, "share_rate": share, "seed": seed,
                                         "acl_mix": tuple(cfg["acl_mixes"][mix_name])})
                w = generate(wcfg, corpus, questions, bm25)
                w.save(out_dir / f"{name}.json")
                st = w.stats
                print(f"{name}: {st['n_requests']} req, {st['distinct_questions']} distinct q, "
                      f"hot-doc frac={st['hot_docs_frac']:.2f}, evidence mix={ {k: round(v, 2) for k, v in st['evidence_acl_mix'].items()} }")


if __name__ == "__main__":
    main()
