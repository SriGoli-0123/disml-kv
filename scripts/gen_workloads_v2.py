#!/usr/bin/env python
"""Generate the v2 workloads listed in configs/v2.json.

    python scripts/gen_workloads_v2.py                 # all
    python scripts/gen_workloads_v2.py hq_main_s0 mq_main_s0
    python scripts/gen_workloads_v2.py --n-requests 2000 --suffix _live   # smaller copies for GPU runs
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.data_prep import load_corpus, load_questions  # noqa: E402
from aclkv.retrieval import Retriever  # noqa: E402
from aclkv.workload import WorkloadConfig, generate_topic  # noqa: E402


def resolve(cfg: dict, name: str) -> dict:
    base = dict(cfg["base_workload"])
    base.update(cfg["workloads"][name])
    base["acl_mix"] = tuple(cfg["acl_mixes"][base["acl_mix"]]) if isinstance(base["acl_mix"], str) else tuple(base["acl_mix"])
    base["name"] = name
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--config", default="configs/v2.json")
    ap.add_argument("--out-dir", default="data/workloads_v2")
    ap.add_argument("--n-requests", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--retrieval", default=None, choices=[None, "bm25", "dense", "hybrid"])
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    names = args.names or list(cfg["workloads"])
    retrievers, data = {}, {}
    out_dir = Path(args.out_dir)
    for name in names:
        wc = resolve(cfg, name)
        if args.n_requests:
            wc["n_requests"] = args.n_requests
        if args.retrieval:
            wc["retrieval"] = args.retrieval
        ds = wc["dataset"]
        if ds not in data:
            root = Path(cfg["datasets"][ds])
            corpus, qs = load_corpus(root / "corpus.jsonl"), load_questions(root / "questions.jsonl")
            data[ds] = (corpus, qs)
        corpus, qs = data[ds]
        rkey = (ds, wc.get("retrieval", "bm25"))
        if rkey not in retrievers:
            ids = sorted(corpus)
            cache = Path(cfg["datasets"][ds]) / "dense_bge-small.npy"
            retrievers[rkey] = Retriever(ids, [corpus[c].title + " " + corpus[c].text for c in ids], rkey[1],
                                         dense_cache=cache)
        t0 = time.time()
        w = generate_topic(WorkloadConfig(**wc), corpus, qs, retrievers[rkey])
        path = out_dir / f"{name}{args.suffix}.json"
        w.save(path)
        st = w.stats
        print(f"{name}{args.suffix}: {st['n_requests']} req, {st['distinct_questions']} distinct q, "
              f"gold recall@k={st['gold_recall_at_k']:.2f}, first-doc seen={st['first_doc_seen_before']:.2f}, "
              f"same doc set seen={st['exact_doc_set_seen_before']:.2f}, docs seen={st['docs_seen_before']:.2f}, "
              f"mix={ {k: round(v, 2) for k, v in st['evidence_acl_mix'].items()} } ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
