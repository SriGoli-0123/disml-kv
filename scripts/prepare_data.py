#!/usr/bin/env python
"""Download HotpotQA (distractor, validation) and build the chunk corpus +
questions with BM25 relevance ranks.

    python scripts/prepare_data.py --max-questions 3000 --tokenizer Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.data_prep import DATASET_URLS, build_dataset, download_hotpot, read_hotpot_parquet, save  # noqa: E402
from aclkv.tokenizer import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa", choices=sorted(DATASET_URLS))
    ap.add_argument("--raw", default=None, help="parquet path (default: data/raw/<dataset>_validation.parquet)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--out-dir", default=None, help="default: data/prepared (hotpotqa) or data/prepared_<dataset>")
    ap.add_argument("--max-questions", type=int, default=3000, help="subsample questions; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None, help="if set, store token counts per chunk")
    args = ap.parse_args()

    raw_default = {"hotpotqa": "data/raw/hotpot_distractor_validation.parquet",
                   "musique": "data/raw/musique_answerable_validation.parquet"}[args.dataset]
    args.raw = args.raw or raw_default
    args.out_dir = args.out_dir or ("data/prepared" if args.dataset == "hotpotqa" else f"data/prepared_{args.dataset}")
    raw = download_hotpot(Path(args.raw), args.url or DATASET_URLS[args.dataset])
    rows = read_hotpot_parquet(raw, args.max_questions or None, args.seed)
    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None
    chunks, questions = build_dataset(args.dataset, rows, tok)
    save(chunks, questions, Path(args.out_dir))
    n_tok = [c.n_tokens for c in chunks if c.n_tokens]
    stats = {
        "questions": len(questions), "chunks": len(chunks),
        "mean_chunk_tokens": (sum(n_tok) / len(n_tok)) if n_tok else None,
        "gold_rank_hist": {},
    }
    hist = {}
    for q in questions:
        for c in q.context:
            if c["chunk_id"] in q.gold_chunk_ids:
                hist[c["rank"]] = hist.get(c["rank"], 0) + 1
    stats["gold_rank_hist"] = {str(k): v for k, v in sorted(hist.items())}
    json.dump(stats, open(Path(args.out_dir) / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
