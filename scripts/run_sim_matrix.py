#!/usr/bin/env python
"""Simulator sweep: workloads x policies x KV budgets x concurrency (no GPU).

Produces results/sim/<tag>/summary.csv (+ one JSON per run with per-request
records) and prints a table.  The same prompts are later replayed on the
real server by scripts/run_bench_matrix.sh, so numbers are directly
comparable (cached tokens are deterministic; TTFT/throughput need the GPU).

    python scripts/run_sim_matrix.py                # full matrix (configs/matrix.json)
    python scripts/run_sim_matrix.py --quick
    python scripts/run_sim_matrix.py --workloads data/workloads/w_share0.50_mix-default_u20_s0.json --policies B1 B3 B5
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.model_info import GIB, kv_geometry  # noqa: E402
from aclkv.pipeline import BuildSettings, build_prompts, load_workload_and_corpus, make_builder  # noqa: E402
from aclkv.policies import get_policy  # noqa: E402
from aclkv.simulator import simulate  # noqa: E402
from aclkv.tokenizer import load_tokenizer  # noqa: E402

COLUMNS = ["workload", "share_rate", "acl_mix", "policy", "kv_gib", "num_blocks", "concurrency", "requests",
           "prompt_tokens", "cached_tokens", "cached_pct", "recomputed_tokens", "evictions", "stalls",
           "max_table_size", "unauthorized_hits", "probe_requests", "probe_unauthorized_hits", "probe_cached_tokens",
           "mean_prompt_tokens", "wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/matrix.json")
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument("--corpus", default="data/prepared/corpus.jsonl")
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--kv-gib", type=float, nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--chat-format", default=None)
    ap.add_argument("--model", default=None, help="for KV geometry (bytes/token)")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--scope-mode", default="hash", choices=["hash", "marker"])
    ap.add_argument("--max-shift", type=int, default=None)
    ap.add_argument("--min-pop", type=float, default=1.0)
    ap.add_argument("--output-tokens", type=int, default=None)
    ap.add_argument("--probe-rate", type=float, default=0.25, help="fraction of requests followed by an adversarial replay")
    ap.add_argument("--probe-mode", default="no_salt", choices=["no_salt", "forged"])
    ap.add_argument("--out-dir", default="results/sim")
    ap.add_argument("--tag", default="matrix")
    ap.add_argument("--save-records", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    q = cfg["quick"] if args.quick else cfg
    policies = args.policies or q["policies"]
    kv_gibs = args.kv_gib or q["kv_gib"]
    concs = args.concurrency or q["concurrency"]
    model = args.model or cfg["model"]
    tok_name = args.tokenizer or cfg["tokenizer"]
    fmt = args.chat_format or cfg["chat_format"]
    out_tokens = args.output_tokens or cfg["max_tokens"]
    workloads = args.workloads or sorted(str(p) for p in Path("data/workloads").glob("w_*.json"))
    if not workloads:
        sys.exit("no workloads found; run scripts/gen_workloads.py first")

    geom = kv_geometry(model)
    print(f"model={model}: {geom.bytes_per_token} bytes/token, {geom.bytes_per_block(cfg['block_size'])/1e6:.2f} MB/block")
    settings = BuildSettings(tokenizer=tok_name, chat_format=fmt, block_size=cfg["block_size"],
                             align_blocks=not args.no_align, scope_mode=args.scope_mode,
                             max_shift=args.max_shift, min_pop=args.min_pop)
    tok = load_tokenizer(tok_name)
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for wpath in workloads:
        workload, corpus = load_workload_and_corpus(wpath, args.corpus)
        builder = make_builder(workload, settings, tok)
        wname = Path(wpath).stem
        for pol_name in policies:
            pol = get_policy(pol_name)
            t0 = time.time()
            prompts = build_prompts(workload, corpus, pol, builder, settings)
            n_tok = sum(p.num_tokens for p in prompts)
            for kv in kv_gibs:
                num_blocks = geom.num_blocks(int(kv * GIB), cfg["block_size"])
                for c in concs:
                    # efficiency run: honest replay only (probes would perturb LRU state)
                    recs, s = simulate(prompts, num_blocks, workload.directory, cfg["block_size"], max_concurrency=c,
                                       output_tokens=out_tokens, probe_rate=0.0, seed=workload.config.seed)
                    # security run: same replay interleaved with adversarial probes
                    _, sec = simulate(prompts, num_blocks, workload.directory, cfg["block_size"], max_concurrency=c,
                                      output_tokens=out_tokens, probe_rate=args.probe_rate, probe_mode=args.probe_mode,
                                      seed=workload.config.seed)
                    s.probe_requests, s.probe_unauthorized_hits, s.probe_cached_tokens = (
                        sec.probe_requests, sec.probe_unauthorized_hits, sec.probe_cached_tokens)
                    s.unauthorized_hits = max(s.unauthorized_hits, sec.unauthorized_hits)
                    row = {
                        "workload": wname, "share_rate": workload.config.share_rate,
                        "acl_mix": "/".join(f"{x:.2f}" for x in workload.config.acl_mix),
                        "policy": pol.name, "kv_gib": kv, "num_blocks": num_blocks, "concurrency": c,
                        "requests": s.requests, "prompt_tokens": s.prompt_tokens, "cached_tokens": s.cached_tokens,
                        "cached_pct": round(s.cached_pct, 2), "recomputed_tokens": s.recomputed_tokens,
                        "evictions": s.evictions, "stalls": s.stalls, "max_table_size": s.max_table_size,
                        "unauthorized_hits": s.unauthorized_hits, "probe_requests": s.probe_requests,
                        "probe_unauthorized_hits": s.probe_unauthorized_hits, "probe_cached_tokens": s.probe_cached_tokens,
                        "mean_prompt_tokens": round(n_tok / max(1, len(prompts)), 1), "wall_s": round(time.time() - t0, 2),
                    }
                    rows.append(row)
                    print(f"{wname:34s} {pol.name:8s} kv={kv:<4} c={c:<3} cached={s.cached_pct:5.1f}%  "
                          f"recomp={s.recomputed_tokens:8d}  evict={s.evictions:6d}  stalls={s.stalls:4d}  "
                          f"unauth={s.unauthorized_hits}  probe_leaks={s.probe_unauthorized_hits}", flush=True)
                    if args.save_records:
                        with open(out_dir / f"{wname}_{pol.name}_kv{kv}_c{c}.json", "w") as f:
                            json.dump({"row": row, "summary": s.to_json(),
                                       "records": [r.__dict__ for r in recs],
                                       "prompts": [p.to_json() for p in prompts] if pol_name == policies[0] else None}, f)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir / 'summary.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
