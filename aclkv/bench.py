"""Closed-loop benchmark against a live vLLM server.

For every policy: reset the prefix cache, replay the workload with ``C``
concurrent workers (each worker sends its next request as soon as the
previous one finishes), record per-request TTFT / latency / cached tokens /
answer text, scrape Prometheus deltas, and run the simulator on the *same*
prompts to obtain evictions, unique blocks and the security audit.

Usage::

    python -m aclkv.bench --workload data/workloads/w_share0.50.json \
        --corpus data/prepared/corpus.jsonl --policies B0 B1 B2 B3 B4 UB \
        --concurrency 16 --kv-gib 2 --out-dir results/bench
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from .client import CompletionResult, VLLMClient
from .context import BuiltPrompt
from .metrics import counter_delta, latency_summary
from .model_info import GIB, kv_geometry
from .pipeline import BuildSettings, build_prompts, load_workload_and_corpus, make_builder
from .policies import get_policy
from .qa_eval import evaluate
from .simulator import simulate
from .tokenizer import load_tokenizer


async def run_closed_loop(client: VLLMClient, session: httpx.AsyncClient, prompts: list[BuiltPrompt],
                          concurrency: int, max_tokens: int, progress: bool = True) -> tuple[list[CompletionResult], float]:
    queue: asyncio.Queue = asyncio.Queue()
    for p in prompts:
        queue.put_nowait(p)
    results: dict[str, CompletionResult] = {}
    done = 0

    async def worker():
        nonlocal done
        while True:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            res = await client.complete(session, p.req_id, p.token_ids, p.cache_salt, max_tokens=max_tokens)
            results[p.req_id] = res
            done += 1
            if progress and done % 50 == 0:
                print(f"    {done}/{len(prompts)} done", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    wall = time.perf_counter() - t0
    return [results[p.req_id] for p in prompts], wall


async def run_open_loop(client: VLLMClient, session: httpx.AsyncClient, prompts: list[BuiltPrompt],
                        rate: float, max_tokens: int, seed: int = 0, progress: bool = True
                        ) -> tuple[list[CompletionResult], float]:
    """Open-loop Poisson arrivals at ``rate`` requests/s: each request is sent at
    its scheduled time regardless of whether earlier ones finished, so queueing
    delay shows up in TTFT exactly as it would for independent users."""
    import random as _random

    rng = _random.Random(seed)
    t, sched = 0.0, []
    for _ in prompts:
        t += rng.expovariate(rate)
        sched.append(t)
    results: dict[str, CompletionResult] = {}
    done = 0
    t0 = time.perf_counter()

    async def one(p: BuiltPrompt, at: float):
        nonlocal done
        delay = at - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        results[p.req_id] = await client.complete(session, p.req_id, p.token_ids, p.cache_salt, max_tokens=max_tokens)
        done += 1
        if progress and done % 100 == 0:
            print(f"    {done}/{len(prompts)} done", flush=True)

    await asyncio.gather(*[one(p, at) for p, at in zip(prompts, sched)])
    wall = time.perf_counter() - t0
    return [results[p.req_id] for p in prompts], wall


def summarize_run(prompts: list[BuiltPrompt], results: list[CompletionResult], wall: float,
                  slo_ttft_s: float | None = None) -> dict:
    ok = [r for r in results if r.ok]
    pt = sum(r.prompt_tokens or 0 for r in ok)
    ct = sum(r.cached_tokens or 0 for r in ok)
    gen = sum(r.completion_tokens or 0 for r in ok)
    by_id = {p.req_id: p for p in prompts}
    qa = evaluate((r.text, by_id[r.req_id].meta.get("answers") or by_id[r.req_id].meta.get("answer", "")) for r in ok)
    goodput = None
    if slo_ttft_s is not None:
        good = sum(1 for r in ok if r.ttft_s is not None and r.ttft_s <= slo_ttft_s)
        goodput = {"slo_ttft_s": slo_ttft_s, "attained": good / max(1, len(results)),
                   "goodput_req_per_s": good / wall if wall > 0 else float("nan")}
    return {
        "goodput": goodput,
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "wall_s": wall,
        "req_per_s": len(ok) / wall if wall > 0 else float("nan"),
        "prompt_tokens": pt,
        "cached_tokens": ct,
        "recomputed_tokens": pt - ct,
        "cached_pct": (100.0 * ct / pt) if pt else 0.0,
        "generation_tokens": gen,
        "prompt_tok_per_s": pt / wall if wall > 0 else float("nan"),
        "ttft": latency_summary([r.ttft_s for r in ok]),
        "ttft_text": latency_summary([r.ttft_text_s for r in ok]),
        "e2e": latency_summary([r.e2e_s for r in ok]),
        "qa": qa,
    }


async def bench_policy(args, client: VLLMClient, session: httpx.AsyncClient, workload, corpus, builder,
                       settings: BuildSettings, policy_name: str, num_blocks: int | None) -> dict:
    pol = get_policy(policy_name)
    prompts = build_prompts(workload, corpus, pol, builder, settings)
    n_dropped = sum(p.n_dropped_by_filter for p in prompts)

    if args.reset:
        ok = await client.reset_prefix_cache(session)
        if not ok:
            print("  WARNING: /reset_prefix_cache failed (start server with VLLM_SERVER_DEV_MODE=1); results may carry over", flush=True)
        await asyncio.sleep(0.5)
    m_before = await client.metrics(session)
    if args.arrival == "poisson":
        print(f"  [{pol.name}] {pol.description}: {len(prompts)} requests, Poisson rate={args.rate}/s", flush=True)
        results, wall = await run_open_loop(client, session, prompts, args.rate, args.max_tokens, seed=args.seed)
    else:
        print(f"  [{pol.name}] {pol.description}: {len(prompts)} requests, concurrency={args.concurrency}", flush=True)
        results, wall = await run_closed_loop(client, session, prompts, args.concurrency, args.max_tokens)
    m_after = await client.metrics(session)
    summary = summarize_run(prompts, results, wall, slo_ttft_s=(args.slo_ttft_ms / 1000.0) if args.slo_ttft_ms else None)
    if args.judge_url and args.judge_model:
        summary["judge"] = await judge_run(session, args, prompts, results)
    summary["prom_delta"] = counter_delta(m_before, m_after)
    summary["n_dropped_by_filter"] = n_dropped

    sim = None
    if num_blocks:
        out_len = {r.req_id: (r.completion_tokens or args.max_tokens) for r in results}
        _, sim_summary = simulate(prompts, num_blocks, workload.directory, settings.block_size,
                                  max_concurrency=args.concurrency, output_tokens=args.max_tokens,
                                  output_lengths=out_len)
        sim = sim_summary.to_json()
    errs = [r.error for r in results if not r.ok][:3]
    print(f"    cached={summary['cached_pct']:.1f}%  ttft p50={summary['ttft']['p50']*1000:.0f}ms "
          f"p95={summary['ttft']['p95']*1000:.0f}ms  req/s={summary['req_per_s']:.2f}  "
          f"EM={summary['qa'].get('em', float('nan')):.3f} F1={summary['qa'].get('f1', float('nan')):.3f}"
          + (f"  sim: evictions={sim['evictions']} unauthorized={sim['unauthorized_hits']}" if sim else "")
          + (f"  ERRORS={summary['errors']} e.g. {errs}" if errs else ""), flush=True)

    return {
        "policy": pol.__dict__,
        "settings": settings.to_json(),
        "args": {k: v for k, v in vars(args).items() if k not in ("policies",)},
        "workload": {"path": str(args.workload), "config": workload.config.to_json(), "stats": workload.stats},
        "summary": summary,
        "simulator": sim,
        "records": [
            {**r.to_json(), "user": p.user, "qid": p.meta.get("qid"), "answer": p.meta.get("answer"),
             "answers": p.meta.get("answers"), "gold": p.meta.get("gold"),
             "num_tokens_sent": p.num_tokens, "cache_salt": p.cache_salt, "ordered_chunk_ids": p.ordered_chunk_ids,
             "n_shared_docs": p.meta.get("n_shared_docs")}
            for p, r in zip(prompts, results)
        ],
    }


async def judge_run(session, args, prompts, results) -> dict:
    """Fixed-rubric LLM judge (semantic correctness), bounded concurrency."""
    from .qa_eval import llm_judge

    by_id = {p.req_id: p for p in prompts}
    sem = asyncio.Semaphore(args.judge_concurrency)
    verdicts: dict[str, bool | None] = {}

    async def one(r):
        p = by_id[r.req_id]
        async with sem:
            verdicts[r.req_id] = await llm_judge(session, args.judge_url, args.judge_model, p.meta.get("question", ""),
                                                 p.meta.get("answer", ""), r.text)

    await asyncio.gather(*[one(r) for r in results if r.ok])
    vals = [v for v in verdicts.values() if v is not None]
    return {"n": len(vals), "accuracy": (sum(vals) / len(vals)) if vals else None,
            "failed": sum(1 for v in verdicts.values() if v is None), "per_request": verdicts}


def out_name(args, pol_slug: str) -> str:
    load = f"r{args.rate:g}" if args.arrival == "poisson" else f"c{args.concurrency}"
    return f"{args.tag + '_' if args.tag else ''}{Path(args.workload).stem}_{load}_kv{args.kv_gib}_{pol_slug}.json"


async def amain(args) -> None:
    workload, corpus = load_workload_and_corpus(args.workload, args.corpus)
    settings = BuildSettings(tokenizer=args.tokenizer, chat_format=args.chat_format, block_size=args.block_size,
                             align_blocks=args.align, scope_mode=args.scope_mode, max_shift=args.max_shift,
                             min_pop=args.min_pop, anchor_k=args.anchor_k)
    tok = load_tokenizer(settings.tokenizer)
    builder = make_builder(workload, settings, tok)
    client = VLLMClient(args.base_url, args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed_policies = []
    for pol in args.policies:
        slug = get_policy(pol).slug
        load = f"r{args.rate:g}" if args.arrival == "poisson" else f"c{args.concurrency}"
        prefix = f"{args.tag + '_' if args.tag else ''}{Path(args.workload).stem}_{load}_"
        candidates = [
            out_dir / f"{prefix}kv{args.kv_gib}_{slug}.json",
            out_dir / f"{prefix}kv{float(args.kv_gib):.1f}_{slug}.json" if args.kv_gib is not None else None,
            out_dir / f"{prefix}kv{int(args.kv_gib)}_{slug}.json" if (args.kv_gib is not None and float(args.kv_gib).is_integer()) else None,
        ]
        already_done = False
        if not getattr(args, "force", False):
            for target_file in candidates:
                if target_file and target_file.exists():
                    try:
                        with open(target_file) as f:
                            data = json.load(f)
                        if "summary" in data and data["summary"].get("req_per_s") is not None:
                            print(f"    [checkpoint] skipping already completed policy {pol} ({target_file.name})", flush=True)
                            already_done = True
                            break
                    except Exception:
                        pass
        if not already_done:
            needed_policies.append(pol)

    if not needed_policies:
        print(f"    [checkpoint] all {len(args.policies)} policies already completed for {Path(args.workload).stem} c={args.concurrency} kv={args.kv_gib}. Skipping run.", flush=True)
        return

    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as session:
        await client.wait_ready(session)
        model = await client.resolve_model(session)
        m = await client.metrics(session)
        num_blocks = None
        cc = m.get("cache_config", {})
        if cc.get("num_gpu_blocks") not in (None, "None", ""):
            num_blocks = int(cc["num_gpu_blocks"])
        elif args.kv_gib:
            try:
                num_blocks = kv_geometry(model).num_blocks(int(args.kv_gib * GIB), args.block_size)
            except KeyError:
                pass
        print(f"server model={model}  num_gpu_blocks={num_blocks}  kv_gib={args.kv_gib}", flush=True)

        if args.warmup > 0:
            wp = build_prompts(workload, corpus, "B0", builder, settings)[: args.warmup]
            await run_closed_loop(client, session, wp, min(args.concurrency, len(wp)), args.max_tokens, progress=False)

        for pol in needed_policies:
            res = await bench_policy(args, client, session, workload, corpus, builder, settings, pol, num_blocks)
            res["server"] = {"model": model, "num_gpu_blocks": num_blocks, "cache_config": cc}
            name = out_name(args, get_policy(pol).slug)
            target_path = out_dir / name
            tmp_path = out_dir / f".tmp_{name}"
            with open(tmp_path, "w") as f:
                json.dump(res, f)
            tmp_path.replace(target_path)
            print(f"    -> {target_path}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--corpus", default="data/prepared/corpus.jsonl")
    ap.add_argument("--policies", nargs="+",
                    default=["NoCache", "PerUser", "Divergence", "Binary", "ACLScope", "ACLScope+Tree", "ACLKV", "Global"])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="served model name (default: first from /v1/models)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--chat-format", default="qwen2.5", choices=["qwen2.5", "qwen3", "llama3", "plain"])
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--align", dest="align", action="store_true", default=True)
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.add_argument("--scope-mode", default="hash", choices=["hash", "marker"])
    ap.add_argument("--max-shift", type=int, default=None)
    ap.add_argument("--anchor-k", type=int, default=0, help="pin the top-k retrieved docs immediately before the question")
    ap.add_argument("--min-pop", type=float, default=1.0, help="reuse-aware: decayed popularity needed to count as hot")
    ap.add_argument("--kv-gib", type=float, default=None, help="label + fallback for block count")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--reset", dest="reset", action="store_true", default=True)
    ap.add_argument("--no-reset", dest="reset", action="store_false")
    ap.add_argument("--out-dir", default="results/bench")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", default=False, help="force re-running benchmarks even if results already exist")
    ap.add_argument("--arrival", default="closed", choices=["closed", "poisson"],
                    help="closed: --concurrency workers; poisson: open-loop arrivals at --rate req/s")
    ap.add_argument("--rate", type=float, default=8.0, help="Poisson arrival rate (req/s)")
    ap.add_argument("--slo-ttft-ms", type=float, default=None, help="report goodput = requests meeting this TTFT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-url", default=None, help="OpenAI-compatible endpoint for the LLM judge")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-concurrency", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None):
    asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":
    main()
