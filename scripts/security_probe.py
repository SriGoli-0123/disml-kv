#!/usr/bin/env python
"""Live timing/cache-hit probe against a running vLLM server.

For a sample of *victim* requests from a workload (those whose prompt ends in
a group- or user-restricted scope), the honest victim request is sent first,
then an attacker who is **not** authorized replays the victim's exact token
sequence straight at the engine (bypassing the middleware):

  no_salt   attacker sends the tokens with no cache_salt
  forged    attacker keeps the barrier offsets but invents scope ids
  stolen    attacker somehow has the victim's exact salt (control: shows the
            HMAC scope ids are the secret; this is *expected* to hit)

Under the insecure global policy (UB: victims send no salt) the ``no_salt``
attacker gets ``cached_tokens`` equal to the victim's whole prefix and a much
lower TTFT — the prefix-cache timing side channel.  Under the scoped policy
(B3/B4/B5) ``no_salt`` and ``forged`` must report cached_tokens == 0 and a
cold TTFT.  The prefix cache is reset before every victim (needs
``VLLM_SERVER_DEV_MODE=1``), so an attacker hit can only come from the
victim's own blocks.

    python scripts/security_probe.py --workload data/workloads/w_share0.50_mix-default_u20_s0.json --n 20
"""
import argparse
import asyncio
import json
import random
import secrets
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.client import VLLMClient  # noqa: E402
from aclkv.metrics import latency_summary  # noqa: E402
from aclkv.pipeline import BuildSettings, build_prompts, load_workload_and_corpus, make_builder  # noqa: E402
from aclkv.scope import Barrier, encode_cache_salt  # noqa: E402
from aclkv.tokenizer import load_tokenizer  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--corpus", default="data/prepared/corpus.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--chat-format", default="qwen2.5")
    ap.add_argument("--policies", nargs="*", default=["UB", "B3", "B4", "B5"])
    ap.add_argument("--n", type=int, default=20, help="victims per policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/security_probe.json")
    args = ap.parse_args()

    workload, corpus = load_workload_and_corpus(args.workload, args.corpus)
    settings = BuildSettings(tokenizer=args.tokenizer, chat_format=args.chat_format)
    builder = make_builder(workload, settings, load_tokenizer(args.tokenizer))
    rng = random.Random(args.seed)
    client = VLLMClient(args.base_url)
    report = {"policies": {}}

    async with httpx.AsyncClient() as s:
        await client.wait_ready(s, 120)
        await client.resolve_model(s)
        for pol in args.policies:
            prompts = build_prompts(workload, corpus, pol, builder, settings)
            restricted = [p for p in prompts if not p.segments[-1].acl.is_public]
            victims = rng.sample(restricted, min(args.n, len(restricted)))
            rows = []
            print(f"\n=== policy {pol}: {len(victims)} victims")
            for v in victims:
                final_acl = v.segments[-1].acl
                outsiders = [u for u in workload.directory.users if not final_acl.allows(u, workload.directory)]
                attacker = rng.choice(outsiders)
                # fresh cache per victim: the only cached state is the victim's own, so any attacker
                # hit is a real leak (otherwise probes would hit *each other's* shared preamble blocks)
                if not await client.reset_prefix_cache(s):
                    sys.exit("/reset_prefix_cache failed: start the server with VLLM_SERVER_DEV_MODE=1")
                cold = await client.complete(s, v.req_id, v.token_ids, v.cache_salt, max_tokens=1)
                warm = await client.complete(s, v.req_id + "-again", v.token_ids, v.cache_salt, max_tokens=1)
                probes = {
                    "no_salt": None,
                    "forged": encode_cache_salt([Barrier(b.token_offset, secrets.token_hex(8)) for b in v.barriers]) if v.barriers else None,
                    "stolen": v.cache_salt,
                }
                row = {"victim": v.req_id, "victim_user": v.user, "attacker": attacker, "acl": final_acl.canonical(),
                       "num_tokens": v.num_tokens, "victim_cold_ttft": cold.ttft_s, "victim_cold_cached": cold.cached_tokens,
                       "victim_warm_ttft": warm.ttft_s, "victim_warm_cached": warm.cached_tokens, "probes": {}}
                for name, salt in probes.items():
                    if name == "forged" and salt is None:
                        continue
                    r = await client.complete(s, f"probe-{name}-{v.req_id}", v.token_ids, salt, max_tokens=1)
                    row["probes"][name] = {"ttft": r.ttft_s, "cached_tokens": r.cached_tokens, "ok": r.ok, "error": r.error}
                rows.append(row)
            summ = {
                "victim_cold_ttft": latency_summary([r["victim_cold_ttft"] for r in rows]),
                "victim_warm_ttft": latency_summary([r["victim_warm_ttft"] for r in rows]),
            }
            for name in ("no_salt", "forged", "stolen"):
                sel = [r["probes"][name] for r in rows if name in r["probes"]]
                if not sel:
                    continue
                leaks = sum(1 for x in sel if (x["cached_tokens"] or 0) > 0)
                summ[name] = {"n": len(sel), "attacker_hits": leaks,
                              "cached_tokens_mean": sum((x["cached_tokens"] or 0) for x in sel) / len(sel),
                              "ttft": latency_summary([x["ttft"] for x in sel])}
            report["policies"][pol] = {"summary": summ, "rows": rows}
            print(f"  victim cold TTFT p50={summ['victim_cold_ttft'].get('p50', float('nan'))*1000:.0f}ms  "
                  f"warm p50={summ['victim_warm_ttft'].get('p50', float('nan'))*1000:.0f}ms")
            for name in ("no_salt", "forged", "stolen"):
                if name in summ:
                    x = summ[name]
                    print(f"  attacker[{name:7s}] hits={x['attacker_hits']}/{x['n']}  cached_tokens mean={x['cached_tokens_mean']:.0f}  "
                          f"TTFT p50={x['ttft'].get('p50', float('nan'))*1000:.0f}ms")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    bad = {p: r["summary"] for p, r in report["policies"].items()
           if p != "UB" and any(r["summary"].get(k, {}).get("attacker_hits", 0) for k in ("no_salt", "forged"))}
    if bad:
        print("SECURITY VIOLATION under scoped policy:", json.dumps(bad, indent=1))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
