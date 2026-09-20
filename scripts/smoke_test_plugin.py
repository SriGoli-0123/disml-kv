#!/usr/bin/env python
"""Live check that the scoped-hash plugin is active on a running vLLM server.

Sends the *same 160-token prompt* under different salts and reads the
deterministic ``cached_tokens`` from the usage payload:

  1. scoped salt S1 = [public@0]                 -> 0   (cold)
  2. S1 again                                    -> 144 (9 of 10 blocks; vLLM recomputes the last one)
  3. S2 = [public@0, group@80]                   -> 80  (blocks 0-4 shared, block 5 carries the group scope)
                                                    * without the plugin this would be 0 *
  4. no salt                                     -> 0   (the un-scoped namespace is empty)
  5. forged scope id at 0                        -> 0
  6. stock salt "abc" twice                      -> 0 then 144 (vLLM's own behaviour preserved)

    python scripts/smoke_test_plugin.py --base-url http://localhost:8000
"""
import argparse
import asyncio
import secrets
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aclkv.client import VLLMClient  # noqa: E402
from aclkv.scope import Barrier, encode_cache_salt, load_scope_key, scope_id  # noqa: E402
from aclkv.tokenizer import load_tokenizer  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--blocks", type=int, default=10)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    words = tok.encode("The quick brown fox jumps over the lazy dog near the riverbank at dawn. ")
    ids = (words * 40)[: args.blocks * 16]
    n = len(ids)
    key = load_scope_key()
    pub, grp = scope_id(key, "public"), scope_id(key, "group:hr")
    s1 = encode_cache_salt([Barrier(0, pub)])
    s2 = encode_cache_salt([Barrier(0, pub), Barrier(80, grp)])
    forged = encode_cache_salt([Barrier(0, secrets.token_hex(8))])
    nonce_ids = tok.encode(secrets.token_hex(4))[:2]   # first block unique per run, independent of earlier runs
    ids = nonce_ids + ids[: n - len(nonce_ids)]

    client = VLLMClient(args.base_url)
    async with httpx.AsyncClient() as s:
        await client.wait_ready(s, 120)
        await client.resolve_model(s)
        await client.reset_prefix_cache(s)

        async def go(label, salt, expect):
            r = await client.complete(s, label, ids, salt, max_tokens=1)
            if not r.ok:
                print(f"  {label:34s} ERROR {r.error}")
                return False
            ok = r.cached_tokens == expect
            print(f"  {label:34s} cached_tokens={r.cached_tokens!s:>5}  expected={expect:<4}  {'OK' if ok else 'MISMATCH'}")
            return ok

        results = []
        print(f"prompt: {n} tokens = {n // 16} blocks; expecting max hit {((n - 1) // 16) * 16}")
        results.append(await go("1 scoped [public@0] cold", s1, 0))
        results.append(await go("2 scoped [public@0] again", s1, ((n - 1) // 16) * 16))
        results.append(await go("3 scoped [public@0, group@80]", s2, 80))
        results.append(await go("4 no salt", None, 0))
        results.append(await go("5 forged scope id", forged, 0))
        results.append(await go("6a stock salt 'abc' cold", "abc", 0))
        results.append(await go("6b stock salt 'abc' again", "abc", ((n - 1) // 16) * 16))
    if all(results):
        print("\nPLUGIN OK: multi-scope hashing is active and stock salts still work.")
    else:
        print("\nFAILED. If test 3 returned 0 the plugin is not loaded in the EngineCore process:"
              "\n  - is the package installed in the same env that runs `vllm serve`? (pip install -e .)"
              "\n  - does `VLLM_PLUGINS` exclude aclkv_scoped_hash?"
              "\n  - look for 'aclkv: scoped prefix-cache hashing installed' in the server log")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
