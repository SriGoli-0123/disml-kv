# Catalog branch: low-memory serving, the order-channel fix, and the evidence catalog

Course topic: *Serving LLMs in Low-Resource Environments*. Aim: reuse KV cache across users when
memory is small, without letting one user learn what a colleague asked.

## 1. Order-channel fix (ACLKV now uses a k-gated tree walk)

Cache-aware ordering (CacheWeaver-style) puts first the documents that other users recently
requested together. If users can see the order of their sources (numbered citations), the order
itself tells them that someone else asked for that combination. On the dev seed (`hq_main_s0`,
2 GiB, C = 16), the plain walk used by v2's ACLKV made the order reveal another user's activity in
32.3% of requests and point at **exactly one** colleague in **21.9%**, even though cache hits were
3-anonymous.

Fix (`aclkv/ordering.py`, ordering `ktree`): the walk only enters a prefix-tree node that at least
k distinct users (counting the requester) have traversed. Measured on the same dev seed:

| policy | cached % | order reveals others | exactly one colleague | smallest set |
|---|---|---|---|---|
| ACLKV before the fix (`ACLKV-ungated`) | 5.68 | 32.3% | 21.9% | 1 |
| **ACLKV (k-gated walk)** | **5.47** | 10.8% | **0.0%** | 2 |
| ACLScope (retrieval order) | 5.17 | 0% | 0% | – |

The leak is measured per request by `ordering.order_exposure` (stored as `meta["order_anon"]`) and
reported by the attack harness as `order_*` columns.

## 2. Frequency-aware eviction baselines

vLLM evicts with LRU. Prior work keeps frequently reused KV longer: RAGCache's PGDSF, HA-RAG, and an
ARC port to vLLM (arXiv 2606.21238). The simulator now implements, on the same block pool:

* `lfu`: in-cache LFU (count restarts after eviction), ties by free order;
* `gdsf`: Greedy-Dual-Size-Frequency (priority = clock + frequency × cost / size) with equal cost
  per block, justified by our live A100 finding that TTFT is linear in recomputed tokens
  (R² = 0.998);
* `arc`: ARC (Megiddo & Modha) with T1/T2 lists over free blocks and ghost lists of evicted hashes.

Empty blocks are always used first and a child block is always evicted before its parent, as in vLLM.
With `eviction="lru"` the simulator is byte-identical to before (checked on 48 committed runs).

## 3. The evidence catalog (`Catalog` policy, `aclkv/catalog.py`)

Users never write to the shared namespace directly. Every `period` requests a publisher looks at the
last `window` requests and publishes the evidence prefixes that **at least k distinct users**
requested at least `min_count` times, within a KV block budget (`budget` × KV blocks). Only published
prefixes are hashed into their shared ACL scope; the order follows only the published catalog.

| pin mode | what publishing does | deployable on |
|---|---|---|
| `hard` | publisher prefills missing catalog blocks; they stay resident until the next publish | needs an engine pin API |
| `soft` | catalog blocks are evicted only after every other free block; never reserved | needs a small eviction patch |
| `none` | only the shared allow-list and the order change | vLLM 0.29 + the aclkv plugin |

Privacy argument. With hard pins every hit on a shared block is a deterministic function of the
published catalog (≥ k users per prefix, batched), so it reveals nothing about who asked since.
With soft or no pins a published block can be evicted and later recomputed by one user, so those
modes also require k-anonymous promotion of the exact served prefix before sharing. (A first
version without this let a colleague single someone out in 13% of probes in a unit test; the test
now guards it.)

Costs that are counted: publisher prefill (`publish_pct`, subtracted in `net_cached_pct`) and the
memory hard pins take away from running requests (`mean_inflight`: how many other requests hold KV
when a request is admitted).

## 4. Dev-seed tuning (hq_main_s0 only)

`python scripts/run_v2.py lowmem catalog_dev --out-dir results/catalog`, then
`python scripts/analyze_catalog.py dev`. Qwen2.5-7B KV geometry (0.5 / 1 / 2 / 4 GiB =
585 / 1,170 / 2,340 / 4,681 blocks), C = 16. Net cached % (requests in flight at admission):

| | 0.5 GiB | 1 GiB | 2 GiB | 4 GiB |
|---|---|---|---|---|
| PerUser | 0.33 (5.0) | 0.61 (11.6) | 1.14 (16) | 2.05 (16) |
| ACLKV (LRU, vLLM) | 4.81 (5.3) | 5.04 (12.1) | 5.47 (16) | 6.12 (16) |
| ACLKV + GDSF | 4.81 | 5.04 | 5.61 | 6.75 |
| ACLKV + LFU | 4.83 | 5.05 | 7.07 | 8.16 |
| **ACLKV + ARC** | 4.87 | 5.10 | **7.57** | **8.59** |
| Catalog, hard pin (best of 14 settings) | 6.31 (**2.4**) | 7.06 (**6.2**) | 8.19 (13.5) | 8.66 (16) |
| Catalog, soft pin (best of 14) | 4.80 (5.3) | 4.95 (12.1) | 7.44 (16) | 8.09 (16) |
| Catalog, no pin (best of 12) | 4.72 (5.3) | 4.84 (12.1) | 5.21 (16) | 5.67 (16) |

Each catalog row is one setting (the one with the highest mean margin over the four budgets), not the best value per column.

Findings on the dev seed:

1. Frequency-aware eviction is the strong baseline: ARC adds +2.1 points over vLLM's LRU at 2 GiB
   (+38% relative) with no privacy cost (same scopes as ACLKV).
2. Hard pinning wins on cached tokens only by reserving memory: at 0.5–1 GiB it halves the number
   of requests that fit at once (5.3 → 2.4, 12.1 → 6.2). On a small GPU that is a throughput loss.
3. Soft pinning keeps concurrency but stays below ARC everywhere; without pinning, the catalog order
   alone gives no more than ACLKV.
4. At 0.5 GiB with a 7B model only about five requests fit; every policy is within 0.1 point.

No catalog setting meets both the efficiency and the capacity rule below on the dev seed.

## 5. Pre-registered test (written and committed before the test workloads existed)

* **Data**: fresh seeds `hq_main_s3`, `hq_main_s4` (HotpotQA) and `mq_main_s3` (MuSiQue), generated
  with `configs/v2.json` *after* this section was committed. No design decision used them.
* **Grid**: Qwen2.5-{0.5B, 1.5B, 7B} × KV {0.5, 1, 2, 4} GiB, C = 16: 12 cells, each the mean over
  the 3 workloads.
* **Candidate** (best dev setting that keeps concurrency): `Catalog` with
  `pin=soft,budget=0.5,period=500,window=2000,order=fp,min_count=2`.
  The best hard-pin setting (`pin=hard`, same other values) is reported descriptively only.
* **Baselines**: ACLKV with LRU, LFU, GDSF and ARC eviction ("best baseline" = the highest of the
  four in each cell).
* **PASS only if all hold**:
  1. the candidate's net cached % beats the best baseline in at least 9 of the 12 cells;
  2. its mean margin over the 12 cells is at least +0.5 points;
  3. its mean requests in flight are at least 95% of ACLKV's (LRU) in every cell;
  4. privacy, colleague (insider) harness on `hq_main_s3`, 7B, 2 GiB: 0 unauthorized hits, nobody
     singled out through cache hits or through the order, smallest anonymity set ≥ 2.
* **If it fails**: the catalog is reported as a negative result; ACLKV with the k-gated walk stays the
  method, and frequency-aware eviction (ARC) is reported as the strongest addition compatible with it.
* **Run once**: `python scripts/gen_workloads_v2.py hq_main_s3 hq_main_s4 mq_main_s3`,
  `python scripts/run_v2.py catalog_test --out-dir results/catalog`,
  `python scripts/analyze_catalog.py test`.

Expected from the dev seed: FAIL on rules 1–2.

**Deviation log.** The first test run crashed before writing any results: on `mq_main_s3` one
request needs 403 KV blocks, but at 7B / 0.5 GiB the descriptive hard-pin setting had pinned half of
the 585 blocks (its reserve was sized from the largest request seen so far). Fix (commit after
435c8cc): if a request cannot fit even after every running request has finished, hard-pinned blocks
are released, deepest first, and counted as `pin_breaks`. This only changes hard pinning; the
candidate (soft pins), the baselines and the rules above are unchanged. No test result had been seen.
