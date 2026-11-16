#!/usr/bin/env python
"""Simulator experiment suites for the v2 paper (CPU only, parallel).

    python scripts/run_v2.py                       # every suite
    python scripts/run_v2.py main privacy          # selected suites
    python scripts/run_v2.py --jobs 8 --quick      # 1000-request prefix of each workload

Suites -> results/v2/<suite>.csv
  main       policies x {HotpotQA, MuSiQue} x seeds (+ pinned, exact-query)
  privacy    insider/bypass attacks + k sweep (privacy/utility frontier)
  ablation   tail scope, layout components, alignment, tree window, min_pop
  pic        prefix caching vs position-independent chunk caching
  scaling    users/groups, prompt length, KV budget, model geometry, ACL mix,
             query noise, concurrency
  placement  where answer-bearing evidence lands under each ordering
  retriever  BM25 vs dense (bge-small) vs hybrid (RRF) retrieval
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CFG = json.load(open("configs/v2.json"))
WDIR = Path("data/workloads_v2")
_CACHE: dict = {}


def _load(wname: str, quick: int):
    from aclkv.data_prep import load_corpus
    from aclkv.workload import Workload

    key = (wname, quick)
    if key not in _CACHE:
        w = Workload.load(WDIR / f"{wname}.json")
        if quick:
            w.requests = w.requests[:quick]
        ds = w.config.dataset
        ck = ("corpus", ds)
        if ck not in _CACHE:
            _CACHE[ck] = load_corpus(Path(CFG["datasets"][ds]) / "corpus.jsonl")
        _CACHE[key] = (w, _CACHE[ck])
    return _CACHE[key]


def _builder(w, settings):
    from aclkv.pipeline import make_builder
    from aclkv.tokenizer import load_tokenizer

    if "tok" not in _CACHE:
        _CACHE["tok"] = load_tokenizer(CFG["tokenizer"])
    return make_builder(w, settings, _CACHE["tok"])


def _settings(opts: dict):
    from aclkv.pipeline import BuildSettings

    return BuildSettings(tokenizer=CFG["tokenizer"], chat_format=CFG["chat_format"], block_size=CFG["block_size"],
                         align_blocks=opts.get("align", True), min_pop=opts.get("min_pop", 1.0),
                         tree_window=opts.get("tree_window", 512), max_shift=opts.get("max_shift"))


def _blocks(model: str, kv: float) -> int:
    from aclkv.model_info import GIB, kv_geometry

    return kv_geometry(model).num_blocks(int(kv * GIB), CFG["block_size"])


def _placement(prompts) -> dict:
    pos, mid, edge_top1, n_docs = [], 0, 0, 0
    for p in prompts:
        ids = p.ordered_chunk_ids
        n = len(ids)
        gold = [c for c in p.meta.get("gold", []) if c in ids]
        for c in gold:
            i = ids.index(c)
            pos.append(i / max(1, n - 1))
            mid += 2 <= i <= n - 3
        n_docs += len(gold)
        top = p.meta.get("top1")
        if top in ids:
            edge_top1 += ids.index(top) in (0, n - 1)
    return {"gold_rel_pos_median": st.median(pos) if pos else float("nan"),
            "gold_in_middle_pct": 100.0 * mid / max(1, n_docs),
            "top1_at_edge_pct": 100.0 * edge_top1 / max(1, len(prompts))}


def run_task(task: dict) -> list[dict]:
    from aclkv.attacks import run_attack_trace
    from aclkv.pipeline import build_prompts
    from aclkv.policies import get_policy
    from aclkv.simulator import simulate

    t0 = time.time()
    w, corpus = _load(task["workload"], task.get("quick", 0))
    settings = _settings(task.get("opts", {}))
    builder = _builder(w, settings)
    pol = get_policy(task["policy"])
    base = {"suite": task["suite"], "workload": task["workload"], "policy": task["policy"],
            "variant": task.get("variant", ""), "dataset": w.config.dataset, "seed": w.config.seed,
            "n_users": w.config.n_users, "docs_per_request": w.config.docs_per_request,
            "query_noise": w.config.query_noise, "acl_mix": "/".join(f"{x:g}" for x in w.config.acl_mix),
            "scenario": w.config.scenario}
    rows = []
    if task["kind"] == "sim":
        prompts = build_prompts(w, corpus, pol, builder, settings)
        n_tok = sum(p.num_tokens for p in prompts)
        pad = sum(sg.n_pad for p in prompts for sg in p.segments)
        shared_docs = sum(p.meta.get("n_shared_docs", 0) for p in prompts) / max(1, len(prompts))
        for g in task["grid"]:
            model = g.get("model", CFG["model"])
            nb = _blocks(model, g["kv"])
            _, s = simulate(prompts, nb, w.directory, CFG["block_size"], max_concurrency=g["c"],
                            output_tokens=CFG["max_tokens"], mode=g.get("mode", "prefix"),
                            pic_recompute=g.get("pic_recompute", 0.15), pic_scoping=g.get("pic_scoping", "acl"))
            rows.append({**base, "model": model, "kv_gib": g["kv"], "concurrency": g["c"], "mode": g.get("mode", "prefix"),
                         "pic_scoping": g.get("pic_scoping", ""), "pic_recompute": g.get("pic_recompute", ""),
                         "num_blocks": nb, "requests": s.requests, "prompt_tokens": s.prompt_tokens,
                         "mean_prompt_tokens": n_tok / max(1, len(prompts)), "pad_frac": pad / max(1, n_tok),
                         "cached_tokens": s.cached_tokens, "cached_pct": s.cached_pct,
                         "recomputed_per_req": s.recomputed_tokens / max(1, s.requests),
                         "evictions": s.evictions, "stalls": s.stalls, "unauthorized_hits": s.unauthorized_hits,
                         "shared_docs_per_req": shared_docs})
        if task.get("placement"):
            for p, r in zip(prompts, w.requests):
                p.meta["top1"] = min(r.docs, key=lambda d: d["rank"])["chunk_id"] if r.docs else None
            pl = _placement(prompts)
            for row in rows:
                row.update(pl)
    elif task["kind"] == "attack":
        g = task["grid"][0]
        a = run_attack_trace(w, corpus, pol, builder, settings, _blocks(CFG["model"], g["kv"]), CFG["block_size"],
                             g["c"], CFG["max_tokens"], probe_rate=task.get("probe_rate", 0.1),
                             bypass_rate=task.get("bypass_rate", 0.05), seed=w.config.seed)
        rows.append({**base, "kv_gib": g["kv"], "concurrency": g["c"], "kanon": pol.kanon, **a.to_json()})
    for r in rows:
        r["task_s"] = round(time.time() - t0, 1)
    return rows


def suite_tasks(suite: str, quick: int) -> list[dict]:
    M = CFG["policies_main"]
    main_grid = [{"kv": CFG["main_kv_gib"], "c": CFG["main_concurrency"]}]
    T = []

    def add(kind, wl, pol, grid=None, **kw):
        T.append({"suite": suite, "kind": kind, "workload": wl, "policy": pol, "grid": grid or main_grid,
                  "quick": quick, **kw})

    if suite == "main":
        for wl in ["hq_main_s0", "hq_main_s1", "hq_main_s2", "mq_main_s0", "mq_main_s1", "mq_main_s2",
                   "hq_pinned_s0", "hq_exact_s0"]:
            for pol in M:
                add("sim", wl, pol, [{"kv": 2, "c": 16}, {"kv": 4, "c": 16}], placement=True)
    elif suite == "privacy":
        pols = ["Global", "Global+Tree", "PerUser", "Divergence", "Binary", "B3", "ACLScope", "ACLScope+Tree",
                "acl_scope/tree:k=2", "ACLKV", "acl_scope/tree:k=5", "acl_scope/tree:k=10",
                "acl_scope/retrieval:k=3", "acl_scope/tree:k=3,tail=inherit"]
        for wl in ["hq_main_s0", "hq_main_s1", "hq_main_s2", "mq_main_s0"]:
            for pol in pols:
                add("attack", wl, pol)
    elif suite == "ablation":
        wl = "hq_main_s0"
        for wl_ in ["hq_main_s0", "hq_main_s1", "hq_main_s2", "mq_main_s0"]:
            for pol, var in [("ACLScope", "tail=private"), ("acl_scope/retrieval:tail=inherit", "tail=inherit"),
                             ("ACLScope+Tree", "tail=private"), ("acl_scope/tree:tail=inherit", "tail=inherit"),
                             ("ACLKV", "tail=private"), ("acl_scope/tree:k=3,tail=inherit", "tail=inherit")]:
                add("sim", wl_, pol, variant=var)
            for pol, var in [("acl_scope/retrieval", "order=retrieval"), ("acl_scope/acl_aware", "order=acl_aware(v1 B4)"),
                             ("acl_scope/reuse_aware", "order=reuse_aware(v1 B5)"),
                             ("acl_scope/reuse_aware:anchor=2", "order=reuse_aware+anchor(v1 B6)"),
                             ("acl_scope/tree", "order=tree"), ("acl_scope/layout", "order=edge-layout"),
                             ("acl_scope/lcr", "order=lcr"), ("acl_scope/tree:k=3", "order=tree,k=3"),
                             ("acl_scope/tree_k:k=3", "order=privacy-aware-tree,k=3")]:
                add("sim", wl_, pol, variant=var, placement=True)
        for align in (True, False):
            for pol in ("ACLScope", "ACLKV"):
                add("sim", wl, pol, variant=f"align={align}", opts={"align": align})
        for win in (64, 512, 4096):
            for pol in ("ACLScope+Tree", "ACLKV"):
                add("sim", wl, pol, variant=f"tree_window={win}", opts={"tree_window": win})
        for mp in (1.0, 2.0, 4.0):
            add("sim", wl, "ACLKV", variant=f"min_pop={mp}", opts={"min_pop": mp})
    elif suite == "pic":
        for wl in ["hq_main_s0", "mq_main_s0"]:
            for r in (0.15, 0.30):
                add("sim", wl, "ACLScope", [{"kv": 2, "c": 16, "mode": "pic", "pic_scoping": "acl", "pic_recompute": r},
                                            {"kv": 2, "c": 16, "mode": "pic", "pic_scoping": "user", "pic_recompute": r},
                                            {"kv": 2, "c": 16, "mode": "pic", "pic_scoping": "global", "pic_recompute": r}],
                    variant=f"pic_r={r}")
            for pol in ("ACLScope", "ACLScope+Tree", "ACLKV", "Global+Tree"):
                add("sim", wl, pol, variant="prefix")
    elif suite == "scaling":
        P = ["PerUser", "ACLScope", "ACLScope+Tree", "ACLKV", "Global+Tree"]
        for wl in ["hq_u20_s0", "hq_main_s0", "hq_u500_s0"]:
            for pol in P:
                add("sim", wl, pol, variant="users")
        for wl in ["hq_main_s0", "hq_d20_s0", "hq_d40_s0"]:
            for pol in P:
                add("sim", wl, pol, [{"kv": 2, "c": 16}, {"kv": 8, "c": 16}], variant="prompt_length")
        for pol in P:
            add("sim", "hq_main_s0", pol, [{"kv": kv, "c": 16} for kv in CFG["kv_sweep_gib"]], variant="kv_budget")
            add("sim", "hq_main_s0", pol, [{"kv": 8, "c": 16, "model": m} for m in
                                           ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-32B-Instruct")],
                variant="model_geometry")
            add("sim", "hq_main_s0", pol, [{"kv": 2, "c": c} for c in (1, 4, 16, 32)], variant="concurrency")
        for wl in ["hq_restrictive_s0", "hq_main_s0", "hq_open_s0"]:
            for pol in P:
                add("sim", wl, pol, variant="acl_mix")
        for wl in ["hq_exact_s0", "hq_noise15_s0", "hq_main_s0", "hq_noise50_s0"]:
            for pol in P:
                add("sim", wl, pol, variant="query_noise")
    elif suite == "retriever":
        for ret in ("", "_dense", "_hybrid"):
            for seed in (0, 1, 2):
                wl = f"hq_main_s{seed}{ret}"
                if not (WDIR / f"{wl}.json").exists():
                    continue
                for pol in ("PerUser", "Divergence", "ACLScope", "ACLScope+Tree", "ACLKV", "Global+Tree"):
                    add("sim", wl, pol, variant=f"retriever={ret.strip('_') or 'bm25'}", placement=True)
    else:
        raise SystemExit(f"unknown suite {suite}")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="*", default=["main", "privacy", "ablation", "pic", "scaling"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", type=int, default=0, help="use only the first N requests of each workload")
    ap.add_argument("--out-dir", default="results/v2")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for suite in args.suites:
        tasks = suite_tasks(suite, args.quick)
        # longest tasks first: attacks, then large workloads
        tasks.sort(key=lambda t: (t["kind"] != "attack", "d40" not in t["workload"], "u500" not in t["workload"]))
        t0 = time.time()
        rows = []
        with get_context("spawn").Pool(args.jobs) as pool:
            for i, rs in enumerate(pool.imap_unordered(run_task, tasks), 1):
                rows.extend(rs)
                if i % 10 == 0 or i == len(tasks):
                    print(f"  [{suite}] {i}/{len(tasks)} tasks  ({time.time() - t0:.0f}s)", flush=True)
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        path = out / f"{suite}.csv"
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f"{suite}: {len(rows)} rows -> {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
