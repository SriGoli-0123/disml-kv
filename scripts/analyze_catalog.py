#!/usr/bin/env python
"""Summaries for the catalog branch (simulator results in results/catalog/).

    python scripts/analyze_catalog.py dev     # tuning table (dev seed): every catalog setting vs baselines
    python scripts/analyze_catalog.py test    # pre-registered pass/fail verdict (docs/CATALOG.md)

"net" = cached prompt tokens minus the tokens the catalog publisher prefilled,
as a % of honest prompt tokens.  "inflight" = mean number of other requests
holding KV when a request is admitted (memory-limited concurrency).
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path("results/catalog")
BASELINE_EVICTIONS = ("lru", "lfu", "gdsf", "arc")


def rows(name):
    return list(csv.DictReader(open(ROOT / f"{name}.csv")))


def label(r):
    if r["policy"] == "Catalog":
        return "Catalog[" + r["catalog"] + "]"
    return r["policy"] + ("" if r.get("eviction", "lru") in ("", "lru") else "+" + r["eviction"].upper())


def f(x):
    return float(x)


def dev():
    lm = [r for r in rows("lowmem") if r["model"] == "Qwen/Qwen2.5-7B-Instruct"]
    cd = rows("catalog_dev")
    kvs = sorted({r["kv_gib"] for r in cd}, key=float)
    table = defaultdict(dict)
    for r in lm + cd:
        table[label(r)][r["kv_gib"]] = r
    base = {kv: max(f(table[f"ACLKV{'' if e == 'lru' else '+' + e.upper()}"][kv]["net_cached_pct"])
                    for e in BASELINE_EVICTIONS) for kv in kvs}
    print("Qwen2.5-7B geometry, dev seed hq_main_s0, C=16.  net cached % (inflight) [publish %]")
    print(f"{'setting':78s}" + "".join(f"{kv + ' GiB':>22s}" for kv in kvs) + f"{'mean margin vs best baseline':>30s}")
    order = sorted(table, key=lambda k: (k.startswith("Catalog"), -st.mean(f(table[k][kv]["net_cached_pct"]) for kv in kvs)))
    for k in order:
        line = f"{k:78s}"
        for kv in kvs:
            r = table[k][kv]
            line += f"{f(r['net_cached_pct']):8.2f} ({f(r['mean_inflight']):4.1f}) [{f(r['publish_pct']):4.2f}]"
        margin = st.mean(f(table[k][kv]["net_cached_pct"]) - base[kv] for kv in kvs)
        line += f"{margin:+30.2f}"
        print(line)
    print("best baseline per KV:", {kv: round(v, 2) for kv, v in base.items()})


def test():
    cfg = json.load(open("configs/v2.json"))["catalog_test"]
    rs = [r for r in rows("catalog_test") if r["suite"] == "catalog_test" and "net_cached_pct" in r and r["net_cached_pct"]]
    att = [r for r in rows("catalog_test") if r.get("singled_out", "") != ""]
    cells = sorted({(r["model"], r["kv_gib"]) for r in rs}, key=lambda c: (c[0], float(c[1])))
    by = defaultdict(list)
    for r in rs:
        by[(label(r), r["model"], r["kv_gib"])].append(r)
    names = ["ACLKV"] + [f"ACLKV+{e.upper()}" for e in BASELINE_EVICTIONS[1:]]
    from aclkv.catalog import CatalogConfig
    cat = "Catalog[" + CatalogConfig.parse(cfg["catalog"]).tag() + "]"
    sec = "Catalog[" + CatalogConfig.parse(cfg["catalog_secondary"]).tag() + "]"
    wins, margins, cap_ok = 0, [], True
    print("primary:", cat, "| secondary (descriptive):", sec)
    print(f"{'model':28s}{'KV':>5s}{'catalog net':>13s}{'best baseline':>26s}{'margin':>8s}{'inflight cat/ACLKV':>20s}"
          f"{'ACLKV(LRU)':>12s}{'secondary net (inflight)':>27s}")
    for m, kv in cells:
        mean = lambda k, col: st.mean(f(r[col]) for r in by[(k, m, kv)])
        for k in names + [cat, sec]:
            assert len(by[(k, m, kv)]) == len(cfg["workloads"]), (k, m, kv, len(by[(k, m, kv)]))
        best = max(names, key=lambda k: mean(k, "net_cached_pct"))
        c, b = mean(cat, "net_cached_pct"), mean(best, "net_cached_pct")
        ci, li = mean(cat, "mean_inflight"), mean("ACLKV", "mean_inflight")
        wins += c > b
        margins.append(c - b)
        cap_ok &= ci >= 0.95 * li
        print(f"{m.split('/')[-1]:28s}{kv:>5s}{c:13.2f}{b:14.2f} ({best:>10s}){c - b:+8.2f}{ci:10.1f}/{li:4.1f}"
              f"{mean('ACLKV', 'net_cached_pct'):12.2f}{mean(sec, 'net_cached_pct'):19.2f} ({mean(sec, 'mean_inflight'):4.1f})")
    unauth = sum(int(r["unauthorized_hits"]) for r in rs if "Global" not in r["policy"])
    priv = {}
    for r in att:
        priv[label(r)] = {k: r[k] for k in ("singled_out", "min_anon_observed", "order_singled_out", "order_min_anon",
                                            "query_revealed", "bypass_hits", "honest_cached_pct", "publish_pct")}
    print("\nprivacy (insider harness):")
    for k, v in priv.items():
        print(f"  {k:60s} {v}")
    pc = priv.get(cat, {})
    anon_ok = lambda v: int(v) == -1 or int(v) >= 2      # -1: nothing revealed at all
    priv_ok = (unauth == 0 and pc and float(pc["singled_out"]) == 0 and float(pc["order_singled_out"]) == 0
               and anon_ok(pc["min_anon_observed"]) and anon_ok(pc["order_min_anon"]))
    crit = {
        "efficiency: wins in >= 9 of 12 cells": wins >= 9,
        "efficiency: mean margin >= +0.5 points": st.mean(margins) >= 0.5,
        "capacity: inflight >= 95% of ACLKV in every cell": cap_ok,
        "privacy: 0 unauthorized hits, nobody singled out (cache and order)": bool(priv_ok),
    }
    print(f"\nwins {wins}/{len(cells)}, mean margin {st.mean(margins):+.2f}, unauthorized hits {unauth}")
    for k, v in crit.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("VERDICT:", "PASS" if all(crit.values()) else "FAIL")


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
