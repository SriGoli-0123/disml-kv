#!/usr/bin/env python
"""Figures (PDF), LaTeX tables and number macros for the paper, generated
from the result files so that no number in the paper is hand-copied.

    python scripts/make_paper_assets.py          # -> paper/figures, paper/tables, paper/numbers.tex

Inputs: results/v2/{main,privacy,ablation,pic,scaling}.csv,
        results/v2/analysis_live_v1.json, data/workloads_v2/*.json,
        results/v2_live/summary.json (optional; GPU runs fill the TBD cells).
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path("results/v2")
P = Path("paper")
FIG, TAB = P / "figures", P / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# fixed entity -> colour mapping (validated categorical palette, light mode; same in every figure)
ENT = {"ACLKV": "#2a78d6", "ACLScope+Tree": "#eb6834", "ACLScope": "#1baf7a", "Divergence": "#eda100",
       "PerUser": "#e87ba4", "Global+Tree": "#008300"}
MARK = {"ACLKV": "o", "ACLScope+Tree": "s", "ACLScope": "^", "Divergence": "D", "PerUser": "v", "Global+Tree": "P"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": "#b9b8b3", "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NUM: dict[str, str] = {}


def rows(name):
    p = R / f"{name}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(x):
    return float(x)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    return st.pstdev(xs) if len(xs) > 1 else 0.0


_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 8: "Eight", 10: "Ten", 16: "Sixteen",
          20: "Twenty", 32: "ThirtyTwo", 40: "Forty", 100: "Hundred", 500: "FiveHundred"}


def macro_name(name: str) -> str:
    """LaTeX macro names are letters only: spell out embedded numbers (kv16 -> kvSixteen)."""
    import re

    digits = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    return re.sub(r"\d+", lambda m: _WORDS.get(int(m.group()), "".join(digits[int(c)] for c in m.group())), name)


def num(name, value, fmt="{:.1f}"):
    NUM[macro_name(name)] = fmt.format(value) if not isinstance(value, str) else value


def write_tab(name, body):
    (TAB / f"{name}.tex").write_text(body)


def direct_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=6.5, color=INK2, va="center")


# ------------------------------------------------------------------ main table
MAIN = rows("main")
POL_ORDER = ["NoCache", "PerUser", "Divergence", "Binary", "ACLScope", "ACLScope+Tree", "ACLKV", "Global", "Global+Tree"]
NAME = {"NoCache": "No caching", "PerUser": "Per-user salt", "Divergence": "Divergence salt",
        "Binary": "Public/private split", "ACLScope": "ACL scopes", "ACLScope+Tree": "ACL scopes + tree order",
        "ACLKV": r"\textsc{AclKV} ($k{=}3$)", "Global": "Global sharing", "Global+Tree": "Global + tree order"}
GUARD = {"NoCache": "full", "PerUser": "per user", "Divergence": "per user$^{\\dagger}$", "Binary": "public/user",
         "ACLScope": "ACL", "ACLScope+Tree": "ACL", "ACLKV": "ACL + $k$-anon.", "Global": "none", "Global+Tree": "none"}


def cell(workloads, pol, kv="2"):
    v = [f(r["cached_pct"]) for r in MAIN if r["workload"] in workloads and r["policy"] == pol and r["kv_gib"] == kv]
    return v


HQ = ["hq_main_s0", "hq_main_s1", "hq_main_s2"]
MQ = ["mq_main_s0", "mq_main_s1", "mq_main_s2"]
PRIV = rows("privacy")


def priv(pol, key, wls=HQ):
    v = [f(r[key]) for r in PRIV if r["policy"] == pol and r["workload"] in wls]
    return mean(v) if v else float("nan")


if MAIN:
    # dev/test agreement
    gaps = []
    for pol in POL_ORDER:
        d = cell(["hq_main_s0"], pol)[0]
        t = mean(cell(["hq_main_s1", "hq_main_s2"], pol))
        gaps.append(abs(d - t))
    num("devTestGap", max(gaps), "{:.2f}")
    lines = [r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Cached prompt tokens (\%)} & & Insider \\ \cmidrule(lr){3-5}",
             r"Policy & Isolation & HotpotQA & MuSiQue & Pinned & vs.\ per-user & singled-out (\%) \\", r"\midrule"]
    peruser_hq = mean(cell(HQ, "PerUser"))
    for pol in POL_ORDER:
        hq, mq, pin = cell(HQ, pol), cell(MQ, pol), cell(["hq_pinned_s0"], pol)
        rel = mean(hq) / peruser_hq if peruser_hq > 0 and pol != "NoCache" else float("nan")
        priv_pol = {"Global": "Global", "Global+Tree": "Global+Tree"}.get(pol, pol)
        so = priv(priv_pol, "singled_out") * 100 if pol != "NoCache" else 0.0
        bold = pol == "ACLKV"
        b = (lambda s: r"\textbf{" + s + "}") if bold else (lambda s: s)
        sep = r"\midrule" + "\n" if pol == "Global" else ""
        lines.append(sep + f"{b(NAME[pol])} & {GUARD[pol]} & {b(f'{mean(hq):.2f}')}$\\pm${sd(hq):.2f} & "
                     f"{b(f'{mean(mq):.2f}')}$\\pm${sd(mq):.2f} & {mean(pin):.1f} & "
                     f"{'--' if rel != rel else f'{rel:.1f}' + r'$\times$'} & {so:.1f} \\\\")
        key = pol.replace("+", "").replace("-", "")
        num(f"hq{key}", mean(hq), "{:.2f}")
        num(f"mq{key}", mean(mq), "{:.2f}")
        num(f"pin{key}", mean(pin), "{:.1f}")
        num(f"exact{key}", mean(cell(["hq_exact_s0"], pol)), "{:.1f}")
        num(f"hqFour{key}", mean(cell(HQ, pol, "4")), "{:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tab("main", "\n".join(lines) + "\n")
    num("aclkvRetainHQ", 100 * mean(cell(HQ, "ACLKV")) / mean(cell(HQ, "ACLScope+Tree")), "{:.0f}")
    num("aclkvRetainMQ", 100 * mean(cell(MQ, "ACLKV")) / mean(cell(MQ, "ACLScope+Tree")), "{:.0f}")
    num("aclkvVsPerUserHQ", mean(cell(HQ, "ACLKV")) / mean(cell(HQ, "PerUser")), "{:.1f}")
    num("aclkvVsPerUserMQ", mean(cell(MQ, "ACLKV")) / mean(cell(MQ, "PerUser")), "{:.1f}")
    num("treeGainHQ", mean(cell(HQ, "ACLScope+Tree")) - mean(cell(HQ, "ACLScope")), "{:.2f}")
    num("privTailCostHQ", mean(cell(HQ, "Global")) - mean(cell(HQ, "ACLScope")), "{:.2f}")
    num("privTailCostExact", mean(cell(["hq_exact_s0"], "Global")) - mean(cell(["hq_exact_s0"], "ACLScope")), "{:.2f}")
    num("meanPromptHQ", mean([f(r["mean_prompt_tokens"]) for r in MAIN if r["workload"] in HQ]), "{:.0f}")
    num("meanPromptMQ", mean([f(r["mean_prompt_tokens"]) for r in MAIN if r["workload"] in MQ]), "{:.0f}")
    num("padFrac", 100 * mean([f(r["pad_frac"]) for r in MAIN if r["workload"] in HQ]), "{:.1f}")

# ------------------------------------------------------------------ privacy table + frontier
if PRIV:
    prow = [("Global", "Global sharing (insecure)"), ("PerUser", "Per-user salt"), ("Divergence", "Divergence salt"),
            ("Binary", "Public/private split"), ("B3", "ACL scopes, inherited query (v1)"),
            ("ACLScope", "ACL scopes"), ("ACLScope+Tree", "ACL scopes + tree order"),
            ("acl_scope/tree:k=2", r"\textsc{AclKV}, $k{=}2$"), ("ACLKV", r"\textsc{AclKV}, $k{=}3$"),
            ("acl_scope/tree:k=5", r"\textsc{AclKV}, $k{=}5$"), ("acl_scope/tree:k=10", r"\textsc{AclKV}, $k{=}10$")]
    lines = [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{HotpotQA insider probes (\%)} & MuSiQue & Cached \\ \cmidrule(lr){2-5}",
             r"Policy & reveal & single & anon$<$5 & query & single (\%) & (\%) \\", r"\midrule"]
    for pol, label in prow:
        b = (lambda s: r"\textbf{" + s + "}") if pol == "ACLKV" else (lambda s: s)
        lines.append(f"{b(label)} & {100*priv(pol,'revealed'):.1f} & {b(f'{100*priv(pol,chr(115)+'ingled_out'):.1f}')} & "
                     f"{100*priv(pol,'anon_lt5'):.1f} & {100*priv(pol,'query_revealed'):.1f} & "
                     f"{100*priv(pol,'singled_out',['mq_main_s0']):.1f} & {priv(pol,'honest_cached_pct'):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tab("privacy", "\n".join(lines) + "\n")
    num("singledACLScope", 100 * priv("ACLScope", "singled_out"))
    num("singledACLScopeMQ", 100 * priv("ACLScope", "singled_out", ["mq_main_s0"]))
    num("singledBinary", 100 * priv("Binary", "singled_out"))
    num("queryRevealedVone", 100 * priv("B3", "query_revealed"))
    num("revealACLKV", 100 * priv("ACLKV", "revealed"))
    num("revealKten", 100 * priv("acl_scope/tree:k=10", "revealed"))
    num("anonLtFiveACLKV", 100 * priv("ACLKV", "anon_lt5"))
    num("singledKtwo", 100 * priv("acl_scope/tree:k=2", "singled_out"))
    num("bypassGlobal", sum(f(r["bypass_hits"]) for r in PRIV if r["policy"] == "Global" and r["workload"] in HQ), "{:,.0f}")
    num("nProbes", sum(f(r["probes"]) for r in PRIV if r["policy"] == "ACLKV" and r["workload"] in HQ), "{:,.0f}")
    num("minAnonACLKV", min(f(r["min_anon_observed"]) for r in PRIV if r["policy"] == "ACLKV"), "{:.0f}")

    fig, ax = plt.subplots(figsize=(3.15, 2.25))
    ks = [("ACLScope+Tree", 1), ("acl_scope/tree:k=2", 2), ("ACLKV", 3), ("acl_scope/tree:k=5", 5), ("acl_scope/tree:k=10", 10)]
    xs = [100 * priv(p, "revealed") for p, _ in ks]
    ys = [priv(p, "honest_cached_pct") for p, _ in ks]
    ax.plot(xs, ys, "-o", color=ENT["ACLKV"], zorder=3, label=r"ACL scopes + $k$-anonymous promotion")
    for (p, k), x, y in zip(ks, xs, ys):
        ax.annotate(f"k={k}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=6.5, color=INK)
    base = [("PerUser", "per-user"), ("Divergence", "divergence"), ("Binary", "public/private"),
            ("ACLScope", "ACL scopes"), ("Global+Tree", "global (insecure)")]
    for p, lab in base:
        x, y = 100 * priv(p, "revealed"), priv(p, "honest_cached_pct")
        ax.plot([x], [y], "o", color="#8a8984", mfc="white", mew=1.2, zorder=2)
        off, ha = {"global (insecure)": ((-6, 3), "right"), "ACL scopes": ((0, -11), "center"),
                   "per-user": ((5, 3), "left"), "divergence": ((5, -6), "left"),
                   "public/private": ((0, -11), "center")}[lab]
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2, ha=ha)
    ax.set_xlabel("Insider probes revealing a colleague's request (%)")
    ax.set_ylabel("Cached prompt tokens (%)")
    ax.set_xlim(-6, 108)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "frontier.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ------------------------------------------------------------------ scaling figure
SC = rows("scaling")
if SC:
    series = ["PerUser", "ACLScope", "ACLScope+Tree", "ACLKV"]
    label = {"PerUser": "per-user salt", "ACLScope": "ACL scopes", "ACLScope+Tree": "ACL + tree", "ACLKV": "AclKV (k=3)"}
    panels = [("kv_budget", lambda r: f(r["kv_gib"]), "KV-cache budget (GiB)", True),
              ("users", lambda r: int(r["n_users"]), "Tenants (users; groups scale)", True),
              ("query_noise", lambda r: f(r["query_noise"]), "Query paraphrase noise", False)]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.05))
    for ax, (var, key, xl, logx) in zip(axes, panels):
        sel = [r for r in SC if r["variant"] == var]
        for pol in series:
            pts = sorted({key(r) for r in sel if r["policy"] == pol})
            ys = [mean([f(r["cached_pct"]) for r in sel if r["policy"] == pol and key(r) == x]) for x in pts]
            ax.plot(pts, ys, marker=MARK[pol], color=ENT[pol], label=label[pol])
            if var == "kv_budget":
                direct_label(ax, pts[-1], ys[-1], label[pol])
        if logx:
            ax.set_xscale("log", base=2)
            ticks = sorted({key(r) for r in sel})
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel(xl)
        ax.set_ylim(bottom=0)
    axes[0].set_xlim(0.8, 60)
    axes[0].set_ylabel("Cached prompt tokens (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "scaling.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    g = lambda var, pol, k: mean([f(r["cached_pct"]) for r in SC if r["variant"] == var and r["policy"] == pol and k(r)])
    for u in (20, 100, 500):
        num(f"usersPerUser{u}", g("users", "PerUser", lambda r: int(r["n_users"]) == u), "{:.2f}")
        num(f"usersACLScope{u}", g("users", "ACLScope", lambda r: int(r["n_users"]) == u), "{:.2f}")
        num(f"usersRatio{u}", g("users", "ACLScope", lambda r: int(r["n_users"]) == u) /
            g("users", "PerUser", lambda r: int(r["n_users"]) == u), "{:.1f}")
    for kv in (1, 2, 4, 8, 16):
        for pol in ("PerUser", "ACLScope", "ACLScope+Tree", "ACLKV"):
            num(f"kv{kv}{pol.replace('+', '')}", g("kv_budget", pol, lambda r: f(r["kv_gib"]) == kv), "{:.1f}")
    num("kanonRetainSixteen", 100 * g("kv_budget", "ACLKV", lambda r: f(r["kv_gib"]) == 16) /
        g("kv_budget", "ACLScope+Tree", lambda r: f(r["kv_gib"]) == 16), "{:.0f}")
    for m, tag in (("Qwen2.5-7B-Instruct", "Seven"), ("Qwen3-8B", "Eight"), ("Qwen2.5-32B-Instruct", "ThirtyTwo")):
        num(f"geom{tag}", g("model_geometry", "ACLScope+Tree", lambda r: r["model"].endswith(m)), "{:.1f}")
    for d in (10, 20, 40):
        num(f"docs{d}", g("prompt_length", "ACLScope+Tree", lambda r: int(r["docs_per_request"]) == d and f(r["kv_gib"]) == 8), "{:.1f}")
    for mx, tag in (("0.2/0.4/0.4", "Restrictive"), ("0.5/0.3/0.2", "Default"), ("0.8/0.15/0.05", "Open")):
        num(f"mix{tag}", g("acl_mix", "ACLScope+Tree", lambda r: r["acl_mix"] == mx), "{:.1f}")
    lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
             r"Setting & Per-user & ACL scopes & ACL + tree & \textsc{AclKV} & Global + tree \\", r"\midrule"]
    allp = ["PerUser", "ACLScope", "ACLScope+Tree", "ACLKV", "Global+Tree"]
    for var, key, vals, fmt in [("users", lambda r: int(r["n_users"]), (20, 100, 500), "{} users"),
                                ("prompt_length", lambda r: (int(r["docs_per_request"]), int(f(r["kv_gib"]))),
                                 ((10, 8), (20, 8), (40, 8)), "{0[0]} docs, 8 GiB"),
                                ("kv_budget", lambda r: f(r["kv_gib"]), (1.0, 4.0, 16.0), "{:g} GiB"),
                                ("model_geometry", lambda r: r["model"].split("/")[1],
                                 ("Qwen2.5-7B-Instruct", "Qwen3-8B", "Qwen2.5-32B-Instruct"), "{} (8 GiB)"),
                                ("acl_mix", lambda r: r["acl_mix"], ("0.2/0.4/0.4", "0.8/0.15/0.05"), "ACL mix {}"),
                                ("concurrency", lambda r: int(r["concurrency"]), (1, 32), "concurrency {}")]:
        for v in vals:
            cells = [mean([f(r["cached_pct"]) for r in SC if r["variant"] == var and r["policy"] == p and key(r) == v]) for p in allp]
            lines.append(fmt.format(v).replace("_", r"\_") + " & " + " & ".join(f"{c:.2f}" for c in cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write_tab("scaling", "\n".join(lines) + "\n")

# ------------------------------------------------------------------ ablations
AB = rows("ablation")
if AB:
    def ab(var, wls=None):
        v = [r for r in AB if r["variant"] == var and (wls is None or r["workload"] in wls)]
        return v
    order_rows = [("order=retrieval", "Retrieval order"), ("order=acl_aware(v1 B4)", "Broad-to-narrow (v1 B4)"),
                  ("order=reuse_aware(v1 B5)", "Popularity-canonical (v1 B5)"),
                  ("order=reuse_aware+anchor(v1 B6)", "+ relevance anchor (v1 B6)"),
                  ("order=lcr", "LongContextReorder"), ("order=edge-layout", "Edge layout"),
                  ("order=tree", "Prefix tree (CacheWeaver-style)"), ("order=tree,k=3", r"Prefix tree, $k{=}3$"),
                  ("order=privacy-aware-tree,k=3", r"Privacy-aware tree, $k{=}3$")]
    lines = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule",
             r"Evidence order & Cached (\%) & Gold mid (\%) & Top-1 at edge (\%) \\", r"\midrule"]
    for var, lab in order_rows:
        rs = ab(var)
        if not rs:
            continue
        lines.append(f"{lab} & {mean([f(r['cached_pct']) for r in rs]):.2f} & {mean([f(r['gold_in_middle_pct']) for r in rs]):.1f} & "
                     f"{mean([f(r['top1_at_edge_pct']) for r in rs]):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tab("ablation_order", "\n".join(lines) + "\n")
    om = lambda var, k="cached_pct": mean([f(r[k]) for r in ab(var)])
    num("ordRetrieval", om("order=retrieval"), "{:.2f}")
    num("ordTree", om("order=tree"), "{:.2f}")
    num("ordEdge", om("order=edge-layout"), "{:.2f}")
    num("ordLCR", om("order=lcr"), "{:.2f}")
    num("ordBsix", om("order=reuse_aware+anchor(v1 B6)"), "{:.2f}")
    num("ordTreeK", om("order=tree,k=3"), "{:.2f}")
    num("ordPrivTreeK", om("order=privacy-aware-tree,k=3"), "{:.2f}")
    num("goldMidRetrieval", om("order=retrieval", "gold_in_middle_pct"))
    num("goldMidLCR", om("order=lcr", "gold_in_middle_pct"))
    num("goldMidTree", om("order=tree", "gold_in_middle_pct"))
    num("topEdgeTree", om("order=tree", "top1_at_edge_pct"))
    tail = {(r["policy"], r["workload"]): f(r["cached_pct"]) for r in AB if r["variant"].startswith("tail=")}
    pairs = [("ACLScope", "acl_scope/retrieval:tail=inherit"), ("ACLScope+Tree", "acl_scope/tree:tail=inherit"),
             ("ACLKV", "acl_scope/tree:k=3,tail=inherit")]
    diffs = [tail[(b, w)] - tail[(a, w)] for a, b in pairs for (p, w) in tail if p == a and (b, w) in tail]
    num("tailCostMax", max(diffs), "{:.2f}")
    al = {r["policy"] + r["variant"]: f(r["cached_pct"]) for r in AB if r["variant"].startswith("align=")}
    num("alignOnACLKV", al["ACLKValign=True"], "{:.2f}")
    num("alignOffACLKV", al["ACLKValign=False"], "{:.2f}")
    num("alignOnACLScope", al["ACLScopealign=True"], "{:.2f}")
    num("alignOffACLScope", al["ACLScopealign=False"], "{:.2f}")
    tw = {r["variant"]: f(r["cached_pct"]) for r in AB if r["variant"].startswith("tree_window") and r["policy"] == "ACLScope+Tree"}
    num("treeWinMin", min(tw.values()), "{:.2f}")
    num("treeWinMax", max(tw.values()), "{:.2f}")

PIC = rows("pic")
if PIC:
    def pic(w, mode, scope="", r=""):
        v = [f(x["cached_pct"]) for x in PIC if x["workload"] == w and x["mode"] == mode and x["pic_scoping"] == scope
             and x["pic_recompute"] == r and (mode == "pic" or x["policy"] == "ACLScope+Tree")]
        return v[0] if v else float("nan")
    lines = [r"\begin{tabular}{@{}l r r@{}}", r"\toprule", r"Reuse mechanism & HotpotQA & MuSiQue \\", r"\midrule",
             f"Prefix, ACL scopes + tree order & {pic('hq_main_s0','prefix'):.2f} & {pic('mq_main_s0','prefix'):.2f} \\\\"]
    for scope, lab in (("acl", "own-ACL scope"), ("user", "per-user scope"), ("global", "no scope (insecure)")):
        for r_ in ("0.15", "0.3"):
            lines.append(f"Position-indep., {lab}, $r{{=}}{float(r_):g}$ & {pic('hq_main_s0','pic',scope,r_):.2f} & {pic('mq_main_s0','pic',scope,r_):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tab("pic", "\n".join(lines) + "\n")
    num("picAclHQ", pic("hq_main_s0", "pic", "acl", "0.15"), "{:.2f}")
    num("picAclMQ", pic("mq_main_s0", "pic", "acl", "0.15"), "{:.2f}")
    num("prefixTreeHQ", pic("hq_main_s0", "prefix"), "{:.2f}")
    num("prefixTreeMQ", pic("mq_main_s0", "prefix"), "{:.2f}")
    num("picGlobalUnauthMQ", sum(f(x["unauthorized_hits"]) for x in PIC if x["workload"] == "mq_main_s0"
                                  and x["pic_scoping"] == "global" and x["pic_recompute"] == "0.15"), "{:.0f}")

# ------------------------------------------------------------------ live v1 pilot
LV = R / "analysis_live_v1.json"
if LV.exists():
    lv = json.load(open(LV))
    fid = lv["fidelity"]
    num("fidN", fid["n_runs"], "{:d}")
    num("fidMAE", fid["mean_abs_err_pp"], "{:.3f}")
    num("fidMax", fid["max_abs_err_pp"], "{:.2f}")
    num("fidR", fid["pearson_r"], "{:.5f}")
    fit16, fit4 = lv["ttft_fits"]["c16_ttft_mean_ms"], lv["ttft_fits"]["c4_ttft_mean_ms"]
    num("ttftSlopeSixteen", fit16["ms_per_1k_recomputed_tokens"] / 1000, "{:.2f}")
    num("ttftRtwoSixteen", fit16["r2"], "{:.3f}")
    num("ttftSlopeFour", fit4["ms_per_1k_recomputed_tokens"] / 1000, "{:.3f}")
    num("ttftRtwoFour", fit4["r2"], "{:.3f}")
    h = lv["headline_live_kv2_c16"]["0.5"]
    num("liveBoneTTFT", h["B1"]["ttft_p50_ms"], "{:.0f}")
    num("liveBsixTTFT", h["B6"]["ttft_p50_ms"], "{:.0f}")
    num("liveBthreeTTFT", h["B3"]["ttft_p50_ms"], "{:.0f}")
    num("liveBoneRPS", h["B1"]["rps"], "{:.1f}")
    num("liveBsixRPS", h["B6"]["rps"], "{:.1f}")
    num("liveBsixCached", h["B6"]["cached_pct"], "{:.1f}")
    num("liveBoneCached", h["B1"]["cached_pct"], "{:.1f}")
    q = lv["quality"]
    lines = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
             r"Overlap & ACL-scope F1 & $\Delta$F1 v1-B5 & $\Delta$F1 v1-B6 [95\% CI] & $p$ \\", r"\midrule"]
    lo, hi = [], []
    for w in sorted(q):
        share = w.split("share")[1][:4]
        b5, b6 = q[w]["B5"]["f1"], q[w]["B6"]["f1"]
        lo.append(b6["ci_low"]); hi.append(b6["ci_high"])
        lines.append(f"{float(share):.2f} & {q[w]['B3_f1']:.3f} & {b5['diff']:+.3f} & {b6['diff']:+.3f} "
                     f"[{b6['ci_low']:+.3f}, {b6['ci_high']:+.3f}] & {b6['p_boot']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tab("live_v1_quality", "\n".join(lines) + "\n")
    num("vOneCIlow", min(lo), "{:+.3f}")
    num("vOneCIhigh", max(hi), "{:+.3f}")
    rows_f = lv["fidelity_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1))
    ax = axes[0]
    xs = [x["live_cached_pct"] for x in rows_f]
    ys = [x["sim_cached_pct"] for x in rows_f]
    ax.plot([0, 52], [0, 52], color="#b9b8b3", lw=0.8, zorder=1)
    ax.plot(xs, ys, "o", ms=3.2, color=ENT["ACLKV"], alpha=0.85, zorder=2, label=f"{len(xs)} live runs")
    ax.set_xlabel("Live vLLM cached tokens (%)")
    ax.set_ylabel("Simulated cached tokens (%)")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.05, f"MAE {fid['mean_abs_err_pp']:.3f} pp\nr = {fid['pearson_r']:.5f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax = axes[1]
    tt = lv["ttft_rows"]
    for c, col, mk in ((4, ENT["ACLKV"], "o"), (16, ENT["ACLScope+Tree"], "s")):
        sel = [x for x in tt if x["c"] == c]
        ax.plot([x["recomp_per_req"] for x in sel], [x["ttft_mean_ms"] for x in sel], mk, ms=3.0, color=col,
                alpha=0.85, label=f"{c} concurrent")
        fit = lv["ttft_fits"][f"c{c}_ttft_mean_ms"]
        xx = [min(x["recomp_per_req"] for x in sel), max(x["recomp_per_req"] for x in sel)]
        ax.plot(xx, [fit["intercept_ms"] + fit["ms_per_1k_recomputed_tokens"] * v / 1000 for v in xx], color=col, lw=1.0)
        direct_label(ax, xx[1], fit["intercept_ms"] + fit["ms_per_1k_recomputed_tokens"] * xx[1] / 1000,
                     f"R²={fit['r2']:.3f}", dy=-6 if c == 4 else 0)
    ax.set_xlabel("Recomputed prompt tokens per request")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(right=max(x["recomp_per_req"] for x in tt) * 1.18)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fidelity.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ------------------------------------------------------------------ stability vs rank
def stability(names, window=25):
    from aclkv.workload import Workload

    hit, tot = defaultdict(int), defaultdict(int)
    for n in names:
        p = Path("data/workloads_v2") / f"{n}.json"
        if not p.exists():
            return None
        w = Workload.load(p)
        recent = []
        for r in w.requests:
            seen = set().union(*recent) if recent else set()
            for d in r.docs:
                tot[d["rank"]] += 1
                hit[d["rank"]] += d["chunk_id"] in seen
            recent.append({d["chunk_id"] for d in r.docs})
            if len(recent) > window:
                recent.pop(0)
    return {k: hit[k] / tot[k] for k in sorted(tot)}


s_hq, s_mq = stability(HQ, 200), stability(MQ, 200)
s_hq_short = stability(HQ, 25)
if s_hq_short:
    num("stabTopHQShort", 100 * s_hq_short[0])
    num("stabTenHQShort", 100 * s_hq_short[9])
if s_hq and s_mq:
    fig, ax = plt.subplots(figsize=(3.15, 1.9))
    for name, s, col, mk in (("HotpotQA (10 passages)", s_hq, ENT["ACLKV"], "o"), ("MuSiQue (20 passages)", s_mq, ENT["ACLScope+Tree"], "s")):
        ax.plot([k + 1 for k in s], [100 * v for v in s.values()], marker=mk, color=col, label=name)
    ax.set_xlabel("Retrieval rank of the passage")
    ax.set_ylabel("Retrieved by one of the\nprevious 200 requests (%)")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "stability.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    num("stabTopHQ", 100 * s_hq[0])
    num("stabTenHQ", 100 * s_hq[9])
    num("stabTopMQ", 100 * s_mq[0])
    num("stabLastMQ", 100 * s_mq[max(s_mq)])

# ------------------------------------------------------------------ repeat gaps and live-probe timing
def repeat_gap(names):
    from aclkv.workload import Workload

    gaps = []
    for n in names:
        p = Path("data/workloads_v2") / f"{n}.json"
        if not p.exists():
            return None
        last = {}
        for i, r in enumerate(Workload.load(p).requests):
            if r.qid in last:
                gaps.append(i - last[r.qid])
            last[r.qid] = i
    return st.median(gaps) if gaps else None


g_hq, g_mq = repeat_gap(HQ), repeat_gap(MQ)
if g_hq:
    num("repeatGapHQ", g_hq, "{:.0f}")
    num("repeatGapMQ", g_mq, "{:.0f}")
if LV.exists():
    num("noiseFloorMax", max(v["noise_floor_answers_differing_B0_vs_B3"] for v in lv["quality"].values()), "{:d}")
SP = Path("results/security_probe_kv2.json")
if SP.exists():
    sp = json.load(open(SP))["policies"]
    num("probeColdUB", sp["UB"]["summary"]["victim_cold_ttft"]["p50"] * 1000, "{:.0f}")
    num("probeBypassUB", sp["UB"]["summary"]["no_salt"]["ttft"]["p50"] * 1000, "{:.0f}")
    num("probeBypassBthree", sp["B3"]["summary"]["no_salt"]["ttft"]["p50"] * 1000, "{:.0f}")
if PIC:
    num("picScopeCost", max(abs(pic(w, "pic", "acl", r_) - pic(w, "pic", "global", r_))
                            for w in ("hq_main_s0", "mq_main_s0") for r_ in ("0.15", "0.3")), "{:.2f}")
if SC:
    nz = lambda v: g("query_noise", "ACLScope+Tree", lambda r: f(r["query_noise"]) == v) - g("query_noise", "ACLScope", lambda r: f(r["query_noise"]) == v)
    num("treeGainExact", nz(0.0), "{:.2f}")
    num("treeGainNoiseFifteen", nz(0.15), "{:.2f}")
    num("treeGainNoiseFifty", nz(0.5), "{:.2f}")

# ------------------------------------------------------------------ workload statistics
wl_stats = {}
for n in HQ + MQ + ["hq_exact_s0"]:
    p = Path("data/workloads_v2") / f"{n}.json"
    if p.exists():
        wl_stats[n] = json.load(open(p))["stats"]
if wl_stats:
    num("distinctHQ", mean([wl_stats[n]["distinct_questions"] for n in HQ]), "{:,.0f}")
    num("distinctMQ", mean([wl_stats[n]["distinct_questions"] for n in MQ]), "{:,.0f}")
    num("recallHQ", mean([wl_stats[n]["gold_recall_at_k"] for n in HQ]), "{:.2f}")
    num("recallMQ", mean([wl_stats[n]["gold_recall_at_k"] for n in MQ]), "{:.2f}")
    num("exactSetHQ", 100 * mean([wl_stats[n]["exact_doc_set_seen_before"] for n in HQ]), "{:.0f}")
    num("docsSeenHQ", 100 * mean([wl_stats[n]["docs_seen_before"] for n in HQ]), "{:.0f}")
    num("exactSetExact", 100 * wl_stats["hq_exact_s0"]["exact_doc_set_seen_before"], "{:.0f}")
    num("publicEvidenceHQ", 100 * mean([wl_stats[n]["evidence_acl_mix"]["public"] for n in HQ]), "{:.0f}")

# ------------------------------------------------------------------ dataset statistics
for ds, root, tag in (("hotpotqa", "data/prepared_hotpotqa_full", "HQ"), ("musique", "data/prepared_musique", "MQ")):
    sp = Path(root) / "stats.json"
    if sp.exists():
        d = json.load(open(sp))
        num(f"questions{tag}", d["questions"], "{:,}")
        num(f"passages{tag}", d["chunks"], "{:,}")
        num(f"passageTokens{tag}", d["mean_chunk_tokens"], "{:.0f}")
if LV.exists() and MAIN:
    num("overstate", lv["headline_live_kv2_c16"]["0.5"]["B6"]["cached_pct"] / mean(cell(HQ, "ACLScope+Tree")), "{:.1f}")

# ------------------------------------------------------------------ retriever robustness (appendix)
RT = rows("retriever")
if RT:
    rets = ["bm25", "dense", "hybrid"]
    pols = ["PerUser", "Divergence", "ACLScope", "ACLScope+Tree", "ACLKV", "Global+Tree"]
    lines = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r"Policy & BM25 & Dense & Hybrid \\", r"\midrule"]
    for pol in pols:
        cells = []
        for ret in rets:
            v = [f(r["cached_pct"]) for r in RT if r["policy"] == pol and r["variant"] == f"retriever={ret}"]
            cells.append(f"{mean(v):.2f}" if v else "--")
        lines.append(f"{NAME.get(pol, pol)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tab("retriever", "\n".join(lines) + "\n")
    for ret in rets:
        for pol in ("PerUser", "ACLScope+Tree", "ACLKV"):
            v = [f(r["cached_pct"]) for r in RT if r["policy"] == pol and r["variant"] == f"retriever={ret}"]
            if v:
                num(f"ret{ret.capitalize()}{pol.replace('+', '')}", mean(v), "{:.2f}")

# ------------------------------------------------------------------ GPU (v2 live) table: filled when results exist
LIVE = Path("results/v2_live/summary.json")
live = json.load(open(LIVE)) if LIVE.exists() else None
gpu_pols = [("PerUser", "Per-user salt"), ("Divergence", "Divergence salt"), ("ACLScope", "ACL scopes"),
            ("ACLScope+Tree", "ACL scopes + tree order"), ("ACLKV", r"\textsc{AclKV} ($k{=}3$)"), ("Global", "Global (insecure)")]


def live_cell(pol, key, scale=1.0, fmt="{:.0f}"):
    if not live:
        return r"\tbd"
    v = [x[key] for x in live["closed"] if x["policy"] == pol and x["model"] == "Qwen2.5-7B-Instruct"
         and float(x["kv"]) == 2.0 and x["workload"].startswith("hq_main")]
    return fmt.format(mean(v) * scale) if v else r"\tbd"


lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
         r"Policy & Cached (\%) & TTFT p50 (ms) & TTFT p95 (ms) & req/s & EM / F1 \\", r"\midrule"]
for pol, lab in gpu_pols:
    qa = r"\tbd"
    if live:
        v = [x for x in live["closed"] if x["policy"] == pol and x["model"] == "Qwen2.5-7B-Instruct"
             and float(x["kv"]) == 2.0 and x["workload"].startswith("hq_main")]
        if v:
            qa = f"{mean([x['em'] for x in v]):.3f} / {mean([x['f1'] for x in v]):.3f}"
    lines.append(f"{lab} & {live_cell(pol, 'cached_pct', fmt='{:.1f}')} & {live_cell(pol, 'ttft_p50_ms')} & "
                 f"{live_cell(pol, 'ttft_p95_ms')} & {live_cell(pol, 'req_per_s', fmt='{:.1f}')} & {qa} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tab("live_v2", "\n".join(lines) + "\n")

# ------------------------------------------------------------------ numbers.tex
out = ["% generated by scripts/make_paper_assets.py -- do not edit by hand"]
for k in sorted(NUM):
    if not k.isalpha():
        raise SystemExit(f"macro name must be letters only: {k}")
    out.append(f"\\newcommand{{\\num{k}}}{{{NUM[k]}}}")
(P / "numbers.tex").write_text("\n".join(out) + "\n")
print(f"{len(NUM)} number macros, tables: {sorted(p.name for p in TAB.glob('*.tex'))}, figures: {sorted(p.name for p in FIG.glob('*.pdf'))}")
