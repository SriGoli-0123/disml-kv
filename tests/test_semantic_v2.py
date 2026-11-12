"""v2 additions to the semantic QA metric: answer aliases and per-setting pairing."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from aclkv.semantic import CharNgramEncoder, SemanticScorer

REPO = Path(__file__).resolve().parents[1]


class StubEncoder:
    name = "stub"

    def __init__(self, table):
        self.table = table

    def encode(self, texts):
        out = np.array([self.table[t] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_single_alias_matches_original_similarity():
    s = SemanticScorer(CharNgramEncoder())
    preds, golds = ["Barack Obama", "Eiffel Tower", ""], ["Obama", "Obama", "Paris"]
    assert s.similarity_multi(preds, [[g] for g in golds]) == s.similarity(list(zip(preds, golds)))
    assert s.evaluate_multi(preds, [[g] for g in golds]) == s.evaluate(list(zip(preds, golds)))


def test_best_alias_counts_and_empty_alias_list_scores_zero():
    enc = StubEncoder({"NYC": [1, 0], "New York City": [0.96, 0.28], "Boston": [0, 1]})
    s = SemanticScorer(enc, threshold=0.8)
    assert s.similarity_multi(["NYC"], [["Boston", "New York City"]])[0] > 0.9
    assert s.similarity_multi(["NYC"], [[]]) == [0.0]
    res = s.evaluate_multi(["NYC", "Boston"], [["Boston", "New York City"], ["Boston"]])
    assert res["sem_match"] == 1.0 and res["sem_rescued"] == 0.5


def _run(pol, workload, answers_right):
    recs = []
    for i, ok in enumerate(answers_right):
        recs.append({"ok": True, "req_id": f"r{i:05d}", "qid": f"q{i}", "text": "Paris" if ok else "Rome",
                     "answer": "Paris", "answers": ["Paris", "City of Light"]})
    return {"policy": {"name": pol}, "workload": {"path": f"data/workloads_v2/{workload}.json", "config": {"share_rate": 0.5}},
            "args": {"concurrency": 16, "kv_gib": 2.0, "arrival": "closed", "rate": 8.0},
            "server": {"model": "m"}, "summary": {}, "records": recs}


def test_v2_runs_are_paired_within_their_own_workload(tmp_path):
    """Two v2 workloads share share_rate=0.5 and the same request ids. Pairing by
    share rate (the v1 key) would mix them; pairing by setting must not."""
    runs = {"A_ACLScope": _run("ACLScope", "wlA", [True, True]), "A_ACLKV": _run("ACLKV", "wlA", [False, False]),
            "B_ACLScope": _run("ACLScope", "wlB", [False, False]), "B_ACLKV": _run("ACLKV", "wlB", [True, True])}
    for name, d in runs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(d))
    out = tmp_path / "sem"
    subprocess.run([sys.executable, str(REPO / "scripts/semantic_eval.py"), "--backend", "char",
                    "--bench-glob", str(tmp_path / "*.json"), "--out-prefix", str(out)], check=True,
                   capture_output=True, text=True, cwd=REPO)
    md = (tmp_path / "sem.md").read_text()
    assert "Paired difference vs ACLScope" in md          # baseline auto-selected for v2
    row = next(line for line in md.splitlines() if line.startswith("| ACLKV |") and "[" in line)
    d_em = float(row.split("|")[4])
    assert d_em == 0.0                                    # -1 on wlA and +1 on wlB, correctly paired
