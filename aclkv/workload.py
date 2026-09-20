"""Synthetic multi-user RAG workload with controlled ACLs and overlap.

Knobs (all seeded, so every policy sees the *same* requests):

- ``n_users`` / ``n_groups`` / ``groups_per_user``: the directory;
- ``acl_mix``: target proportions of public / group / user-private chunks.
  Every chunk is assigned a class by these proportions; group chunks go to a
  uniformly random group, private chunks to a uniformly random user;
- ``share_rate``: fraction of a request's non-gold evidence slots that are
  filled from a small *hot pool* of popular chunks (Zipf popularity) instead
  of the question's own distractors — this is what creates cross-user
  overlap.  ``0`` means no shared documents beyond chance;
- ``question_pool_size``: how many distinct questions the users draw from; a
  small pool makes the same question (and its gold chunks) repeat.

Every request is *already authorized*: the requester can read all of its
evidence (the gold chunks decide who may ask the question), so the middleware
filter should drop nothing — that is checked and reported.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .acl import ACL, Directory
from .data_prep import BM25Index, Chunk, Question
from .ordering import RetrievedDoc


@dataclass
class WorkloadConfig:
    name: str = "default"
    n_users: int = 20
    n_groups: int = 5
    groups_per_user: int = 2
    acl_mix: tuple[float, float, float] = (0.5, 0.3, 0.2)   # public, group, private
    share_rate: float = 0.5
    hot_pool_size: int = 64
    zipf_s: float = 1.0
    n_requests: int = 400
    docs_per_request: int = 10
    question_pool_size: int = 200
    seed: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        d["acl_mix"] = list(self.acl_mix)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "WorkloadConfig":
        d = dict(d)
        d["acl_mix"] = tuple(d["acl_mix"])
        return cls(**d)


@dataclass
class WorkloadRequest:
    req_id: str
    user: str
    qid: str
    question: str
    answer: str
    docs: list[dict]          # [{chunk_id, acl, score, rank, is_gold}]

    def retrieved_docs(self, corpus: dict[str, Chunk]) -> list[RetrievedDoc]:
        out = []
        for d in self.docs:
            c = corpus[d["chunk_id"]]
            out.append(RetrievedDoc(c.chunk_id, c.title, c.text, ACL.parse(d["acl"]), d["rank"], d["score"], d.get("is_gold", False)))
        return out


@dataclass
class Workload:
    config: WorkloadConfig
    directory: Directory
    chunk_acl: dict[str, str]
    requests: list[WorkloadRequest]
    stats: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "config": self.config.to_json(),
                "directory": self.directory.to_json(),
                "chunk_acl": self.chunk_acl,
                "stats": self.stats,
                "requests": [asdict(r) for r in self.requests],
            }, f)

    @classmethod
    def load(cls, path: Path) -> "Workload":
        d = json.load(open(path))
        return cls(
            config=WorkloadConfig.from_json(d["config"]),
            directory=Directory.from_json(d["directory"]),
            chunk_acl=d["chunk_acl"],
            requests=[WorkloadRequest(**r) for r in d["requests"]],
            stats=d.get("stats", {}),
        )


def make_directory(cfg: WorkloadConfig, rng: random.Random) -> Directory:
    users = tuple(f"u{i:03d}" for i in range(cfg.n_users))
    groups = tuple(f"g{i:02d}" for i in range(cfg.n_groups))
    k = min(cfg.groups_per_user, cfg.n_groups)
    membership = {u: frozenset(rng.sample(groups, k)) for u in users}
    return Directory(users, groups, membership)


def assign_acls(chunk_ids: Sequence[str], cfg: WorkloadConfig, directory: Directory, rng: random.Random) -> dict[str, str]:
    p_pub, p_grp, p_usr = cfg.acl_mix
    tot = p_pub + p_grp + p_usr
    p_pub, p_grp = p_pub / tot, p_grp / tot
    out: dict[str, str] = {}
    for cid in chunk_ids:
        r = rng.random()
        if r < p_pub:
            out[cid] = "public"
        elif r < p_pub + p_grp:
            out[cid] = f"group:{rng.choice(directory.groups)}"
        else:
            out[cid] = f"user:{rng.choice(directory.users)}"
    return out


def generate(cfg: WorkloadConfig, corpus: dict[str, Chunk], questions: list[Question], bm25: BM25Index) -> Workload:
    rng = random.Random(cfg.seed)
    nrng = np.random.default_rng(cfg.seed)
    directory = make_directory(cfg, rng)
    chunk_ids = sorted(corpus)
    chunk_acl = assign_acls(chunk_ids, cfg, directory, rng)
    acl_obj = {cid: ACL.parse(a) for cid, a in chunk_acl.items()}

    def readable(user: str, cid: str) -> bool:
        return acl_obj[cid].allows(user, directory)

    # hot pool with Zipf popularity
    hot = rng.sample(chunk_ids, min(cfg.hot_pool_size, len(chunk_ids)))
    hot_w = np.array([1.0 / (i + 1) ** cfg.zipf_s for i in range(len(hot))])
    hot_w /= hot_w.sum()

    # question pool: questions whose gold chunks are readable by at least one user
    qs = list(questions)
    rng.shuffle(qs)
    pool: list[tuple[Question, list[str]]] = []
    for q in qs:
        if any(g not in corpus for g in q.gold_chunk_ids):
            continue
        eligible = [u for u in directory.users if all(readable(u, g) for g in q.gold_chunk_ids)]
        if eligible:
            pool.append((q, eligible))
        if len(pool) >= cfg.question_pool_size:
            break
    if not pool:
        raise RuntimeError("no usable questions (all gold chunks unreadable?)")

    requests: list[WorkloadRequest] = []
    n_hot_total = 0
    n_fill_random = 0
    for i in range(cfg.n_requests):
        q, eligible = rng.choice(pool)
        user = rng.choice(eligible)
        golds = list(q.gold_chunk_ids)
        chosen: list[str] = list(golds)
        n_slots = max(0, cfg.docs_per_request - len(golds))
        n_hot = int(round(cfg.share_rate * n_slots))

        cand = [(c, w) for c, w in zip(hot, hot_w) if c not in chosen and readable(user, c)]
        if cand and n_hot > 0:
            ids = [c for c, _ in cand]
            w = np.array([w for _, w in cand])
            w /= w.sum()
            pick = nrng.choice(len(ids), size=min(n_hot, len(ids)), replace=False, p=w)
            chosen += [ids[j] for j in pick]
            n_hot_total += len(pick)

        distractors = [c["chunk_id"] for c in q.context if c["chunk_id"] not in chosen and readable(user, c["chunk_id"])]
        need = cfg.docs_per_request - len(chosen)
        chosen += distractors[:need]
        need = cfg.docs_per_request - len(chosen)
        if need > 0:  # not enough readable distractors: fill with random readable chunks
            extra = [c for c in rng.sample(chunk_ids, min(len(chunk_ids), 200)) if c not in chosen and readable(user, c)]
            chosen += extra[:need]
            n_fill_random += min(need, len(extra))

        scores = bm25.scores(q.question, chosen)
        ordered = sorted(chosen, key=lambda c: -scores[c])
        docs = [{"chunk_id": c, "acl": chunk_acl[c], "score": round(scores[c], 4), "rank": r, "is_gold": c in golds}
                for r, c in enumerate(ordered)]
        requests.append(WorkloadRequest(f"r{i:05d}", user, q.qid, q.question, q.answer, docs))

    n_docs = sum(len(r.docs) for r in requests)
    mix_counts = {"public": 0, "group": 0, "user": 0}
    for r in requests:
        for d in r.docs:
            mix_counts[d["acl"].split(":")[0]] += 1
    stats = {
        "n_requests": len(requests),
        "distinct_questions": len({r.qid for r in requests}),
        "mean_docs": n_docs / max(1, len(requests)),
        "hot_docs_frac": n_hot_total / max(1, n_docs),
        "random_fill_docs": n_fill_random,
        "evidence_acl_mix": {k: v / max(1, n_docs) for k, v in mix_counts.items()},
    }
    return Workload(cfg, directory, chunk_acl, requests, stats)
