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
    # ---- v2 (scenario "topic" / "pinned"); defaults keep the v1 "injection" workload
    scenario: str = "injection"          # "injection" (v1) | "topic" | "pinned"
    dataset: str = "hotpotqa"
    n_divisions: int = 0                 # nested groups: division ⊃ department
    n_cross_groups: int = 0              # cross-functional groups (overlapping membership)
    cross_groups_per_user: int = 0
    group_acl_split: tuple[float, float, float] = (0.6, 0.25, 0.15)   # dept / division / cross
    n_topics: int = 40
    topic_affinity: float = 0.8          # P(user asks about own department's topics)
    question_zipf_s: float = 1.0         # popularity skew of questions within a topic
    query_noise: float = 0.0             # retrieval-query term dropout (paraphrase proxy)
    retrieval: str = "bm25"              # "bm25" | "dense" | "hybrid"
    n_pinned: int = 0                    # always-included public documents (scenario "pinned")

    def to_json(self) -> dict:
        d = asdict(self)
        d["acl_mix"] = list(self.acl_mix)
        d["group_acl_split"] = list(self.group_acl_split)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "WorkloadConfig":
        d = dict(d)
        d["acl_mix"] = tuple(d["acl_mix"])
        if "group_acl_split" in d:
            d["group_acl_split"] = tuple(d["group_acl_split"])
        return cls(**d)


@dataclass
class WorkloadRequest:
    req_id: str
    user: str
    qid: str
    question: str
    answer: str
    docs: list[dict]          # [{chunk_id, acl, score, rank, is_gold}]
    answers: list[str] = field(default_factory=list)   # answer + aliases (MuSiQue)
    topic: int = -1

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
    pinned_chunk_ids: list[str] = field(default_factory=list)

    def pinned_docs(self, corpus: dict[str, Chunk]) -> list[RetrievedDoc]:
        return [RetrievedDoc(c, corpus[c].title, corpus[c].text, ACL.parse(self.chunk_acl[c]), -1 - i)
                for i, c in enumerate(self.pinned_chunk_ids)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # store only the ACLs of chunks that appear in the workload (keeps files small)
        used = {d["chunk_id"] for r in self.requests for d in r.docs} | set(self.pinned_chunk_ids)
        with open(path, "w") as f:
            json.dump({
                "config": self.config.to_json(),
                "directory": self.directory.to_json(),
                "chunk_acl": {c: a for c, a in self.chunk_acl.items() if c in used},
                "stats": self.stats,
                "pinned_chunk_ids": self.pinned_chunk_ids,
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
            pinned_chunk_ids=d.get("pinned_chunk_ids", []),
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
        requests.append(WorkloadRequest(f"r{i:05d}", user, q.qid, q.question, q.answer, docs, q.answers))

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


# ===================================================================== v2
def make_org_directory(cfg: WorkloadConfig, rng: random.Random) -> tuple[Directory, dict[str, str], dict[str, str]]:
    """Departments (``d..``) nested in divisions (``v..``) plus cross-functional
    groups (``x..``).  Returns (directory, user->department, department->division)."""
    users = tuple(f"u{i:04d}" for i in range(cfg.n_users))
    depts = tuple(f"d{i:02d}" for i in range(cfg.n_groups))
    divs = tuple(f"v{i:02d}" for i in range(cfg.n_divisions))
    cross = tuple(f"x{i:02d}" for i in range(cfg.n_cross_groups))
    dept_div = {d: (divs[i % len(divs)] if divs else "") for i, d in enumerate(depts)}
    user_dept = {u: depts[i % len(depts)] for i, u in enumerate(users)}
    membership = {}
    for u in users:
        g = {user_dept[u]}
        if dept_div[user_dept[u]]:
            g.add(dept_div[user_dept[u]])
        if cross and cfg.cross_groups_per_user:
            g |= set(rng.sample(cross, min(cfg.cross_groups_per_user, len(cross))))
        membership[u] = frozenset(g)
    return Directory(users, depts + divs + cross, membership), user_dept, dept_div


def cluster_topics(questions: list[Question], n_topics: int, seed: int) -> list[int]:
    """Balanced topical clusters of questions (deterministic).

    TF-IDF -> LSA (truncated SVD) -> k-means centroids, then a greedy
    capacity-constrained assignment so every topic holds ceil(N / T)
    questions.  Plain k-means on short questions is very unbalanced (one giant
    cluster plus singletons), which would make users whose department owns a
    singleton topic repeat the same question and inflate cache reuse.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [q.question for q in questions]
    x = TfidfVectorizer(stop_words="english", min_df=2, max_features=50000, sublinear_tf=True).fit_transform(texts)
    dims = max(2, min(100, x.shape[1] - 1, len(texts) - 1))
    z = normalize(TruncatedSVD(n_components=dims, random_state=seed).fit_transform(x))
    km = KMeans(n_clusters=n_topics, random_state=seed, n_init=4).fit(z)
    dist = ((z[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)       # N x T
    cap = -(-len(texts) // n_topics)
    load = [0] * n_topics
    out = [-1] * len(texts)
    for flat in np.argsort(dist, axis=None, kind="stable"):
        i, t = divmod(int(flat), n_topics)
        if out[i] == -1 and load[t] < cap:
            out[i] = t
            load[t] += 1
    return out


def generate_topic(cfg: WorkloadConfig, corpus: dict[str, Chunk], questions: list[Question], retriever) -> Workload:
    """Realistic multi-tenant RAG workload (scenarios "topic" and "pinned").

    * Questions are clustered into topics; each topic is owned by a department.
    * A chunk's ACL follows the topic that owns it: public with p_pub, else
      restricted to the owning department / its division / a cross-functional
      group, or private to one member of the owning department.
    * Users mostly ask about their own department's topics (``topic_affinity``);
      within a topic, questions follow a Zipf popularity (FAQs repeat).
    * Evidence is retrieved top-k from the **whole corpus**, ACL-filtered for
      the requester — cross-request overlap is produced by the retriever, not
      injected.  ``query_noise`` perturbs only the retrieval query.
    """
    from .retrieval import query_dropout

    rng = random.Random(cfg.seed)
    directory, user_dept, dept_div = make_org_directory(cfg, rng)
    depts = [g for g in directory.groups if g.startswith("d")]
    cross = [g for g in directory.groups if g.startswith("x")]
    members = {d: [u for u in directory.users if user_dept[u] == d] for d in depts}

    qs = [q for q in questions if all(g in corpus for g in q.gold_chunk_ids)]
    rng.shuffle(qs)
    qs = qs[: cfg.question_pool_size] if cfg.question_pool_size else qs
    topics = cluster_topics(qs, cfg.n_topics, cfg.seed)
    topic_owner = {t: depts[t % len(depts)] for t in range(cfg.n_topics)}
    dept_topics = {d: [t for t in range(cfg.n_topics) if topic_owner[t] == d] for d in depts}

    # chunk -> owning topic: topic of the first pool question whose context contains it
    chunk_topic: dict[str, int] = {}
    for q, t in zip(qs, topics):
        for c in q.context:
            chunk_topic.setdefault(c["chunk_id"], t)
    p_pub, p_grp, p_usr = cfg.acl_mix
    tot = p_pub + p_grp + p_usr
    p_pub, p_grp = p_pub / tot, p_grp / tot
    s_dept, s_div, s_cross = cfg.group_acl_split
    if not cfg.n_divisions:
        s_div = 0.0
    if not cross:
        s_cross = 0.0
    s_tot = s_dept + s_div + s_cross
    chunk_acl: dict[str, str] = {}
    for cid in sorted(corpus):
        t = chunk_topic.get(cid, rng.randrange(cfg.n_topics))
        owner = topic_owner[t]
        r = rng.random()
        if r < p_pub:
            chunk_acl[cid] = "public"
        elif r < p_pub + p_grp:
            g = rng.random() * s_tot
            if g < s_dept:
                chunk_acl[cid] = f"group:{owner}"
            elif g < s_dept + s_div:
                chunk_acl[cid] = f"group:{dept_div[owner]}"
            else:
                chunk_acl[cid] = f"group:{rng.choice(cross)}"
        else:
            chunk_acl[cid] = f"user:{rng.choice(members[owner])}"

    pinned: list[str] = []
    if cfg.scenario == "pinned" and cfg.n_pinned:
        gold_all = {g for q in qs for g in q.gold_chunk_ids}
        pool = [c for c in sorted(corpus) if c not in gold_all]
        pinned = rng.sample(pool, cfg.n_pinned)
        for c in pinned:
            chunk_acl[c] = "public"

    acl_obj = {cid: ACL.parse(a) for cid, a in chunk_acl.items()}
    by_topic: dict[int, list[Question]] = {}
    for q, t in zip(qs, topics):
        by_topic.setdefault(t, []).append(q)
    topic_w = {t: np.array([1.0 / (i + 1) ** cfg.question_zipf_s for i in range(len(v))]) for t, v in by_topic.items()}
    for t in topic_w:
        topic_w[t] /= topic_w[t].sum()
    nrng = np.random.default_rng(cfg.seed)

    def readable(user: str, cid: str) -> bool:
        return acl_obj[cid].allows(user, directory)

    requests: list[WorkloadRequest] = []
    n_resampled = n_unreadable_gold = 0
    gold_recall = []
    for i in range(cfg.n_requests):
        user = rng.choice(directory.users)
        own = [t for t in dept_topics[user_dept[user]] if t in by_topic]
        for attempt in range(25):
            if own and rng.random() < cfg.topic_affinity:
                t = rng.choice(own)
            else:
                t = rng.choice(sorted(by_topic))
            q = by_topic[t][int(nrng.choice(len(by_topic[t]), p=topic_w[t]))]
            if all(readable(user, g) for g in q.gold_chunk_ids):
                break
            n_resampled += 1
        else:
            n_unreadable_gold += 1
        query = query_dropout(q.question, cfg.query_noise, rng)
        hits = retriever.search(query, cfg.docs_per_request, allowed=lambda c: readable(user, c) and c not in pinned)
        golds = set(q.gold_chunk_ids)
        docs = [{"chunk_id": c, "acl": chunk_acl[c], "score": round(sc, 4), "rank": r, "is_gold": c in golds}
                for r, (c, sc) in enumerate(hits)]
        gold_recall.append(sum(d["is_gold"] for d in docs) / max(1, len(golds)))
        requests.append(WorkloadRequest(f"r{i:05d}", user, q.qid, q.question, q.answer, docs, q.answers, t))

    stats = workload_stats(requests)
    stats.update({
        "scenario": cfg.scenario, "gold_recall_at_k": float(np.mean(gold_recall)),
        "resampled_for_gold_access": n_resampled, "requests_with_unreadable_gold": n_unreadable_gold,
        "n_pinned": len(pinned), "pool_questions": len(qs),
    })
    return Workload(cfg, directory, chunk_acl, requests, stats, pinned)


def workload_stats(requests: list[WorkloadRequest]) -> dict:
    """Overlap diagnostics that do not depend on any caching policy."""
    n_docs = sum(len(r.docs) for r in requests)
    mix = {"public": 0, "group": 0, "user": 0}
    for r in requests:
        for d in r.docs:
            mix[d["acl"].split(":")[0]] += 1
    seen_first, seen_set, seen_any = set(), set(), set()
    first_seen = set_seen = 0
    doc_seen_frac = []
    for r in requests:
        ids = [d["chunk_id"] for d in r.docs]
        if ids and ids[0] in seen_first:
            first_seen += 1
        key = frozenset(ids)
        if key in seen_set:
            set_seen += 1
        doc_seen_frac.append(sum(1 for c in ids if c in seen_any) / max(1, len(ids)))
        if ids:
            seen_first.add(ids[0])
        seen_set.add(key)
        seen_any.update(ids)
    return {
        "n_requests": len(requests),
        "distinct_questions": len({r.qid for r in requests}),
        "mean_docs": n_docs / max(1, len(requests)),
        "evidence_acl_mix": {k: v / max(1, n_docs) for k, v in mix.items()},
        "first_doc_seen_before": first_seen / max(1, len(requests)),
        "exact_doc_set_seen_before": set_seen / max(1, len(requests)),
        "docs_seen_before": float(np.mean(doc_seen_frac)) if doc_seen_frac else 0.0,
    }


def generate_any(cfg: WorkloadConfig, corpus, questions, bm25=None, retriever=None) -> Workload:
    if cfg.scenario == "injection":
        w = generate(cfg, corpus, questions, bm25)
        w.stats.update(workload_stats(w.requests))
        return w
    if retriever is None:
        raise ValueError("scenario %r needs a corpus-level retriever" % cfg.scenario)
    return generate_topic(cfg, corpus, questions, retriever)
