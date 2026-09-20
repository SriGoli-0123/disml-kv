"""HotpotQA (distractor) → corpus of chunks + questions with retrieval ranks.

Each HotpotQA question ships with 10 paragraphs (2 gold + 8 distractors), so
"retrieval" is already done for us; we only need a *relevance order*, which
we compute with BM25 over the whole corpus (``rank_bm25``).  Chunks are
deduplicated by content hash so the same Wikipedia paragraph appearing in
several questions is one chunk with one ACL.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

HF_PARQUET_URL = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet"
_TOKEN_RE = re.compile(r"\w+")


def simple_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    n_tokens: int = 0

    def to_json(self) -> dict:
        return {"chunk_id": self.chunk_id, "title": self.title, "text": self.text, "n_tokens": self.n_tokens}


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    qtype: str
    level: str
    gold_chunk_ids: list[str]
    context: list[dict] = field(default_factory=list)   # [{chunk_id, score, rank}] by relevance

    def to_json(self) -> dict:
        return {"qid": self.qid, "question": self.question, "answer": self.answer, "type": self.qtype,
                "level": self.level, "gold_chunk_ids": self.gold_chunk_ids, "context": self.context}


def chunk_id_for(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:12]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.ids = [c.chunk_id for c in chunks]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self._bm25 = BM25Okapi([simple_tokenize(c.title + " " + c.text) for c in chunks])

    def scores(self, question: str, chunk_ids: Iterable[str]) -> dict[str, float]:
        q = simple_tokenize(question)
        idx = [self.pos[c] for c in chunk_ids]
        s = self._bm25.get_batch_scores(q, idx)
        return {cid: float(v) for cid, v in zip(chunk_ids, s)}


def download_hotpot(dest: Path, url: str = HF_PARQUET_URL) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def read_hotpot_parquet(path: Path, max_questions: int | None = None, seed: int = 0) -> list[dict]:
    import pyarrow.parquet as pq
    import random

    rows = pq.read_table(path).to_pylist()
    if max_questions and max_questions < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:max_questions]
    return rows


def build(rows: list[dict], tokenizer=None) -> tuple[list[Chunk], list[Question]]:
    chunks: dict[str, Chunk] = {}
    questions: list[Question] = []
    for r in rows:
        titles = r["context"]["title"]
        sents = r["context"]["sentences"]
        gold_titles = set(r["supporting_facts"]["title"])
        ctx_ids: list[str] = []
        golds: list[str] = []
        for t, ss in zip(titles, sents):
            text = "".join(ss).strip()
            if not text:
                continue
            cid = chunk_id_for(t, text)
            if cid not in chunks:
                chunks[cid] = Chunk(cid, t, text, len(tokenizer.encode(text)) if tokenizer else 0)
            ctx_ids.append(cid)
            if t in gold_titles:
                golds.append(cid)
        if len(golds) < 2 or len(ctx_ids) < 4:
            continue
        questions.append(Question(r["id"], r["question"].strip(), r["answer"].strip(), r.get("type", ""),
                                  r.get("level", ""), golds, [{"chunk_id": c} for c in ctx_ids]))
    chunk_list = list(chunks.values())
    index = BM25Index(chunk_list)
    for q in questions:
        sc = index.scores(q.question, [c["chunk_id"] for c in q.context])
        ordered = sorted(sc.items(), key=lambda kv: -kv[1])
        q.context = [{"chunk_id": cid, "score": round(s, 4), "rank": i} for i, (cid, s) in enumerate(ordered)]
    return chunk_list, questions


def save(chunks: list[Chunk], questions: list[Question], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    with open(out_dir / "questions.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> dict[str, Chunk]:
    out: dict[str, Chunk] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["chunk_id"]] = Chunk(d["chunk_id"], d["title"], d["text"], d.get("n_tokens", 0))
    return out


def load_questions(path: Path) -> list[Question]:
    out: list[Question] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(Question(d["qid"], d["question"], d["answer"], d.get("type", ""), d.get("level", ""),
                                d["gold_chunk_ids"], d["context"]))
    return out
