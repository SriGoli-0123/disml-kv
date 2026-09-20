import random

import pytest

from aclkv.data_prep import BM25Index, Chunk, Question
from aclkv.workload import Workload, WorkloadConfig, generate

WORDS = [f"w{i}" for i in range(400)]


def make_corpus_and_questions(n_questions=60, rng_seed=1):
    rng = random.Random(rng_seed)
    chunks: dict[str, Chunk] = {}
    questions = []
    all_ids = []

    def new_chunk(prefix):
        cid = f"{prefix}{len(all_ids):04d}"
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(30, 60)))
        chunks[cid] = Chunk(cid, f"Title {cid}", text)
        all_ids.append(cid)
        return cid

    for q in range(n_questions):
        golds = [new_chunk("g"), new_chunk("g")]
        distract = [new_chunk("d") for _ in range(8)]
        qwords = chunks[golds[0]].text.split()[:4] + chunks[golds[1]].text.split()[:3]
        questions.append(Question(f"q{q:03d}", " ".join(qwords) + "?", "yes" if q % 2 else "no", "comparison", "easy",
                                  golds, [{"chunk_id": c} for c in golds + distract]))
    return chunks, questions


@pytest.fixture(scope="session")
def mini_data():
    chunks, questions = make_corpus_and_questions()
    return chunks, questions, BM25Index(list(chunks.values()))


def make_workload(mini_data, **kw) -> Workload:
    chunks, questions, bm25 = mini_data
    cfg = WorkloadConfig(name="t", n_users=8, n_groups=3, groups_per_user=1, n_requests=kw.pop("n_requests", 80),
                         question_pool_size=kw.pop("question_pool_size", 15), hot_pool_size=kw.pop("hot_pool_size", 20), **kw)
    return generate(cfg, chunks, questions, bm25)
