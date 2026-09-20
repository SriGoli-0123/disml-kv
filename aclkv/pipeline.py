"""Glue shared by the simulator sweep and the live benchmark: turn a workload
into scoped prompts for a given policy, deterministically."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .context import BuiltPrompt, ContextBuilder
from .data_prep import Chunk, load_corpus
from .ordering import OrderingConfig, PopularityTracker
from .policies import Policy, get_policy
from .scope import load_scope_key
from .tokenizer import TokenizerLike, load_tokenizer
from .workload import Workload


@dataclass
class BuildSettings:
    tokenizer: str = "Qwen/Qwen2.5-7B-Instruct"
    chat_format: str = "qwen2.5"
    block_size: int = 16
    align_blocks: bool = True
    scope_mode: str = "hash"
    max_shift: int | None = None
    min_pop: float = 1.0
    popularity_decay: float = 0.98

    def to_json(self) -> dict:
        return self.__dict__.copy()


def make_builder(workload: Workload, settings: BuildSettings, tok: TokenizerLike | None = None) -> ContextBuilder:
    tok = tok or load_tokenizer(settings.tokenizer)
    return ContextBuilder(
        tokenizer=tok,
        directory=workload.directory,
        scope_key=load_scope_key(),
        block_size=settings.block_size,
        align_blocks=settings.align_blocks,
        chat_format=settings.chat_format,
        scope_mode=settings.scope_mode,
    )


def build_prompts(workload: Workload, corpus: dict[str, Chunk], policy: Policy | str,
                  builder: ContextBuilder, settings: BuildSettings) -> list[BuiltPrompt]:
    pol = get_policy(policy) if isinstance(policy, str) else policy
    ordering = OrderingConfig(name=pol.ordering, max_shift=settings.max_shift, min_pop=settings.min_pop,
                              popularity=PopularityTracker(settings.popularity_decay))
    out: list[BuiltPrompt] = []
    for r in workload.requests:
        docs = r.retrieved_docs(corpus)
        p = builder.build(r.req_id, r.user, r.question, docs, pol, ordering=ordering)
        p.meta.update({"qid": r.qid, "answer": r.answer, "question": r.question})
        out.append(p)
    return out


def load_workload_and_corpus(workload_path: str | Path, corpus_path: str | Path) -> tuple[Workload, dict[str, Chunk]]:
    return Workload.load(Path(workload_path)), load_corpus(Path(corpus_path))
