"""Glue shared by the simulator sweep and the live benchmark: turn a workload
into scoped prompts for a given policy, deterministically."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import CatalogConfig, EvidenceCatalog
from .context import BuiltPrompt, ContextBuilder
from .data_prep import Chunk, load_corpus
from .ordering import OrderingConfig, PopularityTracker, PrefixTree
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
    anchor_k: int = 0
    tree_window: int = 512
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


def make_ordering_state(pol: Policy, settings: BuildSettings, pinned=(), catalog: CatalogConfig | None = None,
                        num_blocks: int | None = None, output_tokens: int = 48) -> OrderingConfig:
    """Fresh middleware state for one run of ``pol`` (popularity, prefix tree,
    and the evidence catalog for the ``catalog`` ordering)."""
    ordering = OrderingConfig(name=pol.ordering, max_shift=settings.max_shift, min_pop=settings.min_pop,
                              anchor_k=settings.anchor_k,
                              popularity=PopularityTracker(settings.popularity_decay),
                              tree=PrefixTree(settings.tree_window))
    if pol.ordering == "catalog":
        cat = EvidenceCatalog(catalog or CatalogConfig(), k=pol.kanon, num_blocks=num_blocks,
                              block_size=settings.block_size, output_tokens=output_tokens)
        cat.set_base_docs([d.chunk_id for d in pinned])
        ordering.catalog = cat
        ordering.promotion = cat
    return ordering


def build_prompts(workload: Workload, corpus: dict[str, Chunk], policy: Policy | str,
                  builder: ContextBuilder, settings: BuildSettings, catalog: CatalogConfig | None = None,
                  num_blocks: int | None = None, output_tokens: int = 48) -> list[BuiltPrompt]:
    """``catalog``/``num_blocks``/``output_tokens`` only matter for the
    ``catalog`` ordering, whose publisher works within a KV block budget."""
    pol = get_policy(policy) if isinstance(policy, str) else policy
    pinned = workload.pinned_docs(corpus)
    ordering = make_ordering_state(pol, settings, pinned, catalog, num_blocks, output_tokens)
    out: list[BuiltPrompt] = []
    for r in workload.requests:
        docs = r.retrieved_docs(corpus)
        event = (ordering.catalog.maybe_publish(builder, pol, ordering, pinned)
                 if ordering.catalog is not None else None)
        p = builder.build(r.req_id, r.user, r.question, docs, pol, ordering=ordering, pinned=pinned)
        if event is not None:
            p.meta["publish"] = event
        p.meta.update({"qid": r.qid, "answer": r.answer, "answers": r.answers or [r.answer], "question": r.question,
                       "gold": [d["chunk_id"] for d in r.docs if d.get("is_gold")], "topic": r.topic})
        out.append(p)
    return out


def load_workload_and_corpus(workload_path: str | Path, corpus_path: str | Path) -> tuple[Workload, dict[str, Chunk]]:
    return Workload.load(Path(workload_path)), load_corpus(Path(corpus_path))
