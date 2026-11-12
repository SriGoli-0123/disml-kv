"""Context builder (middleware step 3): render the RAG prompt as *segments*,
tokenize each segment independently, optionally pad segments to the KV block
boundary, and attach scope barriers.

Why token ids and not text?  vLLM's completions endpoint accepts a list of
token ids as the prompt, so the engine sees exactly the ids we hashed on our
side; barrier offsets are therefore exact, with no re-tokenisation drift.

Why block alignment?  A block that straddles two documents hashes tokens of
both, so it can only be reused by a request that has the same *pair* in the
same order.  Padding every segment to a multiple of ``block_size`` makes each
document's blocks independent of what follows (proposal risk mitigation:
"begin with block-aligned scope boundaries").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .acl import ACL, PUBLIC, Directory, authorization_filter
from .ordering import OrderingConfig, RetrievedDoc, apply_ordering
from .policies import Policy, SegmentScope, plan_scopes
from .privacy import PromotionTracker
from .scope import Barrier, encode_cache_salt
from .tokenizer import TokenizerLike

DEFAULT_SYSTEM = (
    "You are a precise question-answering assistant for an enterprise knowledge base. "
    "Use only the provided documents."
)
DEFAULT_INSTRUCTIONS = (
    "Answer the question using the documents below. Reply with the shortest possible "
    "answer (a name, date, number, or yes/no) and nothing else."
)


@dataclass(frozen=True)
class ChatFormat:
    name: str
    preamble: str      # format(system=..., instructions=...)
    doc: str           # format(title=..., text=...)
    tail: str          # format(question=...)
    pad_text: str = "\n"


CHAT_FORMATS: dict[str, ChatFormat] = {
    # Qwen2.5 / ChatML
    "qwen2.5": ChatFormat(
        "qwen2.5",
        preamble="<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instructions}\n\nDocuments:\n\n",
        doc="### {title}\n{text}\n\n",
        tail="Question: {question}<|im_end|>\n<|im_start|>assistant\n",
    ),
    # Qwen3 non-thinking mode: the template inserts an empty think block.
    "qwen3": ChatFormat(
        "qwen3",
        preamble="<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instructions}\n\nDocuments:\n\n",
        doc="### {title}\n{text}\n\n",
        tail="Question: {question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    "llama3": ChatFormat(
        "llama3",
        preamble="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                 "<|start_header_id|>user<|end_header_id|>\n\n{instructions}\n\nDocuments:\n\n",
        doc="### {title}\n{text}\n\n",
        tail="Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    # plain text (for the fake tokenizer / non-chat models)
    "plain": ChatFormat(
        "plain",
        preamble="{system}\n{instructions}\n\nDocuments:\n\n",
        doc="### {title}\n{text}\n\n",
        tail="Question: {question}\nAnswer:",
    ),
}


@dataclass(frozen=True)
class Segment:
    name: str            # "preamble" | "pinned" | "doc" | "marker" | "tail"
    start: int
    end: int             # exclusive, includes padding
    n_pad: int
    scope_id: str | None
    acl: ACL
    chunk_id: str | None = None
    shared: bool = True          # scope shared across users (False: per-user / nonce)
    prefix_key: str = ""         # digest of the evidence prefix ending at this segment
    own_acl: ACL | None = None   # the document's own ACL (position-independent caching)


@dataclass
class BuiltPrompt:
    req_id: str
    user: str
    token_ids: list[int]
    segments: list[Segment]
    barriers: list[Barrier]
    cache_salt: str | None
    ordered_chunk_ids: list[str]
    n_dropped_by_filter: int = 0
    policy: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def acl_at(self, token_idx: int) -> ACL:
        """Effective ACL of the prefix ending at ``token_idx`` (inclusive)."""
        return self.segment_at(token_idx).acl if self.segments else PUBLIC

    def segment_at(self, token_idx: int) -> "Segment":
        for s in self.segments:
            if s.start <= token_idx < s.end:
                return s
        return self.segments[-1]

    def to_json(self) -> dict:
        return {
            "req_id": self.req_id,
            "user": self.user,
            "policy": self.policy,
            "num_tokens": self.num_tokens,
            "cache_salt": self.cache_salt,
            "ordered_chunk_ids": self.ordered_chunk_ids,
            "n_dropped_by_filter": self.n_dropped_by_filter,
            "segments": [
                {"name": s.name, "start": s.start, "end": s.end, "n_pad": s.n_pad,
                 "scope_id": s.scope_id, "acl": s.acl.canonical(), "chunk_id": s.chunk_id}
                for s in self.segments
            ],
            "barriers": [{"offset": b.token_offset, "scope_id": b.scope_id, "acl": b.acl_canonical} for b in self.barriers],
            **self.meta,
        }


class ContextBuilder:
    """Turns (user, query, authorized docs) into a scoped token-id prompt."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        directory: Directory,
        scope_key: bytes,
        block_size: int = 16,
        align_blocks: bool = True,
        chat_format: str = "qwen2.5",
        scope_mode: str = "hash",           # "hash" (vLLM plugin) | "marker" (stock vLLM)
        system: str = DEFAULT_SYSTEM,
        instructions: str = DEFAULT_INSTRUCTIONS,
        ordering: OrderingConfig | None = None,
    ):
        self.tok = tokenizer
        self.directory = directory
        self.key = scope_key
        self.block_size = block_size
        self.align = align_blocks
        self.fmt = CHAT_FORMATS[chat_format]
        if scope_mode not in ("hash", "marker"):
            raise ValueError("scope_mode must be 'hash' or 'marker'")
        self.scope_mode = scope_mode
        self.system = system
        self.instructions = instructions
        self.ordering = ordering or OrderingConfig()
        self._pad_ids = self.tok.encode(self.fmt.pad_text)
        if len(self._pad_ids) != 1:
            # fall back to the first id; only matters for exact alignment
            self._pad_ids = self._pad_ids[:1]
        self._cache: dict[str, list[int]] = {}
        self._aud_cache: dict[str, int] = {}

    # ------------------------------------------------------------ helpers
    def _encode_cached(self, text: str) -> list[int]:
        ids = self._cache.get(text)
        if ids is None:
            ids = self.tok.encode(text)
            self._cache[text] = ids
        return ids

    def _pad(self, ids: list[int]) -> tuple[list[int], int]:
        if not self.align:
            return ids, 0
        rem = len(ids) % self.block_size
        if rem == 0:
            return ids, 0
        n = self.block_size - rem
        return ids + self._pad_ids * n, n

    # -------------------------------------------------------------- build
    def _audience_size(self, acl: ACL) -> int:
        key = acl.canonical()
        n = self._aud_cache.get(key)
        if n is None:
            n = len(acl.audience(self.directory))
            self._aud_cache[key] = n
        return n

    def build(
        self,
        req_id: str,
        user: str,
        question: str,
        docs: Sequence[RetrievedDoc],
        policy: Policy,
        ordering: OrderingConfig | None = None,
        pinned: Sequence[RetrievedDoc] = (),
        observe: bool = True,
    ) -> BuiltPrompt:
        """Build one scoped prompt.  ``pinned`` documents (e.g. an always-included
        handbook) go right after the preamble in every ordering.  ``observe=False``
        builds without updating the middleware state (used for what-if analyses)."""
        ordering = ordering or self.ordering
        # 1. authorization filter (defensive; the workload already enforces it)
        kept, dropped = authorization_filter(docs, user, self.directory)
        pinned_kept, pinned_dropped = authorization_filter(pinned, user, self.directory)
        # 3. context builder: ordering
        anchor_k = policy.anchor_k or ordering.anchor_k
        if policy.kanon > 1 and ordering.promotion is None:
            ordering.promotion = PromotionTracker(policy.kanon)
        ordered = apply_ordering(policy.ordering, kept, popularity=ordering.popularity,
                                 min_pop=ordering.min_pop, max_shift=ordering.max_shift,
                                 anchor_k=anchor_k, tree=ordering.tree,
                                 audience_size=self._audience_size, kanon=policy.kanon)
        full = list(pinned_kept) + ordered
        # 2. prefix ACL propagation + 4. scope plan (+ private tail, k-anonymous promotion)
        scopes: list[SegmentScope] = plan_scopes(
            policy.caching, user, full, self.key, tail_scope=policy.tail_scope, kanon=policy.kanon,
            promotion=ordering.promotion if observe else _ReadOnlyPromotion(ordering.promotion),
            audience_size=self._audience_size)
        if observe:
            ordering.popularity.observe([d.chunk_id for d in ordered])
            ordering.tree.observe([d.chunk_id for d in ordered], user)

        texts: list[tuple[str, str, str | None]] = []   # (name, text, chunk_id)
        texts.append(("preamble", self.fmt.preamble.format(system=self.system, instructions=self.instructions), None))
        own = {d.chunk_id: d.acl for d in full}
        for d in pinned_kept:
            texts.append(("pinned", self.fmt.doc.format(title=d.title, text=d.text), d.chunk_id))
        for d in ordered:
            texts.append(("doc", self.fmt.doc.format(title=d.title, text=d.text), d.chunk_id))
        texts.append(("tail", self.fmt.tail.format(question=question), None))
        assert len(texts) == len(scopes)

        token_ids: list[int] = []
        segments: list[Segment] = []
        barriers: list[Barrier] = []
        prev_scope: str | None = None
        for i, ((name, text, chunk_id), sc) in enumerate(zip(texts, scopes)):
            new_scope = sc.scope_id is not None and sc.scope_id != prev_scope
            if new_scope and self.scope_mode == "marker":
                # in-prompt sentinel: works with an unmodified vLLM
                mids, npad = self._pad(self._encode_cached(f"<scope {sc.scope_id}>\n"))
                segments.append(Segment("marker", len(token_ids), len(token_ids) + len(mids), npad, sc.scope_id,
                                        sc.acl, None, sc.shared, sc.prefix_key))
                token_ids.extend(mids)
            start = len(token_ids)
            is_last = i == len(texts) - 1
            ids = self._encode_cached(text)
            npad = 0
            if not is_last:
                ids, npad = self._pad(ids)
            if new_scope and self.scope_mode == "hash":
                barriers.append(Barrier(start, sc.scope_id, sc.acl.canonical()))
            token_ids.extend(ids)
            segments.append(Segment(name, start, len(token_ids), npad, sc.scope_id, sc.acl, chunk_id,
                                    sc.shared, sc.prefix_key, own.get(chunk_id)))
            if sc.scope_id is not None:
                prev_scope = sc.scope_id

        cache_salt = encode_cache_salt(barriers) if (self.scope_mode == "hash" and barriers) else None
        n_shared_docs = sum(1 for seg in segments if seg.name in ("doc", "pinned") and seg.shared)
        return BuiltPrompt(
            req_id=req_id,
            user=user,
            token_ids=token_ids,
            segments=segments,
            barriers=barriers,
            cache_salt=cache_salt,
            ordered_chunk_ids=[d.chunk_id for d in ordered],
            n_dropped_by_filter=len(dropped) + len(pinned_dropped),
            policy=policy.name,
            meta={"ordering": policy.ordering, "caching": policy.caching, "n_docs": len(ordered),
                  "anchor_k": anchor_k, "kanon": policy.kanon, "tail_scope": policy.tail_scope,
                  "n_pinned": len(pinned_kept), "n_shared_docs": n_shared_docs},
        )

    def render_text(self, prompt: BuiltPrompt) -> str:
        return self.tok.decode(prompt.token_ids)


class _ReadOnlyPromotion:
    """Wraps a PromotionTracker so ``plan_scopes`` can query it without recording."""

    def __init__(self, inner):
        self._inner = inner

    def would_share(self, key, user):
        return True if self._inner is None else self._inner.would_share(key, user)

    def observe(self, key, user):
        return None
