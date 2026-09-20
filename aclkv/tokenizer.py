"""Minimal tokenizer abstraction so the middleware/simulator can run without
``transformers`` (and tests can run with a fake tokenizer)."""

from __future__ import annotations

import re
from typing import Protocol, Sequence


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class HFTokenizer:
    """Wraps ``tokenizers.Tokenizer`` (fast, no torch).  Special tokens such as
    ``<|im_start|>`` inside the text are recognised by the fast tokenizer."""

    def __init__(self, name_or_path: str):
        from tokenizers import Tokenizer  # local import: optional dependency

        self.name = name_or_path
        self._tok = Tokenizer.from_pretrained(name_or_path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)


class FakeTokenizer:
    """Deterministic word-level tokenizer for tests/offline runs.

    Every whitespace-delimited word (and each newline) is one token; token ids
    are stable hashes of the word.  Good enough to reason about blocks.
    """

    _re = re.compile(r"\n|[^\s]+")

    def __init__(self):
        self.name = "fake"
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _id(self, w: str) -> int:
        if w not in self._vocab:
            i = len(self._vocab) + 1
            self._vocab[w] = i
            self._inv[i] = w
        return self._vocab[w]

    def encode(self, text: str) -> list[int]:
        return [self._id(w) for w in self._re.findall(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._inv.get(i, "?") for i in ids)


def load_tokenizer(name: str | None) -> TokenizerLike:
    if not name or name == "fake":
        return FakeTokenizer()
    return HFTokenizer(name)
