"""KV-cache geometry: bytes per token per model, blocks for a byte budget.

vLLM sizes the paged KV cache as::

    bytes/token = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes
    num_blocks  = floor(budget_bytes / (bytes/token * block_size))

The table below is used when the HF config cannot be fetched; values are for
bf16/fp16 weights (2 bytes) and *full-attention* layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

_KNOWN: dict[str, tuple[int, int, int]] = {
    # model: (num_layers, num_kv_heads, head_dim)
    "Qwen/Qwen2.5-7B-Instruct": (28, 4, 128),
    "Qwen/Qwen2.5-3B-Instruct": (36, 2, 128),
    "Qwen/Qwen2.5-1.5B-Instruct": (28, 2, 128),
    "Qwen/Qwen2.5-0.5B-Instruct": (24, 2, 64),
    "Qwen/Qwen3-8B": (36, 8, 128),
    "Qwen/Qwen3-4B": (36, 8, 128),
    "Qwen/Qwen3-1.7B": (28, 8, 128),
    "meta-llama/Llama-3.1-8B-Instruct": (32, 8, 128),
    "meta-llama/Llama-3.2-3B-Instruct": (28, 8, 128),
}


@dataclass(frozen=True)
class KVGeometry:
    model: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def bytes_per_block(self, block_size: int = 16) -> int:
        return self.bytes_per_token * block_size

    def num_blocks(self, budget_bytes: int, block_size: int = 16) -> int:
        return budget_bytes // self.bytes_per_block(block_size)

    def to_json(self) -> dict:
        return {"model": self.model, "num_layers": self.num_layers, "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim, "dtype_bytes": self.dtype_bytes, "bytes_per_token": self.bytes_per_token}


def _from_hf_config(model: str, dtype_bytes: int) -> KVGeometry | None:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "config.json")
        cfg = json.load(open(path))
        if "text_config" in cfg:
            cfg = cfg["text_config"]
        layers = int(cfg["num_hidden_layers"])
        kv = int(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
        head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
        return KVGeometry(model, layers, kv, head_dim, dtype_bytes)
    except Exception:
        return None


def kv_geometry(model: str, dtype_bytes: int = 2, allow_download: bool = True) -> KVGeometry:
    if model in _KNOWN:
        l, kv, hd = _KNOWN[model]
        return KVGeometry(model, l, kv, hd, dtype_bytes)
    if allow_download:
        g = _from_hf_config(model, dtype_bytes)
        if g is not None:
            return g
    raise KeyError(f"unknown model {model!r}; add it to aclkv.model_info._KNOWN")


GIB = 1024 ** 3
