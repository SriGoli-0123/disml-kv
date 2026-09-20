"""Async client for a vLLM OpenAI-compatible server.

Sends the *token ids* produced by the context builder to ``/v1/completions``
with streaming so that time-to-first-token (TTFT) can be measured on the
client side; the final usage chunk carries ``prompt_tokens_details.cached_tokens``
(requires ``--enable-prompt-tokens-details`` on the server), which is the
deterministic cache-hit measurement used for security checks.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass

import httpx


@dataclass
class CompletionResult:
    req_id: str
    ok: bool
    error: str | None
    t_send: float
    ttft_s: float | None          # first SSE chunk
    ttft_text_s: float | None     # first chunk with non-empty text
    e2e_s: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    text: str
    finish_reason: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000", model: str | None = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------ admin
    async def wait_ready(self, session: httpx.AsyncClient, timeout_s: float = 1800) -> None:
        t0 = time.time()
        while True:
            try:
                r = await session.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - t0 > timeout_s:
                raise TimeoutError("vLLM server not ready")
            await asyncio.sleep(2)

    async def resolve_model(self, session: httpx.AsyncClient) -> str:
        if self.model:
            return self.model
        r = await session.get(f"{self.base_url}/v1/models", timeout=30)
        r.raise_for_status()
        self.model = r.json()["data"][0]["id"]
        return self.model

    async def reset_prefix_cache(self, session: httpx.AsyncClient) -> bool:
        """Needs ``VLLM_SERVER_DEV_MODE=1`` on the server."""
        try:
            r = await session.post(f"{self.base_url}/reset_prefix_cache", timeout=60)
            return r.status_code == 200
        except Exception:
            return False

    async def metrics(self, session: httpx.AsyncClient) -> dict:
        from .metrics import parse_prometheus

        try:
            r = await session.get(f"{self.base_url}/metrics", timeout=30)
            if r.status_code != 200:
                return {"counters": {}, "cache_config": {}}
            return parse_prometheus(r.text)
        except Exception:
            return {"counters": {}, "cache_config": {}}

    # ------------------------------------------------------- inference
    async def complete(
        self,
        session: httpx.AsyncClient,
        req_id: str,
        token_ids: list[int],
        cache_salt: str | None,
        max_tokens: int = 48,
        temperature: float = 0.0,
        seed: int | None = 0,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        payload: dict = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            payload["seed"] = seed
        if cache_salt:
            payload["cache_salt"] = cache_salt
        if stop:
            payload["stop"] = stop

        t0 = time.perf_counter()
        t_send = time.time()
        t_first = t_first_text = None
        text_parts: list[str] = []
        usage: dict | None = None
        finish = None
        try:
            async with session.stream("POST", f"{self.base_url}/v1/completions", json=payload, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    return CompletionResult(req_id, False, f"HTTP {resp.status_code}: {body}", t_send,
                                            None, None, None, None, None, None, "")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    now = time.perf_counter()
                    obj = json.loads(data)
                    if t_first is None:
                        t_first = now - t0
                    for ch in obj.get("choices", []) or []:
                        txt = ch.get("text") or ""
                        if txt:
                            if t_first_text is None:
                                t_first_text = now - t0
                            text_parts.append(txt)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                    if obj.get("usage"):
                        usage = obj["usage"]
        except Exception as e:  # network / decode errors
            return CompletionResult(req_id, False, f"{type(e).__name__}: {e}", t_send, None, None, None, None, None, None, "")
        e2e = time.perf_counter() - t0
        cached = None
        if usage:
            ptd = usage.get("prompt_tokens_details") or {}
            cached = ptd.get("cached_tokens")
        return CompletionResult(
            req_id, True, None, t_send, t_first, t_first_text, e2e,
            usage.get("prompt_tokens") if usage else None, cached,
            usage.get("completion_tokens") if usage else None,
            "".join(text_parts), finish,
        )
