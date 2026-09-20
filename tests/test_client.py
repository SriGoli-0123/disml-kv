"""Streaming-client parsing against a mock OpenAI-compatible ASGI server."""
import asyncio
import json

import httpx
import pytest

from aclkv.bench import summarize_run
from aclkv.client import VLLMClient
from aclkv.context import BuiltPrompt, Segment
from aclkv.acl import PUBLIC
from aclkv.metrics import counter_delta, parse_prometheus


async def mock_app(scope, receive, send):
    assert scope["type"] == "http"
    path = scope["path"]
    if path == "/health":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
        return
    if path == "/v1/models":
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps({"data": [{"id": "mock-model"}]}).encode()})
        return
    if path == "/metrics":
        body = ('# HELP x\nvllm:prefix_cache_hits_total{engine="0",model_name="m"} 12.0\n'
                'vllm:prefix_cache_queries_total{engine="0",model_name="m"} 40.0\n'
                'vllm:cache_config_info{block_size="16",num_gpu_blocks="2340",engine="0"} 1.0\n')
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body.encode()})
        return
    body = b""
    while True:
        msg = await receive()
        body += msg.get("body", b"")
        if not msg.get("more_body"):
            break
    req = json.loads(body)
    assert isinstance(req["prompt"], list) and req["stream"] is True
    salt = req.get("cache_salt")
    cached = 64 if salt == "hit-me" else 0
    chunks = [
        {"id": "c", "choices": [{"index": 0, "text": "Paris", "finish_reason": None}], "usage": None},
        {"id": "c", "choices": [{"index": 0, "text": "", "finish_reason": "stop"}], "usage": None},
        {"id": "c", "choices": [], "usage": {"prompt_tokens": len(req["prompt"]), "completion_tokens": 1,
                                             "prompt_tokens_details": {"cached_tokens": cached}}},
    ]
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
    for ch in chunks:
        await asyncio.sleep(0.005)
        await send({"type": "http.response.body", "body": f"data: {json.dumps(ch)}\n\n".encode(), "more_body": True})
    await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})


@pytest.mark.asyncio
async def test_complete_parses_stream_usage_and_ttft():
    client = VLLMClient("http://mock")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app)) as s:
        await client.wait_ready(s, 5)
        assert await client.resolve_model(s) == "mock-model"
        r = await client.complete(s, "r1", [1, 2, 3, 4], "hit-me", max_tokens=4)
        assert r.ok and r.text == "Paris" and r.cached_tokens == 64 and r.prompt_tokens == 4
        assert r.ttft_s is not None and r.ttft_text_s is not None and r.e2e_s >= r.ttft_s
        r2 = await client.complete(s, "r2", [1, 2, 3, 4], None)
        assert r2.cached_tokens == 0
        m = await client.metrics(s)
        assert m["cache_config"]["num_gpu_blocks"] == "2340"
        assert counter_delta({"counters": {"vllm:prefix_cache_hits_total": 2.0}}, m)["vllm:prefix_cache_hits_total"] == 10.0

        p = BuiltPrompt("r1", "u", [1, 2, 3, 4], [Segment("tail", 0, 4, 0, None, PUBLIC)], [], None, [], meta={"answer": "Paris"})
        summ = summarize_run([p], [r], wall=1.0)
        assert summ["cached_pct"] == 100.0 * 64 / 4 and summ["qa"]["em"] == 1.0 and summ["req_per_s"] == 1.0


def test_parse_prometheus_sums_labels():
    m = parse_prometheus('vllm:num_preemptions_total{engine="0"} 3\nvllm:num_preemptions_total{engine="1"} 4\n')
    assert m["counters"]["vllm:num_preemptions_total"] == 7.0
