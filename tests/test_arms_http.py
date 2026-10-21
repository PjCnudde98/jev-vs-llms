"""The real Jev and Anthropic arms against mocked HTTP: checks we read the SDK responses right."""

import asyncio
import json

import anthropic
import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_eval.arms.base import RetryableError
from jev_eval.arms.jev import JevArm
from jev_eval.arms.llm import AnthropicArm


def _jev_with(task, handler):
    arm = JevArm("jev", "jev-1.13.0", task)
    arm.fit([])
    arm._client = AsyncTypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0)
    )
    return arm


def test_jev_parses_choice_answer(task, items):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "intent": {
                        "type": "choice",
                        "choice": "shipping",
                        "confidence": 0.6,
                        "probabilities": {"billing": 0.1, "shipping": 0.7, "returns": 0.1, "oos": 0.1},
                    }
                },
                "usage": {"input_tokens": 1000, "output_tokens": 20},
            },
        )

    p = asyncio.run(_jev_with(task, handler).predict(items[0]))
    assert p.valid and p.answers["intent"].label == "shipping"
    assert p.answers["intent"].confidence == 0.7 and p.answers["intent"].vendor_confidence == 0.6
    assert p.cost_usd == pytest.approx(1000 * 0.042e-6)
    q = seen["body"]["questions"]["intent"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(task.fields[0].labels)


def test_jev_rate_limit_is_retryable(task, items):
    arm = _jev_with(task, lambda r: httpx2.Response(429, json={"detail": "slow down"}))
    with pytest.raises(RetryableError):
        asyncio.run(arm.predict(items[0]))


def test_jev_bad_request_is_invalid(task, items):
    arm = _jev_with(task, lambda r: httpx2.Response(422, json={"detail": []}))
    p = asyncio.run(arm.predict(items[0]))
    assert not p.valid and "422" in p.error


def _anthropic_with(task, text, stop="end_turn"):
    def handler(request):
        body = json.loads(request.content)
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["effort"] == "low"
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    arm = AnthropicArm("haiku", "claude-haiku-5-5", task, effort="low")
    arm.fit([])
    arm._client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return arm


def test_anthropic_parses_structured_output(task, items):
    arm = _anthropic_with(task, '{"intent": {"label": "returns", "confidence": 0.8}}')
    p = asyncio.run(arm.predict(items[0]))
    assert p.valid and p.answers["intent"].label == "returns" and p.answers["intent"].confidence == 0.8
    # 100 x $0.10 + 50 x $0.50 + 1000 cached x $0.01 (0.1x input), per million
    assert p.cost_usd == pytest.approx((100 * 0.10 + 50 * 0.50 + 1000 * 0.01) * 1e-6)


def test_anthropic_out_of_schema_is_invalid(task, items):
    p = asyncio.run(_anthropic_with(task, '{"intent": {"label": "refunds", "confidence": 0.8}}').predict(items[0]))
    assert not p.valid and p.raw


def test_anthropic_refusal_is_invalid(task, items):
    p = asyncio.run(_anthropic_with(task, "", stop="refusal").predict(items[0]))
    assert not p.valid and "refusal" in p.error
