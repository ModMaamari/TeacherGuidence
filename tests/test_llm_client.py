"""Tests for LLMClient.get_completion's return_raw support (custom/ollama providers)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsim.clients.llm_client import LLMClient
from agentsim.config import config


def _fake_response(body):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=body)
    return resp


def _mock_async_client(response):
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def test_custom_completion_return_raw_includes_full_body(monkeypatch):
    monkeypatch.setattr(config, "CUSTOM_LLM_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(config, "CUSTOM_LLM_API_KEY", "key")
    body = {
        "id": "gen-abc123",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.00042},
    }
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            LLMClient().get_completion(prompt="hi", model="custom/z-ai/glm-5.2", return_raw=True)
        )

    assert result["text"] == "hello"
    assert result["raw_response"] == body
    assert result["usage"]["total_tokens"] == 3
    assert result["usage"]["cost"] == 0.00042


def test_custom_completion_requests_openrouter_usage_include(monkeypatch):
    # OpenRouter only returns per-call cost when the request explicitly opts in.
    monkeypatch.setattr(config, "CUSTOM_LLM_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(config, "CUSTOM_LLM_API_KEY", "key")
    body = {"choices": [{"message": {"content": "hello"}}], "usage": {}}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(LLMClient().get_completion(prompt="hi", model="custom/z-ai/glm-5.2"))

    sent_json = mock_client.post.call_args.kwargs["json"]
    assert sent_json["usage"] == {"include": True}


def test_ollama_completion_return_raw_includes_full_body(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_ENDPOINT", "http://example.invalid")
    body = {"response": "hello", "done_reason": "stop", "eval_count": 12, "eval_duration": 456}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            LLMClient().get_completion(prompt="hi", model="ollama/qwen3.5:0.8b", return_raw=True)
        )

    assert result["text"] == "hello"
    assert result["raw_response"] == body


def test_ollama_completion_return_raw_strips_context_token_ids(monkeypatch):
    # 'context' is a raw undecoded token-ID list -- confusing and useless in the raw
    # response viewer since Ollama has no detokenize endpoint. It should be dropped
    # and replaced with a plain length, leaving everything else untouched.
    monkeypatch.setattr(config, "OLLAMA_ENDPOINT", "http://example.invalid")
    body = {"response": "hello", "done_reason": "stop", "context": [1, 2, 3, 4, 5]}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            LLMClient().get_completion(prompt="hi", model="ollama/qwen3.5:0.8b", return_raw=True)
        )

    assert "context" not in result["raw_response"]
    assert result["raw_response"]["context_length"] == 5
    assert result["raw_response"]["response"] == "hello"
    assert result["raw_response"]["done_reason"] == "stop"


def test_get_completion_without_return_raw_still_returns_plain_string(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_ENDPOINT", "http://example.invalid")
    body = {"response": "hello"}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(LLMClient().get_completion(prompt="hi", model="ollama/qwen3.5:0.8b"))

    assert result == "hello"


def test_return_raw_unsupported_provider_raises():
    with pytest.raises(ValueError, match="return_raw"):
        asyncio.run(
            LLMClient().get_completion(prompt="hi", model="gpt-4o", return_raw=True)
        )
