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
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            LLMClient().get_completion(prompt="hi", model="custom/z-ai/glm-5.2", return_raw=True)
        )

    assert result["text"] == "hello"
    assert result["raw_response"] == body
    assert result["usage"]["total_tokens"] == 3


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
