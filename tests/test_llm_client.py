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


def test_ollama_completion_sends_response_schema_as_format(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_ENDPOINT", "http://example.invalid")
    body = {"response": '{"tool": "search"}'}
    mock_client = _mock_async_client(_fake_response(body))
    schema = {"type": "object", "properties": {"tool": {"enum": ["search", "finish"]}}}
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(
            LLMClient().get_completion(prompt="hi", model="ollama/qwen3.5:0.8b", response_schema=schema)
        )

    sent_json = mock_client.post.call_args.kwargs["json"]
    assert sent_json["format"] == schema


def test_ollama_completion_omits_format_when_no_schema_given(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_ENDPOINT", "http://example.invalid")
    body = {"response": "hello"}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(LLMClient().get_completion(prompt="hi", model="ollama/qwen3.5:0.8b"))

    sent_json = mock_client.post.call_args.kwargs["json"]
    assert "format" not in sent_json


def test_custom_completion_sends_json_object_mode_when_schema_given(monkeypatch):
    # Full JSON-Schema enforcement isn't reliably supported across OpenRouter models,
    # so only the looser "valid JSON syntax" mode is requested, not the schema itself.
    monkeypatch.setattr(config, "CUSTOM_LLM_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(config, "CUSTOM_LLM_API_KEY", "key")
    body = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
    mock_client = _mock_async_client(_fake_response(body))
    schema = {"type": "object", "properties": {"teacher_decision": {"enum": ["continue"]}}}
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(
            LLMClient().get_completion(prompt="hi", model="custom/z-ai/glm-5.2", response_schema=schema)
        )

    sent_json = mock_client.post.call_args.kwargs["json"]
    assert sent_json["response_format"] == {"type": "json_object"}


def test_custom_completion_omits_response_format_when_no_schema_given(monkeypatch):
    monkeypatch.setattr(config, "CUSTOM_LLM_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(config, "CUSTOM_LLM_API_KEY", "key")
    body = {"choices": [{"message": {"content": "hello"}}], "usage": {}}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(LLMClient().get_completion(prompt="hi", model="custom/z-ai/glm-5.2"))

    sent_json = mock_client.post.call_args.kwargs["json"]
    assert "response_format" not in sent_json


def test_response_schema_unsupported_provider_raises():
    with pytest.raises(ValueError, match="response_schema"):
        asyncio.run(
            LLMClient().get_completion(prompt="hi", model="gpt-4o", response_schema={"type": "object"})
        )


# ---------------------------------------------------------------------------
# NHR@FAU gateway (fau/ provider)
# ---------------------------------------------------------------------------
def _fau_env(monkeypatch):
    monkeypatch.setattr(config, "FAU_LLM_ENDPOINT", "https://hub.nhr.fau.de/api/llmgw/v1")
    monkeypatch.setattr(config, "FAU_LLM_API_KEY", "sk-fau")


def test_fau_completion_posts_to_chat_completions_without_double_v1(monkeypatch):
    _fau_env(monkeypatch)
    body = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b"))

    url = mock_client.post.call_args.args[0]
    assert url == "https://hub.nhr.fau.de/api/llmgw/v1/chat/completions"  # no double /v1
    headers = mock_client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-fau"
    sent_json = mock_client.post.call_args.kwargs["json"]
    assert sent_json["model"] == "gpt-oss-120b"  # fau/ prefix stripped


def test_fau_completion_never_sends_response_format_even_with_schema(monkeypatch):
    # The gateway's json_object mode corrupts gpt-oss-120b output, so the schema must
    # never be turned into a response_format request (unlike the custom provider).
    _fau_env(monkeypatch)
    body = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
    mock_client = _mock_async_client(_fake_response(body))
    schema = {"type": "object", "properties": {"teacher_decision": {"enum": ["continue"]}}}
    with patch("httpx.AsyncClient", return_value=mock_client):
        asyncio.run(
            LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b", response_schema=schema)
        )

    sent_json = mock_client.post.call_args.kwargs["json"]
    assert "response_format" not in sent_json


def test_fau_completion_return_raw_and_usage_without_cost(monkeypatch):
    _fau_env(monkeypatch)
    body = {
        "id": "chatcmpl-xyz",
        "choices": [{"message": {"content": "hello", "reasoning_content": "thinking"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b", return_raw=True)
        )

    assert result["text"] == "hello"
    assert result["raw_response"] == body
    assert result["usage"]["total_tokens"] == 30
    assert result["usage"]["cost"] is None  # academic gateway, no billing


def test_fau_completion_coerces_none_content_to_empty(monkeypatch):
    # A reasoning model can truncate to empty content when its budget is exhausted.
    _fau_env(monkeypatch)
    body = {"choices": [{"message": {"content": None}}], "usage": {}}
    mock_client = _mock_async_client(_fake_response(body))
    with patch("httpx.AsyncClient", return_value=mock_client):
        text = asyncio.run(LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b"))
    assert text == ""


def test_fau_completion_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(config, "FAU_LLM_ENDPOINT", "https://hub.nhr.fau.de/api/llmgw/v1")
    monkeypatch.setattr(config, "FAU_LLM_API_KEY", None)
    with pytest.raises(ValueError, match="FAU_LLM_API_KEY"):
        asyncio.run(LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b"))
