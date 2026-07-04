"""Tests for LLMClient.get_completion's return_raw support (custom/ollama providers)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentsim.clients.llm_client import LLMClient
from agentsim.config import config


def _fake_response(body):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=body)
    return resp


def _status_response(body, status_code=200, headers=None):
    """A response with a real status_code/headers and a raise_for_status that raises
    on >= 400, for exercising the retry/backoff path."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json = MagicMock(return_value=body)

    def _raise():
        if status_code >= 400:
            raise httpx.HTTPStatusError("err", request=MagicMock(), response=resp)

    resp.raise_for_status = MagicMock(side_effect=_raise)
    return resp


def _mock_async_client(response):
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _mock_async_client_seq(responses):
    """Mock client whose .post yields the given responses in order across calls."""
    client = AsyncMock()
    client.post = AsyncMock(side_effect=responses)
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


# ---------------------------------------------------------------------------
# 429/503 retry + backoff on the FAU gateway
# ---------------------------------------------------------------------------
def test_fau_retries_on_429_then_succeeds(monkeypatch):
    _fau_env(monkeypatch)
    ok_body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    responses = [
        _status_response({}, status_code=429),
        _status_response({}, status_code=429),
        _status_response(ok_body, status_code=200),
    ]
    mock_client = _mock_async_client_seq(responses)
    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("agentsim.clients.llm_client.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        text = asyncio.run(LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b"))

    assert text == "ok"
    assert mock_client.post.await_count == 3      # two 429s then success
    assert sleep_mock.await_count == 2            # backed off before each retry


def test_fau_honors_retry_after_header(monkeypatch):
    _fau_env(monkeypatch)
    ok_body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    responses = [
        _status_response({}, status_code=429, headers={"Retry-After": "7"}),
        _status_response(ok_body, status_code=200),
    ]
    mock_client = _mock_async_client_seq(responses)
    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("agentsim.clients.llm_client.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        asyncio.run(LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b"))

    # Slept at least the server-requested 7s (plus a little jitter), not the 1s default.
    slept = sleep_mock.await_args.args[0]
    assert 7.0 <= slept < 8.0


def test_fau_gives_up_after_max_retries_and_raises(monkeypatch):
    _fau_env(monkeypatch)
    # Always 429: after self.max_retries retries the final 429 surfaces via raise_for_status.
    always_429 = [_status_response({}, status_code=429) for _ in range(10)]
    mock_client = _mock_async_client_seq(always_429)
    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("agentsim.clients.llm_client.asyncio.sleep", new=AsyncMock()):
        client = LLMClient()
        client.max_retries = 3
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.get_completion(prompt="hi", model="fau/gpt-oss-120b"))

    assert mock_client.post.await_count == 4       # initial + 3 retries


def test_fau_does_not_retry_on_400(monkeypatch):
    _fau_env(monkeypatch)
    mock_client = _mock_async_client_seq([_status_response({}, status_code=400)])
    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("agentsim.clients.llm_client.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(LLMClient().get_completion(prompt="hi", model="fau/gpt-oss-120b"))

    assert mock_client.post.await_count == 1       # non-retryable, no backoff
    assert sleep_mock.await_count == 0


# ---------------------------------------------------------------------------
# Provider-fallback router (get_completion_with_fallback)
# ---------------------------------------------------------------------------
def _http_error(status=429):
    return httpx.HTTPStatusError("rate limited", request=MagicMock(),
                                 response=_status_response({}, status_code=status))


def test_router_returns_first_success():
    client = LLMClient()
    calls = []

    async def fake_get_completion(*, prompt, model, **kw):
        calls.append((model, kw.get("max_retries")))
        return {"text": "ok from " + model}

    client.get_completion = fake_get_completion
    result, used = asyncio.run(client.get_completion_with_fallback(
        ["fau/gpt-oss-120b", "custom/openai/gpt-oss-120b:free", "custom/openai/gpt-oss-120b"],
        prompt="hi", return_raw=True,
    ))
    assert used == "fau/gpt-oss-120b"
    assert result["text"].endswith("fau/gpt-oss-120b")
    assert calls == [("fau/gpt-oss-120b", 0)]  # first is fail-fast, and it succeeded


def test_router_falls_through_on_rate_limit():
    client = LLMClient()
    seen = []

    async def fake_get_completion(*, prompt, model, **kw):
        seen.append(model)
        if model != "custom/openai/gpt-oss-120b:free":
            raise _http_error(429)
        return {"text": "free served it"}

    client.get_completion = fake_get_completion
    result, used = asyncio.run(client.get_completion_with_fallback(
        ["fau/gpt-oss-120b", "custom/openai/gpt-oss-120b:free", "custom/openai/gpt-oss-120b"],
        prompt="hi",
    ))
    assert used == "custom/openai/gpt-oss-120b:free"
    assert seen == ["fau/gpt-oss-120b", "custom/openai/gpt-oss-120b:free"]


def test_router_uses_paid_last_and_raises_if_all_fail():
    client = LLMClient()
    order = []

    async def fake_get_completion(*, prompt, model, **kw):
        order.append((model, kw.get("max_retries")))
        raise _http_error(429)

    client.get_completion = fake_get_completion
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.get_completion_with_fallback(
            ["fau/gpt-oss-120b", "custom/openai/gpt-oss-120b:free", "custom/openai/gpt-oss-120b"],
            prompt="hi",
        ))
    # last model called with default retries (None = normal), earlier ones fail-fast (0)
    assert order[0][1] == 0 and order[1][1] == 0
    assert order[2][1] is None


def test_router_requires_models():
    with pytest.raises(ValueError):
        asyncio.run(LLMClient().get_completion_with_fallback([], prompt="hi"))
