"""Tests for the shared timed_completion call-logging helper."""

import asyncio

from agentsim.teacher_guidance.llm_call_log import timed_completion


class DictStub:
    """Mimics a real LLMClient honoring return_raw."""

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, return_raw=False, **kw):
        assert return_raw is True
        return {"text": f"echo:{prompt}", "raw_response": {"id": "abc", "choices": []}}


class BareStringStub:
    """Mimics a test stub that ignores return_raw and returns a plain string."""

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        return f"echo:{prompt}"


class UsageStub:
    """Mimics a real LLMClient returning token usage/cost alongside the raw response."""

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, return_raw=False, **kw):
        return {
            "text": f"echo:{prompt}",
            "raw_response": {"id": "abc"},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }


def test_timed_completion_captures_raw_response_and_timing():
    entry, text = asyncio.run(
        timed_completion(DictStub(), prompt="hi", model="m", temperature=0.1, max_tokens=100, attempt=2)
    )
    assert text == "echo:hi"
    assert entry["attempt"] == 2
    assert entry["response_text"] == "echo:hi"
    assert entry["prompt"] == "hi"
    assert entry["raw_response"] == {"id": "abc", "choices": []}
    assert entry["elapsed_ms"] >= 0
    assert entry["started_at"] <= entry["ended_at"]
    assert entry["usage"] is None


def test_timed_completion_tolerates_bare_string_client():
    entry, text = asyncio.run(
        timed_completion(BareStringStub(), prompt="hi", model="m", temperature=0.1, max_tokens=100)
    )
    assert text == "echo:hi"
    assert entry["attempt"] == 1
    assert entry["raw_response"] is None
    assert entry["usage"] is None


def test_timed_completion_captures_usage_and_cost():
    entry, _ = asyncio.run(
        timed_completion(UsageStub(), prompt="hi", model="m", temperature=0.1, max_tokens=100)
    )
    assert entry["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001}
