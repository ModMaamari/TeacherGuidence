"""Shared call-level logging for every LLM call in the Teacher Guidance pipeline.

Captures one log entry per actual HTTP request (not per logical student/teacher
turn), so a failed repair attempt's raw text and the full provider response body
are preserved rather than only the winning attempt's extracted text.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def timed_completion(
    llm_client: Any,
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempt: int = 1,
) -> Tuple[Dict[str, Any], str]:
    """Call ``llm_client.get_completion(..., return_raw=True)`` and return
    ``(call_log_entry, response_text)``.

    Tolerates clients (e.g. test stubs) that ignore ``return_raw`` and return a bare
    string instead of ``{"text": ..., "raw_response": ...}`` -- ``raw_response`` is
    ``None`` in that case.
    """
    started_at = _utcnow_iso()
    t0 = time.time()
    result = await llm_client.get_completion(
        prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, return_raw=True,
    )
    elapsed_ms = (time.time() - t0) * 1000
    ended_at = _utcnow_iso()

    if isinstance(result, dict):
        text = result.get("text", "")
        raw_response = result.get("raw_response")
    else:
        text = result
        raw_response = None

    entry = {
        "attempt": attempt,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_ms": elapsed_ms,
        "prompt": prompt,
        "response_text": text,
        "raw_response": raw_response,
    }
    return entry, text
