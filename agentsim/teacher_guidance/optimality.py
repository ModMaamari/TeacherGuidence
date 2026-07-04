"""
Deterministic path-optimality signals for a finished episode.

Correctness alone doesn't make a trace good training data: a student can stumble onto
the right answer after redundant searches, failed extractions, and repeated no-ops.
Those sloppy paths teach bad habits. These signals quantify how *efficient* a trajectory
was so the filtering/selection stages can prefer clean, optimal paths.

Everything here is question-agnostic -- it inspects only the shape of the trajectory
(tool sequence, repetition, whether a search surfaced new documents, tool errors), never
the specific question or gold answer -- so selecting on it improves path quality without
overfitting to any dataset.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

_FAILED_STATUSES = {"error", "invalid"}


def _action_of(step: Dict[str, Any]) -> Dict[str, Any]:
    return ((step.get("student_action") or {}).get("action") or {})


def _action_signature(action: Dict[str, Any]) -> str:
    """A canonical (tool, params) signature so an identical re-issued call is detectable
    regardless of key ordering."""
    tool = action.get("tool", "") or ""
    params = action.get("params", {}) or {}
    try:
        params_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
    except TypeError:
        params_key = str(params)
    return f"{tool}::{params_key}"


def _search_result_doc_ids(observation: Dict[str, Any]) -> List[str]:
    results = ((observation.get("data") or {}).get("results") or [])
    ids: List[str] = []
    for r in results:
        if isinstance(r, dict) and r.get("doc_id"):
            ids.append(r["doc_id"])
        elif isinstance(r, str):
            ids.append(r)
    return ids


def compute_path_optimality(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return per-episode efficiency signals from the exported step records.

    * ``failed_tool_count`` -- steps whose tool errored / was invalid.
    * ``repeated_action_count`` -- steps re-issuing an exact earlier (tool, params) call.
    * ``wasted_search_count`` -- non-failed searches that surfaced no *new* document.
    * ``wasted_step_count`` -- distinct steps that were failed, repeated, or a wasted
      search (a step counts once even if it trips several of these).
    * ``step_efficiency`` -- ``1 - wasted_step_count / used_steps`` (1.0 = every step
      pulled its weight).
    """
    used = len(steps)
    failed = repeated = wasted_search = 0
    seen_sigs = set()
    seen_docs = set()
    wasted_step_indices = set()

    for i, s in enumerate(steps):
        action = _action_of(s)
        observation = s.get("tool_observation") or {}
        tool = action.get("tool", "") or ""
        is_failed = observation.get("status") in _FAILED_STATUSES

        if is_failed:
            failed += 1
            wasted_step_indices.add(i)

        sig = _action_signature(action)
        if tool != "finish" and sig in seen_sigs:
            repeated += 1
            wasted_step_indices.add(i)
        seen_sigs.add(sig)

        if tool == "search" and not is_failed:
            result_ids = _search_result_doc_ids(observation)
            if result_ids and all(d in seen_docs for d in result_ids):
                wasted_search += 1
                wasted_step_indices.add(i)
            seen_docs.update(result_ids)

    wasted_steps = len(wasted_step_indices)
    efficiency = round(1 - wasted_steps / used, 4) if used else 0.0
    return {
        "used_steps": used,
        "failed_tool_count": failed,
        "repeated_action_count": repeated,
        "wasted_search_count": wasted_search,
        "wasted_step_count": wasted_steps,
        "step_efficiency": efficiency,
    }
