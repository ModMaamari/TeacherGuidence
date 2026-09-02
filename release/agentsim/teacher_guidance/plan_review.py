"""Helpers for the optional plan-review preflight phase."""

from __future__ import annotations

from typing import Any, Dict, List


def _tools_in_plan(plan: Dict[str, Any]) -> List[str]:
    return [str(step.get("intended_tool", "")) for step in (plan or {}).get("steps", []) or []]


def compute_plan_review_metrics(
    initial_plan: Dict[str, Any],
    revised_plan: Dict[str, Any],
    review_full: Dict[str, Any],
) -> Dict[str, Any]:
    """Lightweight, deterministic plan-review metrics."""
    initial_tools = _tools_in_plan(initial_plan)
    revised_tools = _tools_in_plan(revised_plan)
    private = (review_full or {}).get("private_diagnosis", {}) or {}
    decision = (review_full or {}).get("teacher_decision")
    return {
        "initial_step_count": len(initial_tools),
        "revised_step_count": len(revised_tools),
        "plan_changed": initial_tools != revised_tools,
        "revision_skipped": decision == "accept_plan",
        "initial_tools": initial_tools,
        "revised_tools": revised_tools,
        "initial_covers_verification": "verify" in initial_tools,
        "revised_covers_verification": "verify" in revised_tools,
        "teacher_decision": (review_full or {}).get("teacher_decision"),
        "premature_answering_risk": bool(private.get("premature_answering_risk", False)),
    }
