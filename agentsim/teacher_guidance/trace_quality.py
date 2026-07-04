"""
Trace-quality gate: decide whether one episode is good enough to become training data.

A trace is worth fine-tuning on only if it teaches the behaviour we want the student to
generalize: reach a *correct* answer, *grounded* in evidence it actually retrieved, via a
*clean* path (no invalid JSON, no wasted/redundant steps), that *stops on its own* rather
than being force-finished at the budget, and that never saw leaked gold information.

Every criterion is a general property of the trajectory, not a fact about the specific
question, so accepting on them selects for good agentic behaviour without overfitting.
Thresholds live in ``TraceCriteria`` so the filtering CLI can tune strictness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# stop_reason values (see teacher_guided_agent_step._update_done). A "natural" finish is
# the student choosing to finish and being accepted -- not a budget-exhaustion force.
NATURAL_FINISH_REASONS = {"teacher_accept"}


@dataclass
class TraceCriteria:
    require_correct: bool = True
    require_grounded: bool = True
    forbid_gold_answer_leak: bool = True
    require_all_steps_json_valid: bool = True
    require_natural_finish: bool = True
    min_step_efficiency: float = 0.5
    max_wasted_steps: int = 1
    min_steps: int = 1  # reject degenerate zero/one-step "traces"


def evaluate_trace(episode: Dict[str, Any], criteria: TraceCriteria = TraceCriteria()) -> Dict[str, Any]:
    """Return ``{"accepted": bool, "reasons": [failure_code, ...]}``. ``reasons`` is empty
    iff accepted; each code names a criterion the trace failed."""
    reasons: List[str] = []
    fm = episode.get("final_metrics", {}) or {}
    po = episode.get("path_optimality", {}) or {}
    steps = episode.get("steps", []) or []

    if criteria.require_correct and not fm.get("answer_correct"):
        reasons.append("incorrect")
    # answer_grounded may be absent on older runs; treat missing as a fail only when
    # grounding is required (recompute upstream if you need to re-score legacy data).
    if criteria.require_grounded and not fm.get("answer_grounded", False):
        reasons.append("ungrounded")
    if criteria.forbid_gold_answer_leak and any(
        (s.get("leakage_check") or {}).get("gold_answer_leaked") for s in steps
    ):
        reasons.append("gold_answer_leaked")
    if criteria.require_all_steps_json_valid and not all(
        (s.get("metrics") or {}).get("json_valid", False) for s in steps
    ):
        reasons.append("invalid_step")
    if criteria.require_natural_finish and episode.get("stop_reason") not in NATURAL_FINISH_REASONS:
        reasons.append("not_natural_finish")
    if po.get("step_efficiency", 0.0) < criteria.min_step_efficiency:
        reasons.append("low_efficiency")
    if po.get("wasted_step_count", 0) > criteria.max_wasted_steps:
        reasons.append("too_many_wasted_steps")
    if len(steps) < criteria.min_steps:
        reasons.append("too_few_steps")

    return {"accepted": not reasons, "reasons": reasons}
