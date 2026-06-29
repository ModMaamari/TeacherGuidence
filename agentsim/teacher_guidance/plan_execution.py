"""
Formal plan validation and programmatic adherence tracking.

A formal plan is an ordered list of steps, each naming an ``intended_tool``. The
student still chooses each action, but the environment deterministically verifies, per
step, whether the executed tool matches the plan's expected tool, and reports an
episode-level adherence score. This is the "programmatic enforcement": a checker that
labels followed/deviated steps (it does not silently rewrite tool params).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agentsim.teacher_guidance.schemas import TOOLS


def validate_formal_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that a plan has well-formed, executable steps."""
    errors: List[str] = []
    if not isinstance(plan, dict):
        return False, ["plan_not_object"]
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False, ["missing_steps"]
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step_{i}_not_object")
            continue
        if step.get("intended_tool") not in TOOLS:
            errors.append(f"step_{i}_invalid_tool:{step.get('intended_tool')}")
        if not str(step.get("goal", "")).strip():
            errors.append(f"step_{i}_missing_goal")
    return len(errors) == 0, errors


class PlanTracker:
    """Aligns executed actions with the formal plan's steps (one per action) and
    records whether each was followed."""

    def __init__(self, plan: Dict[str, Any]):
        self.steps: List[Dict[str, Any]] = (
            plan.get("steps", []) if isinstance(plan, dict) else []
        )
        self.cursor = 0
        self.followed = 0
        self.evaluated = 0

    def expected_step(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.cursor < len(self.steps):
            return self.steps[self.cursor]
        return None

    def expected_tool(self) -> Optional[str]:
        step = self.expected_step()
        return step.get("intended_tool") if step else None

    def record(self, action_tool: str) -> Dict[str, Any]:
        """Record the executed tool against the current expected step and advance."""
        expected = self.expected_tool()
        followed = expected is not None and action_tool == expected
        if expected is not None:
            self.evaluated += 1
            if followed:
                self.followed += 1
        result = {
            "plan_cursor": self.cursor,
            "expected_tool": expected,
            "plan_step_followed": followed,
        }
        self.cursor += 1
        return result

    def adherence(self) -> float:
        """Fraction of plan-covered steps whose executed tool matched the plan."""
        return round(self.followed / self.evaluated, 4) if self.evaluated else 0.0
