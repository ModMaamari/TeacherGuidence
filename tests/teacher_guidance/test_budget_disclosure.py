"""Tests for budget-disclosure (hidden-budget) mode in student/plan prompts."""

from agentsim.teacher_guidance.prompts import build_student_prompt, build_initial_plan_prompt
from agentsim.teacher_guidance.schemas import GuidanceConfig, PlanReviewConfig


def _state(disclose):
    return {"question": "who?", "step": 2, "budget": 9, "disclose_budget": disclose}


def test_disclosed_budget_shows_step_of_budget():
    p = build_student_prompt(_state(True), GuidanceConfig(level=3), force_finish=False)
    assert "Step 2 of budget 9." in p


def test_hidden_budget_omits_budget_and_urges_efficiency():
    p = build_student_prompt(_state(False), GuidanceConfig(level=3), force_finish=False)
    assert "of budget 9" not in p
    assert "budget" not in p.lower().split("this is step")[0] or "budget" not in p.lower()
    assert "this is step 2" in p.lower()
    assert "as soon as you are confident" in p.lower()


def test_hidden_budget_default_is_disclosed():
    # Missing key -> disclosed (backward compatible).
    p = build_student_prompt({"question": "q", "step": 1, "budget": 5}, GuidanceConfig(level=3), False)
    assert "Step 1 of budget 5." in p


def test_hidden_budget_plan_prompt_asks_for_shortest_plan_without_budget():
    cfg = PlanReviewConfig(enabled=True, max_initial_plan_steps=9)
    p = build_initial_plan_prompt(_state(False), cfg)
    assert "budget of 9" not in p
    assert "SHORTEST efficient plan" in p


def test_disclosed_plan_prompt_states_budget():
    cfg = PlanReviewConfig(enabled=True, max_initial_plan_steps=9)
    p = build_initial_plan_prompt(_state(True), cfg)
    assert "budget of 9 tool-use steps" in p


def test_force_finish_reveals_final_step_in_hidden_mode():
    p = build_student_prompt(_state(False), GuidanceConfig(level=3), force_finish=True)
    assert "final budget step" in p.lower()
    assert '"finish"' in p
