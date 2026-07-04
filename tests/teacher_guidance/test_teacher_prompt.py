"""Tests for the teacher prompt's final-answer correctness judgment."""

from agentsim.teacher_guidance.prompts import build_teacher_prompt
from agentsim.teacher_guidance.schemas import GuidanceConfig


def _prompt(is_final):
    return build_teacher_prompt(
        state={"question": "q", "step": 5, "budget": 5},
        gold={"answer": "Delhi"},
        student_action={"action": {"tool": "finish", "params": {"answer": "Delhi"}}},
        tool_observation={"tool": "finish", "status": "ok"},
        guidance_config=GuidanceConfig(level=3),
        is_final_answer=is_final,
    )


def test_final_step_prompt_asks_for_correctness_fields():
    p = _prompt(is_final=True)
    assert "final_answer_correct" in p
    assert "final_answer_score" in p
    assert "compar" in p.lower()  # instructs comparing to gold


def test_non_final_step_prompt_omits_correctness_fields():
    p = _prompt(is_final=False)
    assert "final_answer_correct" not in p
    assert "final_answer_score" not in p


def test_default_is_non_final():
    p = build_teacher_prompt(
        state={"question": "q", "step": 1, "budget": 5},
        gold={"answer": "Delhi"},
        student_action={"action": {"tool": "search", "params": {}}},
        tool_observation={"tool": "search", "status": "ok"},
        guidance_config=GuidanceConfig(level=3),
    )
    assert "final_answer_correct" not in p
