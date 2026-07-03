"""Tests for the Pydantic student/teacher JSON contract models."""

import pytest
from pydantic import ValidationError

from agentsim.teacher_guidance.pydantic_schemas import (
    StudentActionModel,
    StudentPlanModel,
    RevisedStudentPlanModel,
    TeacherEvaluationModel,
    TeacherPlanReviewModel,
)


def test_student_action_schema_generation_includes_tool_enum():
    schema = StudentActionModel.model_json_schema()
    tool_schema = schema["$defs"]["ToolCallModel"]["properties"]["tool"]
    assert set(tool_schema["enum"]) == {
        "decompose", "reformulate", "search", "extract", "verify", "synthesize", "finish",
    }


def test_student_action_accepts_valid_payload():
    m = StudentActionModel.model_validate({
        "thought": "search first",
        "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group", "k": 5}},
        "new_facts_extracted": [],
    })
    assert m.action.tool == "search"
    assert m.action.params["k"] == 5


def test_student_action_rejects_invalid_tool():
    with pytest.raises(ValidationError) as exc_info:
        StudentActionModel.model_validate({"action": {"tool": "google_it", "params": {}}})
    assert any("tool" in str(e["loc"]) for e in exc_info.value.errors())


def test_student_action_rejects_finish_without_answer():
    with pytest.raises(ValidationError) as exc_info:
        StudentActionModel.model_validate({"action": {"tool": "finish", "params": {}}})
    assert any("finish_missing_answer" in e["msg"] for e in exc_info.value.errors())


def test_student_action_accepts_finish_with_answer():
    m = StudentActionModel.model_validate({"action": {"tool": "finish", "params": {"answer": "Delhi"}}})
    assert m.action.params["answer"] == "Delhi"


def test_student_action_rejects_invalid_decision_category():
    with pytest.raises(ValidationError):
        StudentActionModel.model_validate({
            "action": {"tool": "search", "params": {}},
            "decision": {"category": "not_a_real_category"},
        })


def test_student_plan_schema_generation_includes_intended_tool_enum():
    schema = StudentPlanModel.model_json_schema()
    step_schema = schema["$defs"]["PlanStepModel"]["properties"]["intended_tool"]
    assert "finish" in step_schema["enum"]


def test_student_plan_accepts_valid_payload():
    m = StudentPlanModel.model_validate({
        "plan_summary": "search then answer",
        "steps": [{"step_id": 1, "goal": "find HQ", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [],
        "stop_condition": "have HQ",
    })
    assert m.steps[0].intended_tool == "search"


def test_revised_student_plan_accepts_valid_payload():
    m = RevisedStudentPlanModel.model_validate({
        "revision_summary": "added verify",
        "plan_summary": "search, verify, answer",
        "steps": [{"step_id": 1, "goal": "g", "intended_tool": "verify", "rationale": "r", "depends_on": []}],
        "teacher_feedback_used": ["added verification"],
        "stop_condition": "verified",
    })
    assert m.steps[0].intended_tool == "verify"


def test_teacher_evaluation_accepts_valid_payload():
    m = TeacherEvaluationModel.model_validate({
        "guidance_level": 3,
        "student_visible": {"score_continuous": 0.7},
        "private_diagnosis": {"step_correct": True},
        "teacher_decision": "continue",
    })
    assert m.teacher_decision == "continue"


def test_teacher_evaluation_rejects_invalid_decision():
    with pytest.raises(ValidationError):
        TeacherEvaluationModel.model_validate({"teacher_decision": "explode"})


def test_teacher_plan_review_accepts_valid_payload():
    m = TeacherPlanReviewModel.model_validate({
        "plan_review_enabled": True,
        "review_guidance_level": 3,
        "student_visible": {"feedback": "Solid plan."},
        "private_diagnosis": {},
        "teacher_decision": "accept_plan",
    })
    assert m.teacher_decision == "accept_plan"


def test_teacher_plan_review_rejects_invalid_decision():
    with pytest.raises(ValidationError):
        TeacherPlanReviewModel.model_validate({"teacher_decision": "explode"})
