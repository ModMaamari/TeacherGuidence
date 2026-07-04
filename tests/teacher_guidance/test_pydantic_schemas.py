"""Tests for the Pydantic student/teacher JSON contract models."""

import pytest
from pydantic import ValidationError

from agentsim.teacher_guidance.pydantic_schemas import (
    StudentActionModel,
    StudentActionGenerationModel,
    StudentFinishActionGenerationModel,
    StudentPlanModel,
    StudentPlanGenerationModel,
    RevisedStudentPlanModel,
    RevisedStudentPlanGenerationModel,
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


# ---------------------------------------------------------------------------
# *GenerationModel variants: stricter schemas used only to constrain Ollama's
# generation (see pydantic_schemas.py docstring for why an all-optional schema
# backfires under grammar-constrained decoding). The base *Model classes above stay
# lenient for post-hoc validation of already-produced JSON.
# ---------------------------------------------------------------------------
def test_student_action_generation_model_requires_substantive_thought():
    with pytest.raises(ValidationError) as exc_info:
        StudentActionGenerationModel.model_validate({"action": {"tool": "search", "params": {"query": "x"}}})
    assert any(e["loc"] == ("thought",) for e in exc_info.value.errors())


def test_student_action_generation_model_accepts_real_thought():
    m = StudentActionGenerationModel.model_validate({
        "thought": "I should search for the company headquarters first.",
        "action": {"tool": "search", "params": {"query": "x"}},
    })
    assert m.action.tool == "search"


def test_student_action_base_model_still_allows_missing_thought():
    # The lenient validation model (used by json_utils.validate_student_action) must
    # not reject old/legacy data or non-Ollama-generated JSON just because it lacks
    # a thought field -- that would be a validation-time behavior change, which is
    # not what this fix is about.
    m = StudentActionModel.model_validate({"action": {"tool": "search", "params": {}}})
    assert m.thought == ""


def test_student_plan_generation_model_requires_at_least_one_step():
    with pytest.raises(ValidationError) as exc_info:
        StudentPlanGenerationModel.model_validate({
            "plan_summary": "a reasonably long plan summary", "steps": [], "stop_condition": "done",
        })
    assert any(e["loc"] == ("steps",) for e in exc_info.value.errors())


def test_student_plan_generation_model_requires_step_goal_and_rationale():
    with pytest.raises(ValidationError):
        StudentPlanGenerationModel.model_validate({
            "plan_summary": "a reasonably long plan summary",
            "steps": [{"step_id": 1, "intended_tool": "search", "depends_on": []}],
            "stop_condition": "done",
        })


def test_student_plan_generation_model_accepts_well_formed_plan():
    m = StudentPlanGenerationModel.model_validate({
        "plan_summary": "Search for the company then verify the answer.",
        "steps": [{"step_id": 1, "goal": "find HQ", "intended_tool": "search",
                   "rationale": "need the source doc first", "depends_on": []}],
        "stop_condition": "have a verified answer",
    })
    assert len(m.steps) == 1


def test_student_plan_base_model_still_allows_empty_steps():
    m = StudentPlanModel.model_validate({"plan_summary": "", "steps": [], "stop_condition": ""})
    assert m.steps == []


def test_revised_student_plan_generation_model_requires_content():
    with pytest.raises(ValidationError):
        RevisedStudentPlanGenerationModel.model_validate({"steps": []})


def test_revised_student_plan_generation_model_accepts_well_formed_plan():
    m = RevisedStudentPlanGenerationModel.model_validate({
        "revision_summary": "added a verification step",
        "plan_summary": "Search then verify then answer.",
        "steps": [{"step_id": 1, "goal": "find HQ", "intended_tool": "search",
                   "rationale": "need the source doc", "depends_on": []}],
        "stop_condition": "verified",
    })
    assert m.steps[0].intended_tool == "search"


# ---------------------------------------------------------------------------
# Per-tool params (discriminated union, generation-only): a freeform params dict let
# the model get away with an empty {} for tools that need real content (e.g. a
# "decompose" with no sub_questions is a silent no-op). Only StudentActionGenerationModel
# enforces this; the base ToolCallModel (freeform params) stays lenient for validation.
# ---------------------------------------------------------------------------
def test_generation_model_rejects_decompose_with_empty_params():
    with pytest.raises(ValidationError):
        StudentActionGenerationModel.model_validate({
            "thought": "I need to break this question into sub-questions to search separately.",
            "action": {"tool": "decompose", "params": {}},
        })


def test_generation_model_accepts_decompose_with_sub_questions():
    m = StudentActionGenerationModel.model_validate({
        "thought": "I need to break this question into sub-questions to search separately.",
        "action": {"tool": "decompose", "params": {"sub_questions": ["who is X?", "what year?"]}},
    })
    assert m.action.params.sub_questions == ["who is X?", "what year?"]


def test_generation_model_rejects_search_with_empty_params():
    with pytest.raises(ValidationError):
        StudentActionGenerationModel.model_validate({
            "thought": "I should search for the company's headquarters location.",
            "action": {"tool": "search", "params": {}},
        })


def test_generation_model_accepts_search_with_query():
    m = StudentActionGenerationModel.model_validate({
        "thought": "I should search for the company's headquarters location.",
        "action": {"tool": "search", "params": {"query": "Oberoi Group HQ"}},
    })
    assert m.action.params.query == "Oberoi Group HQ"
    assert m.action.params.k == 5  # default


def test_generation_model_rejects_extract_missing_doc_ids():
    with pytest.raises(ValidationError):
        StudentActionGenerationModel.model_validate({
            "thought": "I found a relevant document and should extract the key fact from it.",
            "action": {"tool": "extract", "params": {"target_facts": ["some fact"]}},
        })


def test_generation_model_accepts_extract_with_doc_ids_and_facts():
    m = StudentActionGenerationModel.model_validate({
        "thought": "I found a relevant document and should extract the key fact from it.",
        "action": {"tool": "extract", "params": {"doc_ids": ["q1::doc0"], "target_facts": ["HQ is in Delhi"]}},
    })
    assert m.action.params.doc_ids == ["q1::doc0"]


def test_generation_model_synthesize_needs_no_params():
    m = StudentActionGenerationModel.model_validate({
        "thought": "I have enough extracted facts now, time to synthesize a draft answer.",
        "action": {"tool": "synthesize", "params": {}},
    })
    assert m.action.tool == "synthesize"


def test_generation_model_finish_requires_answer_via_discriminated_union():
    with pytest.raises(ValidationError):
        StudentActionGenerationModel.model_validate({
            "thought": "I believe I have found the final answer to this question.",
            "action": {"tool": "finish", "params": {}},
        })
    m = StudentActionGenerationModel.model_validate({
        "thought": "I believe I have found the final answer to this question.",
        "action": {"tool": "finish", "params": {"answer": "Delhi", "citations": []}},
    })
    assert m.action.params.answer == "Delhi"


def test_base_tool_call_model_still_allows_empty_params_for_any_tool():
    # Legacy/older data or the lenient post-hoc validation path must not reject an
    # empty params dict just because a stricter generation-only schema now exists.
    from agentsim.teacher_guidance.json_utils import validate_student_action
    ok, errors = validate_student_action({"action": {"tool": "decompose", "params": {}}})
    assert ok, errors


def test_generation_schema_includes_discriminated_action_union():
    schema = StudentActionGenerationModel.model_json_schema()
    action_schema = schema["properties"]["action"]
    assert action_schema["discriminator"]["propertyName"] == "tool"
    assert set(action_schema["discriminator"]["mapping"].keys()) == {
        "decompose", "reformulate", "search", "extract", "verify", "synthesize", "finish",
    }


# ---------------------------------------------------------------------------
# Finish-only generation schema: used on the final force-finish step so the model
# turns its context into a real answer instead of searching again (which left the
# system fabricating answer "unknown"). See pydantic_schemas.py.
# ---------------------------------------------------------------------------
def test_finish_generation_schema_pins_tool_to_finish():
    schema = StudentFinishActionGenerationModel.model_json_schema()
    action_schema = schema["properties"]["action"]
    # A single FinishCall $ref -- no discriminated union of other tools.
    assert "discriminator" not in action_schema
    assert action_schema["$ref"].endswith("/FinishCall")
    finish_call = schema["$defs"]["FinishCall"]
    assert finish_call["properties"]["tool"]["const"] == "finish"


def test_finish_generation_model_rejects_non_finish_tool():
    with pytest.raises(ValidationError):
        StudentFinishActionGenerationModel.model_validate({
            "thought": "I still want to search for more evidence before answering.",
            "action": {"tool": "search", "params": {"query": "x"}},
        })


def test_finish_generation_model_requires_non_empty_answer():
    with pytest.raises(ValidationError):
        StudentFinishActionGenerationModel.model_validate({
            "thought": "I am ready to give my final answer to the question now.",
            "action": {"tool": "finish", "params": {"answer": ""}},
        })


def test_finish_generation_model_accepts_real_answer():
    m = StudentFinishActionGenerationModel.model_validate({
        "thought": "Based on the retrieved documents the answer is clearly Broadcasting House.",
        "action": {"tool": "finish", "params": {"answer": "Broadcasting House in London", "citations": []}},
    })
    assert m.action.tool == "finish"
    assert m.action.params.answer == "Broadcasting House in London"
