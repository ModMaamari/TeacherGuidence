"""Tests for teacher_guidance.json_utils."""

from agentsim.teacher_guidance.json_utils import (
    extract_first_json_object,
    parse_json_object,
    parse_student_action,
    parse_teacher_evaluation,
    validate_student_action,
    validate_teacher_evaluation,
)


def test_extract_plain_object():
    assert extract_first_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_with_prose_and_fence():
    raw = 'Sure!\n```json\n{"a": {"b": 2}}\n```\nDone.'
    extracted = extract_first_json_object(raw)
    assert extracted == '{"a": {"b": 2}}'


def test_extract_respects_braces_in_strings():
    raw = '{"text": "a } b { c", "n": 1}'
    extracted = extract_first_json_object(raw)
    assert extracted == raw


def test_parse_invalid_json_records_failure():
    obj, info = parse_json_object("not json at all")
    assert obj == {}
    assert info["json_valid"] is False
    assert info["errors"]


def test_parse_student_action_valid():
    raw = """{
      "thought": "search first",
      "decision": {"category": "need_retrieval", "parametric_knowledge_used": false},
      "action": {"tool": "search", "params": {"query": "Oberoi family", "k": 5}},
      "new_facts_extracted": []
    }"""
    action, info = parse_student_action(raw)
    assert info["json_valid"] and info["action_valid"]
    assert action.action.tool == "search"
    assert action.action.params["k"] == 5


def test_validate_student_action_rejects_bad_tool():
    ok, errors = validate_student_action(
        {"action": {"tool": "google_it", "params": {}}}
    )
    assert not ok
    assert any("invalid_tool" in e for e in errors)


def test_parse_teacher_evaluation_valid():
    raw = """{
      "guidance_level": 3,
      "student_visible": {"score_binary": 0, "score_continuous": 0.35, "feedback": "partial"},
      "private_diagnosis": {"step_correct": false},
      "teacher_decision": "continue"
    }"""
    ev, info = parse_teacher_evaluation(raw)
    assert info["json_valid"] and info["eval_valid"]
    assert ev.teacher_decision == "continue"
    assert ev.student_visible["score_continuous"] == 0.35


def test_validate_teacher_evaluation_rejects_bad_decision():
    ok, errors = validate_teacher_evaluation(
        {"student_visible": {}, "private_diagnosis": {}, "teacher_decision": "explode"}
    )
    assert not ok
    assert any("invalid_teacher_decision" in e for e in errors)


def test_plan_review_config_new_fields():
    from agentsim.teacher_guidance.schemas import PlanReviewConfig
    # defaults
    d = PlanReviewConfig.from_mode_config({"plan_review": {"enabled": True}})
    assert d.planner == "student" and d.planning_steps == 1 and d.formal_plan is False
    # overrides
    o = PlanReviewConfig.from_mode_config({"plan_review": {
        "enabled": True, "planner": "teacher", "planning_steps": 3, "formal_plan": True}})
    assert o.planner == "teacher" and o.planning_steps == 3 and o.formal_plan is True
    # planning_steps is clamped to >= 1
    assert PlanReviewConfig.from_mode_config({"plan_review": {"planning_steps": 0}}).planning_steps == 1


def test_repair_trailing_comma():
    obj, info = parse_json_object('{"action": {"tool": "search", "params": {"k": 5,},},}')
    assert info["json_valid"] and info["repaired"]
    assert obj["action"]["tool"] == "search"


def test_repair_curly_quotes():
    obj, info = parse_json_object('{“action”: {“tool”: “finish”}}')
    assert info["json_valid"] and info["repaired"]
    assert obj["action"]["tool"] == "finish"


def test_valid_json_not_marked_repaired():
    obj, info = parse_json_object('{"a": 1}')
    assert info["json_valid"] and info["repaired"] is False


def test_student_prompt_has_format_example_without_gold():
    from agentsim.teacher_guidance.prompts import build_student_prompt
    from agentsim.teacher_guidance.schemas import GuidanceConfig
    state = {"question": "Where is the Oberoi Group HQ?", "step": 1, "budget": 5,
             "previous_actions": [], "retrieved_docs": [], "extracted_facts": [],
             "draft_answer": None, "sub_questions": [], "previous_teacher_guidance": None}
    prompt = build_student_prompt(state, GuidanceConfig(level=3), force_finish=False)
    assert "Format example only" in prompt
    # the example is a placeholder, not dataset content, and never the gold answer
    assert "Delhi" not in prompt
    assert "<your search terms>" in prompt


def test_validate_finish_requires_answer():
    ok, errors = validate_student_action({"action": {"tool": "finish", "params": {}}})
    assert not ok and any("finish_missing_answer" in e for e in errors)
    ok2, _ = validate_student_action({"action": {"tool": "finish", "params": {"answer": "Delhi"}}})
    assert ok2


def test_plan_prompts_are_budget_aware():
    from agentsim.teacher_guidance.prompts import build_initial_plan_prompt
    from agentsim.teacher_guidance.schemas import PlanReviewConfig
    cfg = PlanReviewConfig(enabled=True, max_initial_plan_steps=6)
    state = {"question": "q?", "budget": 5}
    prompt = build_initial_plan_prompt(state, cfg)
    assert "budget of 5 tool-use steps" in prompt
    # step cap is min(max_initial_plan_steps=6, budget=5) = 5
    assert "at most 5 steps" in prompt
    # without a budget, falls back to the configured max
    no_budget = build_initial_plan_prompt({"question": "q?", "budget": 0}, cfg)
    assert "at most 6 steps" in no_budget
    assert "budget of" not in no_budget
