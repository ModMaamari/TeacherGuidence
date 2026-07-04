"""Tests for the trace-quality acceptance gate."""

from agentsim.teacher_guidance.trace_quality import TraceCriteria, evaluate_trace


def _good_episode(**over):
    ep = {
        "final_metrics": {"answer_correct": True, "answer_grounded": True},
        "path_optimality": {"step_efficiency": 1.0, "wasted_step_count": 0},
        "stop_reason": "teacher_accept",
        "steps": [
            {"metrics": {"json_valid": True}, "leakage_check": {"gold_answer_leaked": False}},
            {"metrics": {"json_valid": True}, "leakage_check": {"gold_answer_leaked": False}},
        ],
    }
    ep.update(over)
    return ep


def test_clean_correct_grounded_trace_accepted():
    v = evaluate_trace(_good_episode())
    assert v["accepted"] is True
    assert v["reasons"] == []


def test_incorrect_rejected():
    ep = _good_episode(final_metrics={"answer_correct": False, "answer_grounded": True})
    v = evaluate_trace(ep)
    assert v["accepted"] is False
    assert "incorrect" in v["reasons"]


def test_ungrounded_rejected():
    ep = _good_episode(final_metrics={"answer_correct": True, "answer_grounded": False})
    assert "ungrounded" in evaluate_trace(ep)["reasons"]


def test_gold_answer_leak_rejected():
    ep = _good_episode(steps=[
        {"metrics": {"json_valid": True}, "leakage_check": {"gold_answer_leaked": True}},
    ])
    assert "gold_answer_leaked" in evaluate_trace(ep)["reasons"]


def test_invalid_step_rejected():
    ep = _good_episode(steps=[
        {"metrics": {"json_valid": False}, "leakage_check": {}},
    ])
    assert "invalid_step" in evaluate_trace(ep)["reasons"]


def test_forced_finish_rejected_by_default():
    ep = _good_episode(stop_reason="budget_forced_finish")
    assert "not_natural_finish" in evaluate_trace(ep)["reasons"]


def test_forced_finish_allowed_when_configured():
    ep = _good_episode(stop_reason="budget_forced_finish")
    v = evaluate_trace(ep, TraceCriteria(require_natural_finish=False))
    assert v["accepted"] is True


def test_low_efficiency_and_wasted_steps_rejected():
    ep = _good_episode(path_optimality={"step_efficiency": 0.3, "wasted_step_count": 3})
    reasons = evaluate_trace(ep)["reasons"]
    assert "low_efficiency" in reasons
    assert "too_many_wasted_steps" in reasons


def test_multiple_reasons_collected():
    ep = _good_episode(
        final_metrics={"answer_correct": False, "answer_grounded": False},
        stop_reason="error",
    )
    reasons = set(evaluate_trace(ep)["reasons"])
    assert {"incorrect", "ungrounded", "not_natural_finish"} <= reasons
