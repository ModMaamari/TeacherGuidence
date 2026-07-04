"""Tests for deterministic path-optimality signals."""

from agentsim.teacher_guidance.optimality import compute_path_optimality


def _step(tool, params=None, status="ok", results=None):
    obs = {"tool": tool, "status": status}
    if results is not None:
        obs["data"] = {"results": [{"doc_id": d} for d in results]}
    return {
        "student_action": {"action": {"tool": tool, "params": params or {}}},
        "tool_observation": obs,
    }


def test_clean_path_is_fully_efficient():
    steps = [
        _step("search", {"query": "q1"}, results=["d1", "d2"]),
        _step("extract", {"doc_ids": ["d1"], "target_facts": ["f"]}),
        _step("finish", {"answer": "x"}),
    ]
    m = compute_path_optimality(steps)
    assert m["failed_tool_count"] == 0
    assert m["repeated_action_count"] == 0
    assert m["wasted_search_count"] == 0
    assert m["wasted_step_count"] == 0
    assert m["step_efficiency"] == 1.0


def test_failed_tool_counts_as_wasted():
    steps = [
        _step("search", {"query": "q1"}, results=["d1"]),
        _step("extract", {"doc_ids": ["d1"], "target_facts": ["nope"]}, status="error"),
        _step("finish", {"answer": "x"}),
    ]
    m = compute_path_optimality(steps)
    assert m["failed_tool_count"] == 1
    assert m["wasted_step_count"] == 1
    assert m["step_efficiency"] == round(1 - 1 / 3, 4)


def test_repeated_action_detected():
    steps = [
        _step("search", {"query": "same"}, results=["d1"]),
        _step("search", {"query": "same"}, results=["d1"]),  # identical + no new docs
        _step("finish", {"answer": "x"}),
    ]
    m = compute_path_optimality(steps)
    assert m["repeated_action_count"] == 1
    assert m["wasted_search_count"] == 1
    # The second search trips both flags but is a single wasted step.
    assert m["wasted_step_count"] == 1


def test_wasted_search_when_no_new_docs():
    steps = [
        _step("search", {"query": "q1"}, results=["d1", "d2"]),
        _step("search", {"query": "q2"}, results=["d1"]),  # different query, but nothing new
        _step("finish", {"answer": "x"}),
    ]
    m = compute_path_optimality(steps)
    assert m["wasted_search_count"] == 1
    assert m["repeated_action_count"] == 0  # different params, not a repeat
    assert m["wasted_step_count"] == 1


def test_finish_repeat_not_penalized():
    # A forced finish after an earlier finish attempt shouldn't count as a repeat.
    steps = [
        _step("finish", {"answer": "x"}),
        _step("finish", {"answer": "x"}),
    ]
    m = compute_path_optimality(steps)
    assert m["repeated_action_count"] == 0


def test_empty_episode():
    m = compute_path_optimality([])
    assert m["used_steps"] == 0
    assert m["step_efficiency"] == 0.0
