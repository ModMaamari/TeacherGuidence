"""Tests for the pure aggregation functions in build_experiment_matrix_report.py.

Plotting and report file I/O are excluded (not meaningfully unit-testable / no need to
assert on matplotlib output).
"""

from scripts.build_experiment_matrix_report import (
    aggregate_episodes,
    step_count_distribution,
)


def _episode(steps, final_metrics=None, plan_review=None):
    return {
        "qid": "q1",
        "final_metrics": final_metrics or {},
        "steps": steps,
        "plan_review": plan_review or {},
    }


def _step(json_valid=True, leaked=False, student_calls=None, teacher_calls=None):
    return {
        "metrics": {"json_valid": json_valid},
        "leakage_check": {"gold_answer_leaked": leaked},
        "student_calls": student_calls or [],
        "teacher_calls": teacher_calls or [],
    }


def test_aggregate_episodes_empty_returns_zero_episodes():
    assert aggregate_episodes([]) == {"num_episodes": 0}


def test_aggregate_episodes_basic_metrics():
    episodes = [
        _episode(
            steps=[_step(), _step()],
            final_metrics={"exact_match": True, "answer_correct": True, "f1": 1.0, "supporting_doc_recall": 1.0},
        ),
        _episode(
            steps=[_step(), _step(), _step()],
            final_metrics={"exact_match": False, "answer_correct": False, "f1": 0.5, "supporting_doc_recall": 0.5},
        ),
    ]
    m = aggregate_episodes(episodes)
    assert m["num_episodes"] == 2
    assert m["exact_match"] == 0.5
    assert m["answer_correct"] == 0.5
    assert m["f1"] == 0.75
    assert m["supporting_doc_recall"] == 0.75
    assert m["avg_steps"] == 2.5
    assert m["median_steps"] == 2.5
    assert m["min_steps"] == 2
    assert m["max_steps"] == 3


def test_aggregate_episodes_invalid_json_and_leakage_rates():
    episodes = [
        _episode(steps=[_step(json_valid=False), _step(json_valid=True)]),
        _episode(steps=[_step(json_valid=True), _step(leaked=True)]),
    ]
    m = aggregate_episodes(episodes)
    assert m["invalid_json_rate"] == 0.25  # 1 of 4 steps invalid
    assert m["leakage_rate"] == 0.5  # 1 of 2 episodes leaked


def test_aggregate_episodes_sums_teacher_calls_tokens_and_cost():
    call_a = {"usage": {"total_tokens": 100, "cost": 0.01}}
    call_b = {"usage": {"total_tokens": 50, "cost": None}}  # ollama: no cost
    episodes = [
        _episode(
            steps=[_step(student_calls=[call_b], teacher_calls=[call_a])],
            plan_review={"initial_plan_calls": [call_b], "rounds": [{"review_calls": [call_a]}]},
        )
    ]
    m = aggregate_episodes(episodes)
    # teacher calls: 1 (step) + 1 (round review) = 2
    assert m["total_teacher_calls"] == 2
    # tokens: step student(50) + step teacher(100) + initial_plan(50) + review(100) = 300
    assert m["total_tokens"] == 300
    # cost: only the two call_a entries have a real cost (0.01 each)
    assert m["total_cost_usd"] == 0.02


def test_aggregate_episodes_uses_exact_match_fallback_when_answer_correct_absent():
    episodes = [_episode(steps=[], final_metrics={"exact_match": True})]
    m = aggregate_episodes(episodes)
    assert m["answer_correct"] == 1.0


def test_step_count_distribution():
    episodes = [_episode(steps=[_step()] * n) for n in (2, 5, 5, 20)]
    dist = step_count_distribution(episodes)
    assert dist["min"] == 2
    assert dist["max"] == 20
    assert dist["mean"] == 8.0
    assert dist["median"] == 5.0


def test_step_count_distribution_empty():
    assert step_count_distribution([]) == {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
