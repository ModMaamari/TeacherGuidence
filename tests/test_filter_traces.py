"""Tests for the pure partitioning logic of the trace-filtering CLI."""

from agentsim.teacher_guidance.trace_quality import TraceCriteria
from scripts.filter_traces import partition_episodes


def _ep(qid, correct=True, grounded=True, stop="teacher_accept", eff=1.0, wasted=0, leaked=False):
    return {
        "qid": qid,
        "final_metrics": {"answer_correct": correct, "answer_grounded": grounded},
        "path_optimality": {"step_efficiency": eff, "wasted_step_count": wasted},
        "stop_reason": stop,
        "steps": [{"metrics": {"json_valid": True}, "leakage_check": {"gold_answer_leaked": leaked}}],
    }


def test_partition_keeps_only_good_traces():
    episodes = [
        _ep("q1"),                       # good
        _ep("q2", correct=False),        # incorrect
        _ep("q3", grounded=False),       # ungrounded
        _ep("q4", stop="budget_forced_finish"),  # forced
        _ep("q5"),                       # good
    ]
    accepted, num_rejected, reasons = partition_episodes(episodes, TraceCriteria())
    assert [e["qid"] for e in accepted] == ["q1", "q5"]
    assert num_rejected == 3
    assert reasons["incorrect"] == 1
    assert reasons["ungrounded"] == 1
    assert reasons["not_natural_finish"] == 1


def test_partition_respects_relaxed_criteria():
    episodes = [_ep("q1", stop="budget_forced_finish")]
    accepted, num_rejected, _ = partition_episodes(
        episodes, TraceCriteria(require_natural_finish=False)
    )
    assert len(accepted) == 1
    assert num_rejected == 0


def test_partition_empty():
    accepted, num_rejected, reasons = partition_episodes([], TraceCriteria())
    assert accepted == [] and num_rejected == 0 and reasons == {}
