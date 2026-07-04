"""Tests for the question-level stratified split."""

from agentsim.teacher_guidance.dataset_split import question_stratum, split_by_question


def _ep(qid, answer):
    return {"qid": qid, "gold_answer": answer}


def test_question_stratum_classes():
    assert question_stratum(_ep("q", "yes")) == "boolean"
    assert question_stratum(_ep("q", "no")) == "boolean"
    assert question_stratum(_ep("q", "born in 1931")) == "numeric"
    assert question_stratum(_ep("q", "Kelly Osbourne")) == "entity"


def test_split_is_by_question_no_crossing():
    eps = [_ep(f"q{i}", "Entity Name") for i in range(20)]
    assignment = split_by_question(eps)
    # Every qid gets exactly one split.
    assert set(assignment.keys()) == {f"q{i}" for i in range(20)}
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_split_is_deterministic():
    eps = [_ep(f"q{i}", "Entity") for i in range(30)]
    assert split_by_question(eps, seed=7) == split_by_question(eps, seed=7)


def test_split_respects_ratios_per_stratum():
    eps = [_ep(f"e{i}", "Entity") for i in range(10)] + [_ep(f"b{i}", "yes") for i in range(10)]
    assignment = split_by_question(eps, ratios=(0.8, 0.1, 0.1))
    # 8/1/1 within each of the two strata.
    for prefix in ("e", "b"):
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(10):
            counts[assignment[f"{prefix}{i}"]] += 1
        assert counts["train"] == 8
        assert counts["val"] == 1
        assert counts["test"] == 1


def test_ratios_must_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        split_by_question([_ep("q", "x")], ratios=(0.5, 0.2, 0.2))
