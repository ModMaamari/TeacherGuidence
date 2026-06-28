"""Tests for the trajectory explorer data-access layer."""

import json

import pytest

from agentsim.teacher_guidance.viewer import data_access as da


def _episode(qid, em, f1, answer, steps=2):
    return {
        "episode_id": f"sample_{qid}",
        "qid": qid,
        "query": f"question {qid}?",
        "gold_answer": "Delhi",
        "final_answer": answer,
        "guidance_level": 3,
        "student_model": "custom/qwen/qwen3.7-plus",
        "teacher_model": "custom/z-ai/glm-5.2",
        "plan_review": {"enabled": True},
        "steps": [{"t": i + 1, "student_action": {"action": {"tool": "search"}}} for i in range(steps)],
        "final_answer_dummy": None,
        "final_metrics": {"exact_match": em, "f1": f1, "supporting_doc_recall": 1.0},
        "stop_reason": "teacher_accept" if em else "budget_forced_finish",
    }


@pytest.fixture
def output_root(tmp_path):
    run = tmp_path / "sim_x" / "run123" / "hotpot_questions"
    for i, ep in enumerate([_episode("q1", True, 1.0, "Delhi"), _episode("q2", False, 0.0, "Mumbai")], start=1):
        sd = run / f"sample_{i:03d}"
        sd.mkdir(parents=True)
        (sd / da.EPISODE_FILENAME).write_text(json.dumps(ep) + "\n", encoding="utf-8")
    return tmp_path


def test_find_runs(output_root):
    runs = da.find_runs(output_root)
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == "sim_x/run123"
    assert r["num_episodes"] == 2
    assert r["guidance_level"] == 3
    assert r["mean_exact_match"] == 0.5
    assert r["mean_f1"] == 0.5
    assert r["stop_reasons"] == {"teacher_accept": 1, "budget_forced_finish": 1}


def test_get_run_episodes(output_root):
    eps = da.get_run_episodes(output_root, "sim_x/run123")
    assert {e["qid"] for e in eps} == {"q1", "q2"}
    q1 = next(e for e in eps if e["qid"] == "q1")
    assert q1["exact_match"] is True
    assert q1["num_steps"] == 2
    assert q1["plan_review_enabled"] is True


def test_get_episode_full(output_root):
    ep = da.get_episode(output_root, "sim_x/run123", "q2")
    assert ep is not None
    assert ep["final_answer"] == "Mumbai"
    assert ep["final_metrics"]["exact_match"] is False
    assert da.get_episode(output_root, "sim_x/run123", "missing") is None


def test_traversal_is_rejected(output_root):
    # run_id escaping the output root must resolve to nothing
    assert da.get_run_episodes(output_root, "../../etc") == []
    assert da.get_episode(output_root, "..", "q1") is None


def test_empty_root(tmp_path):
    assert da.find_runs(tmp_path / "nope") == []
