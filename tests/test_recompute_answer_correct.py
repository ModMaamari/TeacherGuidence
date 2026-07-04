"""Tests for the answer_correct migration script's pure logic."""

import json

from scripts.recompute_answer_correct import recompute_episode, process_file


def test_recompute_episode_flips_stale_false_positive():
    episode = {
        "final_answer": "Kelly Osbourne, a British singer-songwriter, hosted the show.",
        "gold_answer": "Kelly Lee Osbourne",
        "final_metrics": {"answer_correct": False, "exact_match": False, "f1": 0.25},
    }
    new_episode, changed, regressed = recompute_episode(episode)
    assert changed is True
    assert regressed is False
    assert new_episode["final_metrics"]["answer_correct"] is True


def test_recompute_episode_leaves_already_correct_untouched():
    episode = {
        "final_answer": "Delhi",
        "gold_answer": "Delhi",
        "final_metrics": {"answer_correct": True, "exact_match": True, "f1": 1.0},
    }
    _, changed, regressed = recompute_episode(episode)
    assert changed is False
    assert regressed is False


def test_recompute_episode_leaves_genuinely_wrong_untouched():
    episode = {
        "final_answer": "Mumbai",
        "gold_answer": "Delhi",
        "final_metrics": {"answer_correct": False, "exact_match": False, "f1": 0.0},
    }
    _, changed, regressed = recompute_episode(episode)
    assert changed is False
    assert regressed is False


def test_recompute_episode_skips_records_missing_final_metrics():
    episode = {"final_answer": "x", "gold_answer": "y"}
    new_episode, changed, regressed = recompute_episode(episode)
    assert changed is False
    assert regressed is False
    assert new_episode == episode


def test_process_file_rewrites_only_changed_records(tmp_path):
    path = tmp_path / "teacher_guidance_episodes.jsonl"
    rows = [
        {"final_answer": "Kelly Osbourne, a British singer-songwriter, hosted the show.",
         "gold_answer": "Kelly Lee Osbourne", "final_metrics": {"answer_correct": False}},
        {"final_answer": "Mumbai", "gold_answer": "Delhi",
         "final_metrics": {"answer_correct": False}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    num_records, num_changed, num_regressed = process_file(path)
    assert (num_records, num_changed, num_regressed) == (2, 1, 0)

    updated = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert updated[0]["final_metrics"]["answer_correct"] is True
    assert updated[1]["final_metrics"]["answer_correct"] is False


def test_process_file_dry_run_does_not_write(tmp_path):
    path = tmp_path / "teacher_guidance_episodes.jsonl"
    row = {"final_answer": "Kelly Osbourne, a British singer-songwriter, hosted the show.",
           "gold_answer": "Kelly Lee Osbourne", "final_metrics": {"answer_correct": False}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    num_records, num_changed, num_regressed = process_file(path, dry_run=True)
    assert (num_records, num_changed, num_regressed) == (1, 1, 0)

    unchanged = json.loads(path.read_text(encoding="utf-8"))
    assert unchanged["final_metrics"]["answer_correct"] is False
