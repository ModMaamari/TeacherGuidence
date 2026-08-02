"""Tests for the pre-collection smoke gates.

These gates exist to stop a broken dataset or a broken run *before* the expensive
generation step, so the thing that actually has to be tested is that they FAIL when
something is wrong -- a preflight that always passes is worse than none.
"""

from __future__ import annotations

import json

import pytest

from agentsim.teacher_guidance.converters import convert_dataset
from scripts.smoke_datasets import audit_episodes, find_dataset_pairs, preflight

HOTPOT = {
    "id": "hp1",
    "question": "Where is the Oberoi Group head office?",
    "answer": "Delhi",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": {"title": ["Oberoi family", "The Oberoi Group"], "sent_id": [0, 0]},
    "context": {
        "title": ["Oberoi family", "The Oberoi Group", "Distractor"],
        "sentences": [
            ["The Oberoi family runs hotels."],
            ["The Oberoi Group is headquartered in Delhi.", "It is large."],
            ["Unrelated text about penguins."],
        ],
    },
}


def _write_dataset(tmp_path, questions, corpus, name="hotpotqa_validation"):
    q_path = tmp_path / f"{name}_questions.jsonl"
    c_path = tmp_path / f"{name}_corpus.jsonl"
    q_path.write_text("\n".join(json.dumps(r) for r in questions) + "\n", encoding="utf-8")
    c_path.write_text("\n".join(json.dumps(r) for r in corpus) + "\n", encoding="utf-8")
    return q_path, c_path


def _good_dataset(tmp_path):
    questions, corpus, _ = convert_dataset("hotpotqa", [HOTPOT], "validation", strict=True)
    return questions, corpus, _write_dataset(tmp_path, questions, corpus)


def test_preflight_passes_on_a_well_formed_dataset(tmp_path):
    _, _, (q_path, c_path) = _good_dataset(tmp_path)
    problems, stats = preflight("hotpotqa_validation", q_path, c_path)
    assert problems == []
    assert stats["questions"] == 1
    assert stats["gold_granularity"] == {"sentence": 1}
    assert stats["retrieval_checked"] == 1


def test_preflight_catches_missing_corpus_documents(tmp_path):
    questions, corpus, _ = convert_dataset("hotpotqa", [HOTPOT], "validation", strict=True)
    q_path, c_path = _write_dataset(tmp_path, questions, corpus[:1])  # drop gold doc 1
    problems, _ = preflight("hotpotqa_validation", q_path, c_path)
    assert problems, "dropping a referenced document must be caught"


def test_preflight_catches_duplicate_qids(tmp_path):
    questions, corpus, _ = convert_dataset("hotpotqa", [HOTPOT], "validation", strict=True)
    q_path, c_path = _write_dataset(tmp_path, questions * 2, corpus)
    problems, _ = preflight("hotpotqa_validation", q_path, c_path)
    assert any("duplicate qid" in p for p in problems)


def test_preflight_catches_unreachable_gold_documents(tmp_path):
    """Schema-valid rows whose corpus is filed under the wrong qid: the retriever groups
    by qid, so the gold documents become unreachable at generation time."""
    questions, corpus, _ = convert_dataset("hotpotqa", [HOTPOT], "validation", strict=True)
    broken = [dict(d, qid="other-question") for d in corpus]
    q_path, c_path = _write_dataset(tmp_path, questions, broken)
    problems, _ = preflight("hotpotqa_validation", q_path, c_path)
    assert problems


def test_find_dataset_pairs_discovers_prepared_files(tmp_path):
    _good_dataset(tmp_path)
    (tmp_path / "stray_questions.jsonl").write_text("{}\n", encoding="utf-8")  # no corpus
    pairs = find_dataset_pairs(tmp_path)
    assert [p[0] for p in pairs] == ["hotpotqa_validation"]


# --- online audit ------------------------------------------------------------------
def _episode(**over):
    ep = {
        "qid": "hp1",
        "dataset": "hotpotqa",
        "schema_version": "1.0",
        "framework_commit": "abc123",
        "config_hash": "cfg1",
        "generated_at": "2026-08-02T00:00:00+00:00",
        "teacher_models_used": ["fau/gpt-oss-120b"],
        "final_metrics": {"answer_correct": True},
        "steps": [
            {
                "t": 1,
                "metrics": {"json_valid": True, "invalid_action": False},
                "leakage_check": {"gold_answer_leaked": False},
            }
        ],
    }
    ep.update(over)
    return ep


def _write_episodes(tmp_path, episodes):
    run = tmp_path / "run" / "hotpot_questions" / "sample_0001"
    run.mkdir(parents=True)
    (run / "teacher_guidance_episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_audit_passes_on_healthy_episodes(tmp_path):
    root = _write_episodes(tmp_path, [_episode()])
    problems, stats = audit_episodes(root, "hotpotqa", 1)
    assert problems == []
    assert stats["episodes"] == 1 and stats["answer_correct"] == 1


def test_audit_catches_wrong_dataset_label(tmp_path):
    """The exact failure the provenance fix was for: traces silently labelled hotpotqa
    while actually generated from another source."""
    root = _write_episodes(tmp_path, [_episode(dataset="musique")])
    problems, _ = audit_episodes(root, "hotpotqa", 1)
    assert any("provenance is wrong" in p for p in problems)


@pytest.mark.parametrize("field", ["schema_version", "framework_commit", "config_hash"])
def test_audit_catches_missing_provenance(tmp_path, field):
    root = _write_episodes(tmp_path, [_episode(**{field: ""})])
    problems, _ = audit_episodes(root, "hotpotqa", 1)
    assert any(field in p for p in problems)


def test_audit_catches_leaked_gold_answer(tmp_path):
    ep = _episode()
    ep["steps"][0]["leakage_check"] = {"gold_answer_leaked": True}
    root = _write_episodes(tmp_path, [ep])
    problems, _ = audit_episodes(root, "hotpotqa", 1)
    assert any("leaked" in p for p in problems)


def test_audit_catches_missing_teacher(tmp_path):
    root = _write_episodes(tmp_path, [_episode(teacher_models_used=[])])
    problems, _ = audit_episodes(root, "hotpotqa", 1)
    assert any("no teacher call" in p for p in problems)


def test_audit_allows_missing_teacher_when_deliberately_skipped(tmp_path):
    """skip_teacher runs are the no-guidance control arm -- absence of a teacher is
    expected there and must not be reported as a fault."""
    ep = _episode(teacher_models_used=[])
    ep["steps"][0]["teacher_skipped"] = True
    root = _write_episodes(tmp_path, [ep])
    problems, _ = audit_episodes(root, "hotpotqa", 1)
    assert problems == []


def test_audit_catches_high_parse_failure_rate(tmp_path):
    ep = _episode()
    ep["steps"] = [
        {"t": i, "metrics": {"json_valid": False}, "leakage_check": {}} for i in range(4)
    ]
    root = _write_episodes(tmp_path, [ep])
    problems, stats = audit_episodes(root, "hotpotqa", 1)
    assert any("failed to parse" in p for p in problems)
    assert stats["step_parse_failure_rate"] == 1.0


def test_audit_catches_short_run(tmp_path):
    root = _write_episodes(tmp_path, [_episode()])
    problems, _ = audit_episodes(root, "hotpotqa", 5)
    assert any("1/5 episodes" in p for p in problems)


def test_audit_reports_no_episodes(tmp_path):
    problems, _ = audit_episodes(tmp_path, "hotpotqa", 3)
    assert any("no episodes produced" in p for p in problems)
