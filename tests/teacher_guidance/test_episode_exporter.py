"""Tests for the Teacher Guidance episode exporter."""

import json

from agentsim.workflow.context import WorkflowContext
from agentsim.teacher_guidance.episode_exporter import TeacherGuidanceEpisodeExporter


def _context():
    ctx = WorkflowContext(
        task_id="q1",
        query="Where is the Oberoi Group headquartered?",
        metadata={
            "sample_id": "sample_001",
            "student_model": "student",
            "teacher_model": "teacher",
            "guidance": {"level": 3},
            "gold": {
                "qid": "q1",
                "answer": "Delhi",
                "supporting_titles": ["The Oberoi Group"],
                "supporting_facts": [{"title": "The Oberoi Group", "sent_id": 0}],
                "gold_doc_ids": ["q1::doc0"],
            },
            "retrieval_scope": {"qid": "q1"},
            "retrieved_doc_ids": ["q1::doc0"],
            "extracted_facts": [{"doc_id": "q1::doc0", "span": "headquartered in Delhi", "fact": "HQ Delhi"}],
            "final_answer": "Delhi",
            "stop_reason": "teacher_accept",
            "teacher_guided_steps": [
                {
                    "t": 1,
                    "student_prompt": "STUDENT PROMPT (no gold here)",
                    "student_raw": '{"action": {"tool": "search"}}',
                    "student_action": {"action": {"tool": "search"}},
                    "tool_observation": {"tool": "search", "status": "ok"},
                    "teacher_prompt": "TEACHER PROMPT with gold Delhi",
                    "teacher_raw": '{"teacher_decision": "continue"}',
                    "teacher_full": {"private_diagnosis": {"main_error": "none"}, "teacher_decision": "continue"},
                    "student_visible_guidance": {"score": 0.7, "feedback": "Good."},
                    "leakage_check": {"gold_answer_leaked": False},
                    "metrics": {"json_valid": True},
                    "stop_condition": "CONTINUE",
                }
            ],
        },
    )
    return ctx


def test_export_writes_all_files(tmp_path):
    exporter = TeacherGuidanceEpisodeExporter()
    episode = exporter.export_episode(_context(), str(tmp_path))

    for name in [
        "teacher_guidance_episodes.jsonl",
        "student_sft.jsonl",
        "teacher_sft.jsonl",
        "student_visible_guidance.jsonl",
        "teacher_guidance_metrics.json",
    ]:
        assert (tmp_path / name).exists(), name

    assert episode["final_answer"] == "Delhi"
    assert episode["stop_reason"] == "teacher_accept"


def test_one_episode_and_one_sft_row_per_step(tmp_path):
    TeacherGuidanceEpisodeExporter().export_episode(_context(), str(tmp_path))
    episodes = (tmp_path / "teacher_guidance_episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    student_rows = (tmp_path / "student_sft.jsonl").read_text(encoding="utf-8").strip().splitlines()
    teacher_rows = (tmp_path / "teacher_sft.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(episodes) == 1
    assert len(student_rows) == 1
    assert len(teacher_rows) == 1


def test_student_sft_input_has_no_gold_or_diagnosis(tmp_path):
    TeacherGuidanceEpisodeExporter().export_episode(_context(), str(tmp_path))
    student_rows = (tmp_path / "student_sft.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(student_rows[0])
    assert "Delhi" not in row["input"]
    assert row["metadata"]["gold_answer_hidden"] is True


def test_final_metrics_computed_without_retriever(tmp_path):
    episode = TeacherGuidanceEpisodeExporter().export_episode(_context(), str(tmp_path))
    # EM should be true (final answer "Delhi" vs gold "Delhi"); doc recall 1.0
    assert episode["final_metrics"]["exact_match"] is True
    assert episode["final_metrics"]["supporting_doc_recall"] == 1.0
