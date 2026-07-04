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
            "budget": 5,
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
                    "student_calls": [{"attempt": 1, "started_at": "t0", "ended_at": "t1",
                                        "elapsed_ms": 123.4, "prompt": "STUDENT PROMPT (no gold here)",
                                        "response_text": '{"action": {"tool": "search"}}', "raw_response": {"id": "s1"}}],
                    "student_call_ms": 123.4,
                    "student_action": {"action": {"tool": "search"}},
                    "tool_observation": {"tool": "search", "status": "ok"},
                    "teacher_prompt": "TEACHER PROMPT with gold Delhi",
                    "teacher_raw": '{"teacher_decision": "continue"}',
                    "teacher_calls": [{"attempt": 1, "started_at": "t0", "ended_at": "t1",
                                        "elapsed_ms": 567.8, "prompt": "TEACHER PROMPT with gold Delhi",
                                        "response_text": '{"teacher_decision": "continue"}', "raw_response": {"id": "t1"}}],
                    "teacher_call_ms": 567.8,
                    "teacher_full": {"private_diagnosis": {"main_error": "none"}, "teacher_decision": "continue"},
                    "student_visible_guidance": {"score": 0.7, "feedback": "Good."},
                    "leakage_check": {"gold_answer_leaked": False},
                    "metrics": {"json_valid": True},
                    "stop_condition": "CONTINUE",
                    "step_started_at": "2026-07-01T09:00:00+00:00",
                    "step_ended_at": "2026-07-01T09:00:01+00:00",
                    "step_elapsed_ms": 999.9,
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
    # budget (5) and used_steps (1 step in the synthetic context) are distinct
    assert episode["budget"] == 5
    assert episode["used_steps"] == 1


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


def test_exported_step_keeps_raw_prompts_and_timing(tmp_path):
    # student_prompt/student_raw/teacher_prompt/teacher_raw and the *_ms timing fields
    # exist on the in-memory step record but were previously dropped when building the
    # exported episode row (the file the Trajectory Explorer viewer actually reads).
    episode = TeacherGuidanceEpisodeExporter().export_episode(_context(), str(tmp_path))
    step = episode["steps"][0]
    assert step["student_prompt"] == "STUDENT PROMPT (no gold here)"
    assert step["student_raw"] == '{"action": {"tool": "search"}}'
    assert step["student_call_ms"] == 123.4
    assert step["teacher_prompt"] == "TEACHER PROMPT with gold Delhi"
    assert step["teacher_raw"] == '{"teacher_decision": "continue"}'
    assert step["teacher_call_ms"] == 567.8
    assert step["step_started_at"] == "2026-07-01T09:00:00+00:00"
    assert step["step_ended_at"] == "2026-07-01T09:00:01+00:00"
    assert step["step_elapsed_ms"] == 999.9
    assert step["student_calls"][0]["raw_response"] == {"id": "s1"}
    assert step["teacher_calls"][0]["raw_response"] == {"id": "t1"}

    # And the whole row round-trips through JSON, matching what the viewer's /api/episode
    # endpoint would actually serve.
    written = (tmp_path / "teacher_guidance_episodes.jsonl").read_text(encoding="utf-8").strip()
    row = json.loads(written)
    assert row["steps"][0]["teacher_prompt"] == "TEACHER PROMPT with gold Delhi"
    assert row["steps"][0]["teacher_calls"][0]["raw_response"] == {"id": "t1"}


def test_final_metrics_computed_without_retriever(tmp_path):
    episode = TeacherGuidanceEpisodeExporter().export_episode(_context(), str(tmp_path))
    # EM should be true (final answer "Delhi" vs gold "Delhi"); doc recall 1.0
    assert episode["final_metrics"]["exact_match"] is True
    assert episode["final_metrics"]["supporting_doc_recall"] == 1.0


def test_final_metrics_include_teacher_judgment_when_present(tmp_path):
    ctx = _context()
    ctx.metadata["teacher_final_judgment"] = {"correct": 1, "score": 0.9}
    fm = TeacherGuidanceEpisodeExporter().export_episode(ctx, str(tmp_path))["final_metrics"]
    assert fm["teacher_answer_correct"] == 1
    assert fm["teacher_answer_score"] == 0.9


def test_final_metrics_teacher_judgment_none_when_absent(tmp_path):
    fm = TeacherGuidanceEpisodeExporter().export_episode(_context(), str(tmp_path))["final_metrics"]
    assert fm["teacher_answer_correct"] is None
    assert fm["teacher_answer_score"] is None
