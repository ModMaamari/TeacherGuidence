"""Tests for formal plan validation and adherence tracking."""

import asyncio
import json

from agentsim.teacher_guidance.plan_execution import validate_formal_plan, PlanTracker
from agentsim.workflow.context import WorkflowContext
from agentsim.components.control.teacher_guided_agent_step import TeacherGuidedAgentStep


def test_validate_formal_plan():
    ok, errors = validate_formal_plan({"steps": [
        {"step_id": 1, "goal": "find", "intended_tool": "search"},
        {"step_id": 2, "goal": "pull", "intended_tool": "extract"},
    ]})
    assert ok and not errors

    bad, errors = validate_formal_plan({"steps": [{"step_id": 1, "goal": "", "intended_tool": "google"}]})
    assert not bad
    assert any("invalid_tool" in e for e in errors)
    assert any("missing_goal" in e for e in errors)

    assert validate_formal_plan({})[0] is False


def test_plan_tracker_records_adherence():
    tracker = PlanTracker({"steps": [
        {"intended_tool": "search"}, {"intended_tool": "extract"}, {"intended_tool": "finish"},
    ]})
    r1 = tracker.record("search")
    assert r1["expected_tool"] == "search" and r1["plan_step_followed"] is True
    r2 = tracker.record("verify")  # deviation
    assert r2["expected_tool"] == "extract" and r2["plan_step_followed"] is False
    r3 = tracker.record("finish")
    assert r3["plan_step_followed"] is True
    assert tracker.adherence() == round(2 / 3, 4)


CORPUS = [
    {"doc_id": "q1::doc0", "qid": "q1", "title": "The Oberoi Group",
     "text": "The Oberoi Group is headquartered in Delhi.",
     "sentences": ["The Oberoi Group is headquartered in Delhi."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
]


class StubLLM:
    def __init__(self, student_json, teacher_json):
        self.student_json, self.teacher_json = student_json, teacher_json

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        if "teacher evaluating" in prompt:
            return self.teacher_json
        # expose that the prompt carried the expected plan step
        StubLLM.last_student_prompt = prompt
        return self.student_json


def _context(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in CORPUS:
            f.write(json.dumps(row) + "\n")
    return WorkflowContext(
        task_id="q1", query="Where is the Oberoi Group headquartered?",
        metadata={
            "corpus_path": str(corpus_path), "student_model": "s", "teacher_model": "t",
            "guidance": {"level": 3, "leak_policy": "strict"},
            "plan_review_config": {"enabled": True, "formal_plan": True},
            "revised_plan": {"steps": [
                {"step_id": 1, "goal": "find HQ", "intended_tool": "search"},
                {"step_id": 2, "goal": "extract", "intended_tool": "extract"},
            ]},
            "gold": {"answer": "Delhi", "supporting_titles": [], "supporting_facts": [], "gold_doc_ids": ["q1::doc0"]},
            "retrieval_scope": {"qid": "q1", "candidate_doc_ids": ["q1::doc0"]},
        },
    )


def test_formal_plan_adherence_in_step(tmp_path):
    student = json.dumps({"thought": "search", "decision": {"category": "need_retrieval"},
                          "action": {"tool": "search", "params": {"query": "Oberoi HQ", "k": 2}},
                          "new_facts_extracted": []})
    teacher = json.dumps({"guidance_level": 3, "student_visible": {"score_continuous": 0.7, "feedback": "ok"},
                          "private_diagnosis": {}, "teacher_decision": "continue"})
    ctx = _context(tmp_path)
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=StubLLM(student, teacher))
    asyncio.run(comp.execute(ctx))

    m = ctx.metadata["teacher_guided_steps"][0]["metrics"]
    assert m["expected_tool"] == "search"
    assert m["plan_step_followed"] is True
    assert ctx.metadata["plan_adherence"] == 1.0
    # the expected plan step was surfaced in the student prompt
    assert "expected to execute now" in StubLLM.last_student_prompt
