"""Tests for the teacher_guided_plan_review component."""

import asyncio
import json

from agentsim.components.base import ComponentRegistry
from agentsim.workflow.context import WorkflowContext
from agentsim.components.control.teacher_guided_plan_review import TeacherGuidedPlanReview


def _context(plan_review_config):
    return WorkflowContext(
        task_id="q1",
        query="Where is the Oberoi Group headquartered?",
        metadata={
            "student_model": "stub",
            "teacher_model": "stub",
            "guidance": {"level": 3, "max_feedback_words": 40, "leak_policy": "strict"},
            "plan_review_config": plan_review_config,
            "gold": {
                "answer": "Delhi",
                "supporting_titles": ["The Oberoi Group"],
                "supporting_facts": [{"title": "The Oberoi Group", "sent_id": 0}],
                "gold_doc_ids": ["q1::doc0"],
            },
            "retrieval_scope": {"qid": "q1", "candidate_doc_ids": ["q1::doc0"]},
        },
    )


class StubLLM:
    def __init__(self, initial, review, revised):
        self.initial, self.review, self.revised = initial, review, revised
        self.calls = 0

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        self.calls += 1
        if "teacher reviewing" in prompt:
            return self.review
        if "Revise your plan" in prompt:
            return self.revised
        return self.initial


def test_registered():
    assert "teacher_guided_plan_review" in ComponentRegistry.list_components()


def test_disabled_short_circuits():
    ctx = _context({"enabled": False})
    comp = TeacherGuidedPlanReview(config={}, llm_client=None)
    result = asyncio.run(comp.execute(ctx))
    assert result.data["verdict"] == "PROCEED"
    assert ctx.metadata["plan_review"] == {"enabled": False}
    assert "revised_plan" not in ctx.metadata


def test_enabled_flow_records_and_sanitizes():
    initial = json.dumps({
        "plan_summary": "search then answer",
        "steps": [{"step_id": 1, "goal": "find HQ", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [], "stop_condition": "have HQ",
    })
    # Teacher tries to leak the gold answer.
    review = json.dumps({
        "plan_review_enabled": True, "review_guidance_level": 3,
        "student_visible": {"score_continuous": 0.5, "feedback": "Add verification; answer is Delhi."},
        "private_diagnosis": {"premature_answering_risk": True},
        "teacher_decision": "revise_plan",
    })
    revised = json.dumps({
        "revision_summary": "added verify",
        "plan_summary": "search, verify, answer",
        "steps": [
            {"step_id": 1, "goal": "find HQ", "intended_tool": "search", "rationale": "x", "depends_on": []},
            {"step_id": 2, "goal": "verify", "intended_tool": "verify", "rationale": "y", "depends_on": [1]},
        ],
        "teacher_feedback_used": ["added verification"], "stop_condition": "verified HQ",
    })
    ctx = _context({"enabled": True, "review_guidance_level": 3})
    comp = TeacherGuidedPlanReview(config={}, llm_client=StubLLM(initial, review, revised))
    result = asyncio.run(comp.execute(ctx))

    assert result.data["verdict"] == "PROCEED"
    record = ctx.metadata["plan_review"]
    assert record["enabled"] is True
    # gold answer sanitized out of student-visible plan feedback
    assert "Delhi" not in record["student_visible_plan_feedback"]["feedback"]
    assert record["leakage_check"]["gold_answer_leaked"] is True
    # revised plan stored and added verification
    assert ctx.metadata["revised_plan"]["steps"][1]["intended_tool"] == "verify"
    assert record["metrics"]["revised_covers_verification"] is True
