"""Tests for the teacher_guided_plan_review component."""

import asyncio
import json

from agentsim.components.base import ComponentRegistry
from agentsim.workflow.context import WorkflowContext
from agentsim.components.control.teacher_guided_plan_review import (
    TeacherGuidedPlanReview,
    STUDENT_PLAN_SCHEMA,
    REVISED_STUDENT_PLAN_SCHEMA,
    TEACHER_PLAN_REVIEW_SCHEMA,
)
from agentsim.teacher_guidance.guidance_policy import FALLBACK_FEEDBACK


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


def test_skip_teacher_uses_initial_plan_with_zero_teacher_calls():
    initial = json.dumps({
        "plan_summary": "search then answer",
        "steps": [{"step_id": 1, "goal": "find HQ", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [], "stop_condition": "have HQ",
    })

    class NoTeacherCallStub:
        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "teacher reviewing" in prompt:
                raise AssertionError("teacher should never be called when skip_teacher is set")
            return initial

    ctx = _context({"enabled": True, "review_guidance_level": 3})
    ctx.metadata["skip_teacher"] = True
    comp = TeacherGuidedPlanReview(config={}, llm_client=NoTeacherCallStub())
    result = asyncio.run(comp.execute(ctx))

    assert result.data["verdict"] == "PROCEED"
    record = ctx.metadata["plan_review"]
    assert record["skip_teacher_review"] is True
    assert record["rounds"] == []
    assert record["initial_plan_calls"][0]["response_text"] == initial
    assert ctx.metadata["revised_plan"] == json.loads(initial)


class SequencedStub:
    """Returns the initial plan, then a sequence of teacher reviews, with revisions."""

    def __init__(self, initial, reviews, revised):
        self.initial = initial
        self.reviews = list(reviews)
        self.revised = revised
        self.review_i = 0
        self.calls = 0

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        self.calls += 1
        if "teacher reviewing" in prompt:
            r = self.reviews[min(self.review_i, len(self.reviews) - 1)]
            self.review_i += 1
            return r
        if "Revise your plan" in prompt:
            return self.revised
        return self.initial


def test_multistep_planning_loop_runs_until_accept():
    initial = json.dumps({
        "plan_summary": "v0", "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [], "stop_condition": "done",
    })
    revise = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 0.5, "feedback": "revise"},
                         "private_diagnosis": {}, "teacher_decision": "revise_plan"})
    accept = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 1.0, "feedback": "good"},
                         "private_diagnosis": {}, "teacher_decision": "accept_plan"})
    revised = json.dumps({"revision_summary": "r", "plan_summary": "v1",
                          "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search", "rationale": "x", "depends_on": []}],
                          "teacher_feedback_used": [], "stop_condition": "done"})

    ctx = _context({"enabled": True, "review_guidance_level": 3, "planning_steps": 3})
    # round 1 revise -> round 2 accept (stops before round 3)
    stub = SequencedStub(initial, [revise, accept], revised)
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    result = asyncio.run(comp.execute(ctx))

    record = ctx.metadata["plan_review"]
    assert record["num_planning_rounds"] == 2
    assert record["metrics"]["revisions_done"] == 1
    assert record["revision_skipped"] is False
    # calls: initial + (review+revise) + review(accept) = 4
    assert stub.calls == 4
    assert record["rounds"][-1]["accepted"] is True


def test_multistep_planning_loop_records_timing():
    initial = json.dumps({
        "plan_summary": "v0", "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [], "stop_condition": "done",
    })
    revise = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 0.5, "feedback": "revise"},
                         "private_diagnosis": {}, "teacher_decision": "revise_plan"})
    accept = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 1.0, "feedback": "good"},
                         "private_diagnosis": {}, "teacher_decision": "accept_plan"})
    revised = json.dumps({"revision_summary": "r", "plan_summary": "v1",
                          "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search", "rationale": "x", "depends_on": []}],
                          "teacher_feedback_used": [], "stop_condition": "done"})

    ctx = _context({"enabled": True, "review_guidance_level": 3, "planning_steps": 3})
    stub = SequencedStub(initial, [revise, accept], revised)
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    record = ctx.metadata["plan_review"]
    assert isinstance(record["initial_plan_call_ms"], float) and record["initial_plan_call_ms"] >= 0
    assert isinstance(record["plan_review_elapsed_ms"], float) and record["plan_review_elapsed_ms"] >= 0
    round1, round2 = record["rounds"]
    assert isinstance(round1["review_call_ms"], float) and round1["review_call_ms"] >= 0
    assert isinstance(round1["revision_call_ms"], float) and round1["revision_call_ms"] >= 0
    assert isinstance(round2["review_call_ms"], float) and round2["review_call_ms"] >= 0
    assert "revision_call_ms" not in round2  # accepted -> no revision call made

    assert len(record["initial_plan_calls"]) == 1
    assert len(round1["review_calls"]) == 1 and len(round1["revision_calls"]) == 1
    assert len(round2["review_calls"]) == 1 and "revision_calls" not in round2
    assert record["plan_review_started_at"] <= record["plan_review_ended_at"]


def test_planning_loop_respects_max_rounds():
    initial = json.dumps({"plan_summary": "v0", "steps": [], "uncertainties": [], "stop_condition": "x"})
    revise = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 0.4, "feedback": "again"},
                         "private_diagnosis": {}, "teacher_decision": "revise_plan"})
    revised = json.dumps({"revision_summary": "r", "plan_summary": "vN", "steps": [],
                          "teacher_feedback_used": [], "stop_condition": "x"})
    ctx = _context({"enabled": True, "review_guidance_level": 3, "planning_steps": 2})
    stub = SequencedStub(initial, [revise, revise], revised)
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    asyncio.run(comp.execute(ctx))
    record = ctx.metadata["plan_review"]
    # never accepted -> exactly planning_steps rounds, each with a revision
    assert record["num_planning_rounds"] == 2
    assert record["metrics"]["revisions_done"] == 2


def test_teacher_planner_authors_and_sanitizes_plan():
    # Teacher writes the plan and (wrongly) puts the gold answer in a step goal.
    teacher_plan = json.dumps({
        "plan_summary": "Search for both, then compare professions.",
        "steps": [
            {"step_id": 1, "goal": "search for the answer Delhi", "intended_tool": "search", "rationale": "x", "depends_on": []},
            {"step_id": 2, "goal": "verify", "intended_tool": "verify", "rationale": "y", "depends_on": [1]},
        ],
        "uncertainties": [], "stop_condition": "verified",
    })

    class TeacherStub:
        def __init__(self): self.calls = 0
        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            self.calls += 1
            return teacher_plan

    ctx = _context({"enabled": True, "planner": "teacher"})
    stub = TeacherStub()
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    result = asyncio.run(comp.execute(ctx))

    assert result.data["verdict"] == "PROCEED"
    record = ctx.metadata["plan_review"]
    assert record["planner"] == "teacher"
    assert record["initial_student_plan"] is None
    # exactly one LLM call (teacher) — no student drafting/revision
    assert stub.calls == 1
    # the revised plan is the teacher plan, with the gold answer sanitized out
    plan_text = json.dumps(ctx.metadata["revised_plan"])
    assert "Delhi" not in plan_text
    assert record["leakage_check"]["gold_answer_leaked"] is True
    assert ctx.metadata["revised_plan"]["steps"][1]["intended_tool"] == "verify"
    assert isinstance(record["plan_call_ms"], float) and record["plan_call_ms"] >= 0
    assert isinstance(record["plan_review_elapsed_ms"], float) and record["plan_review_elapsed_ms"] >= 0
    assert len(record["plan_calls"]) == 1
    assert record["plan_review_started_at"] <= record["plan_review_ended_at"]


class TruncatedReviewStub:
    """Initial plan ok; first teacher review is truncated garbage, retry succeeds."""

    def __init__(self, initial, bad_review, good_review):
        self.initial, self.bad_review, self.good_review = initial, bad_review, good_review
        self.review_calls = 0

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        if "teacher reviewing" in prompt:
            self.review_calls += 1
            if self.review_calls == 1:
                return self.bad_review
            return self.good_review
        return self.initial


def test_teacher_plan_review_repairs_truncated_response():
    initial = json.dumps({
        "plan_summary": "v0", "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search",
        "rationale": "x", "depends_on": []}], "uncertainties": [], "stop_condition": "done",
    })
    # Genuinely unparseable (no JSON object at all -- unterminated JSON gets recovered
    # by the json_repair fallback tier now, so it no longer exercises the repair path).
    bad_review = "The plan looks reasonable but I didn't return JSON here."
    good_review = json.dumps({
        "plan_review_enabled": True, "review_guidance_level": 3,
        "student_visible": {"score_continuous": 0.8, "feedback": "Solid plan, add a verification step."},
        "private_diagnosis": {}, "teacher_decision": "accept_plan",
    })
    ctx = _context({"enabled": True, "review_guidance_level": 3})
    stub = TruncatedReviewStub(initial, bad_review, good_review)
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    result = asyncio.run(comp.execute(ctx))

    record = ctx.metadata["plan_review"]
    assert record["rounds"][0]["teacher_plan_review_repair_attempts"] == 1
    assert record["teacher_plan_review_repair_attempts"] == 1
    assert record["student_visible_plan_feedback"]["feedback"] != ""
    assert "verification" in record["student_visible_plan_feedback"]["feedback"]
    assert stub.review_calls == 2
    assert result.data["verdict"] == "PROCEED"

    # The failed first attempt's truncated raw text is preserved, not just the
    # winning retry.
    review_calls = record["rounds"][0]["review_calls"]
    assert len(review_calls) == 2
    assert review_calls[0]["response_text"] == bad_review
    assert review_calls[1]["response_text"] == good_review
    assert record["rounds"][0]["review_call_ms"] == sum(c["elapsed_ms"] for c in review_calls)
    assert record["plan_review_started_at"] <= record["plan_review_ended_at"]
    assert len(record["initial_plan_calls"]) == 1


class AlwaysTruncatedReviewStub:
    """Every teacher review call returns truncated garbage (retries exhausted)."""

    def __init__(self, initial, bad_review):
        self.initial, self.bad_review = initial, bad_review
        self.review_calls = 0

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        if "teacher reviewing" in prompt:
            self.review_calls += 1
            return self.bad_review
        return self.initial


def test_teacher_plan_review_falls_back_when_repair_exhausted():
    initial = json.dumps({
        "plan_summary": "v0", "steps": [], "uncertainties": [], "stop_condition": "done",
    })
    bad_review = "Sorry, I can't format this as JSON right now."
    ctx = _context({"enabled": True, "review_guidance_level": 3})
    stub = AlwaysTruncatedReviewStub(initial, bad_review)
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    record = ctx.metadata["plan_review"]
    # default teacher_max_repair_attempts is 1 -> base call + 1 retry = 2 calls
    assert stub.review_calls == 2
    assert record["teacher_plan_review_repair_attempts"] == 1
    assert record["teacher_plan_review_full"] == {}
    # the guidance_policy fallback kicks in: never blank, even with retries exhausted
    assert record["student_visible_plan_feedback"]["feedback"] == FALLBACK_FEEDBACK
    assert record["leakage_check"]["feedback_fallback_used"] is True


def test_accept_plan_skips_revision():
    initial = json.dumps({
        "plan_summary": "search then verify then answer",
        "steps": [{"step_id": 1, "goal": "find", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [], "stop_condition": "have answer",
    })
    review = json.dumps({
        "plan_review_enabled": True, "review_guidance_level": 3,
        "student_visible": {"score_continuous": 1.0, "feedback": "Well-structured. Proceed."},
        "private_diagnosis": {"plan_valid": True},
        "teacher_decision": "accept_plan",
    })
    # If the revision branch were taken, this would be returned (and detected).
    revised = json.dumps({"revision_summary": "SHOULD NOT BE USED", "plan_summary": "x", "steps": []})

    ctx = _context({"enabled": True, "review_guidance_level": 3})
    stub = StubLLM(initial, review, revised)
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    result = asyncio.run(comp.execute(ctx))

    assert result.data["verdict"] == "PROCEED"
    record = ctx.metadata["plan_review"]
    assert record["revision_skipped"] is True
    assert record["revised_student_plan_raw"] is None
    # revised plan == initial plan (no extra LLM call)
    assert ctx.metadata["revised_plan"]["plan_summary"] == "search then verify then answer"
    assert "SHOULD NOT BE USED" not in json.dumps(ctx.metadata["revised_plan"])
    # exactly two LLM calls were made (initial plan + teacher review), not three
    assert stub.calls == 2
    assert record["metrics"]["revision_skipped"] is True


def test_plan_review_calls_request_constrained_output_schemas():
    initial = json.dumps({
        "plan_summary": "v0", "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search", "rationale": "x", "depends_on": []}],
        "uncertainties": [], "stop_condition": "done",
    })
    revise = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 0.5, "feedback": "revise"},
                         "private_diagnosis": {}, "teacher_decision": "revise_plan"})
    accept = json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                         "student_visible": {"score_continuous": 1.0, "feedback": "good"},
                         "private_diagnosis": {}, "teacher_decision": "accept_plan"})
    revised = json.dumps({"revision_summary": "r", "plan_summary": "v1",
                          "steps": [{"step_id": 1, "goal": "g", "intended_tool": "search", "rationale": "x", "depends_on": []}],
                          "teacher_feedback_used": [], "stop_condition": "done"})

    class SchemaCapturingStub:
        def __init__(self):
            self.calls = []  # (kind, response_schema)

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, response_schema=None, **kw):
            if "teacher reviewing" in prompt:
                self.calls.append(("review", response_schema))
                return revise if len([c for c in self.calls if c[0] == "review"]) == 1 else accept
            if "Revise your plan" in prompt:
                self.calls.append(("revision", response_schema))
                return revised
            self.calls.append(("initial", response_schema))
            return initial

    ctx = _context({"enabled": True, "review_guidance_level": 3, "planning_steps": 3})
    stub = SchemaCapturingStub()
    comp = TeacherGuidedPlanReview(config={}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    schemas_by_kind = dict(stub.calls)
    assert schemas_by_kind["initial"] == STUDENT_PLAN_SCHEMA
    assert schemas_by_kind["review"] == TEACHER_PLAN_REVIEW_SCHEMA
    assert schemas_by_kind["revision"] == REVISED_STUDENT_PLAN_SCHEMA
