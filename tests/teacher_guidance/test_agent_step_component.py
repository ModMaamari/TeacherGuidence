"""End-to-end test of the teacher_guided_agent_step component with a stub LLM."""

import asyncio
import json

import pytest

from agentsim.workflow.context import WorkflowContext
from agentsim.components.control.teacher_guided_agent_step import (
    TeacherGuidedAgentStep,
    STUDENT_ACTION_SCHEMA,
    STUDENT_FINISH_ACTION_SCHEMA,
    TEACHER_EVALUATION_SCHEMA,
    _extract_teacher_final_judgment,
)
from agentsim.teacher_guidance.guidance_policy import FALLBACK_FEEDBACK


CORPUS = [
    {"doc_id": "q1::doc0", "qid": "q1", "title": "The Oberoi Group",
     "text": "The Oberoi Group is headquartered in Delhi.",
     "sentences": ["The Oberoi Group is headquartered in Delhi."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
]


class StubLLM:
    """Returns canned student/teacher JSON based on a marker in the prompt."""

    def __init__(self, student_json, teacher_json):
        self.student_json = student_json
        self.teacher_json = teacher_json

    async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
        if "teacher evaluating" in prompt:
            return self.teacher_json
        return self.student_json


class RouterStubLLM(StubLLM):
    """Adds a provider-fallback router that always serves the 2nd model (as if FAU was
    rate-limited)."""

    def __init__(self, student_json, teacher_json):
        super().__init__(student_json, teacher_json)
        self.router_calls = []

    async def get_completion_with_fallback(self, models, *, prompt, **kw):
        self.router_calls.append(list(models))
        text = self.teacher_json if "teacher evaluating" in prompt else self.student_json
        return text, models[1]  # pretend the free fallback served it


def _context(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in CORPUS:
            f.write(json.dumps(row) + "\n")
    return WorkflowContext(
        task_id="q1",
        query="Where is the Oberoi Group headquartered?",
        metadata={
            "corpus_path": str(corpus_path),
            "student_model": "stub",
            "teacher_model": "stub",
            "guidance": {"level": 3, "max_feedback_words": 40, "leak_policy": "strict"},
            "gold": {
                "answer": "Delhi",
                "supporting_titles": ["The Oberoi Group"],
                "supporting_facts": [{"title": "The Oberoi Group", "sent_id": 0}],
                "gold_doc_ids": ["q1::doc0"],
            },
            "retrieval_scope": {"qid": "q1", "candidate_doc_ids": ["q1::doc0"]},
        },
    )


def test_search_step_records_and_continues(tmp_path):
    student = json.dumps({
        "thought": "search",
        "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters", "k": 3}},
        "new_facts_extracted": [],
    })
    teacher = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_binary": 1, "score_continuous": 0.7, "feedback": "Good retrieval."},
        "private_diagnosis": {"retrieved_gold_doc": True},
        "teacher_decision": "continue",
    })
    ctx = _context(tmp_path)
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=StubLLM(student, teacher))
    result = asyncio.run(comp.execute(ctx))

    assert result.success and result.data["verdict"] == "PROCEED"
    assert len(ctx.metadata["teacher_guided_steps"]) == 1
    assert ctx.metadata["retrieved_doc_ids"] == ["q1::doc0"]
    assert ctx.metadata["last_teacher_guidance_for_student"]["score"] == 0.7
    assert not ctx.metadata.get("done")


def test_extract_teacher_final_judgment_shapes():
    assert _extract_teacher_final_judgment(None) is None
    assert _extract_teacher_final_judgment({"step_correct": True}) is None
    assert _extract_teacher_final_judgment({"final_answer_correct": 1, "final_answer_score": 0.9}) == {"correct": 1, "score": 0.9}
    assert _extract_teacher_final_judgment({"final_answer_correct": True, "final_answer_score": 1.0}) == {"correct": 1, "score": 1.0}
    assert _extract_teacher_final_judgment({"final_answer_correct": "yes"}) == {"correct": 1, "score": 1.0}
    # score-only -> derive binary; clamp out-of-range
    assert _extract_teacher_final_judgment({"final_answer_score": 0.3}) == {"correct": 0, "score": 0.3}
    assert _extract_teacher_final_judgment({"final_answer_score": 1.7}) == {"correct": 1, "score": 1.0}
    # garbage score, binary present
    assert _extract_teacher_final_judgment({"final_answer_correct": 0, "final_answer_score": "n/a"}) == {"correct": 0, "score": 0.0}


def test_finish_step_captures_teacher_final_judgment(tmp_path):
    student = json.dumps({
        "thought": "I have the answer now.",
        "action": {"tool": "finish", "params": {"answer": "Delhi", "citations": []}},
    })
    teacher = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_continuous": 0.9, "feedback": "good"},
        "private_diagnosis": {"final_answer_correct": 1, "final_answer_score": 0.95},
        "teacher_decision": "accept_finish",
    })
    ctx = _context(tmp_path)
    comp = TeacherGuidedAgentStep(config={"step_index": 3, "budget": 5}, llm_client=StubLLM(student, teacher))
    asyncio.run(comp.execute(ctx))
    assert ctx.metadata["teacher_final_judgment"] == {"correct": 1, "score": 0.95}


def test_teacher_call_uses_router_when_configured(tmp_path):
    student = json.dumps({
        "thought": "I will search for the headquarters.",
        "action": {"tool": "search", "params": {"query": "Oberoi HQ", "k": 3}},
    })
    teacher = json.dumps({
        "guidance_level": 3, "student_visible": {"score_continuous": 0.6, "feedback": "ok"},
        "private_diagnosis": {}, "teacher_decision": "continue",
    })
    router = ["fau/gpt-oss-120b", "custom/openai/gpt-oss-120b:free", "custom/openai/gpt-oss-120b"]
    ctx = _context(tmp_path)
    ctx.metadata["teacher_router"] = router
    stub = RouterStubLLM(student, teacher)
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    assert stub.router_calls and stub.router_calls[0] == router
    # The recorded teacher call notes which model actually served it.
    step = ctx.metadata["teacher_guided_steps"][0]
    assert step["teacher_calls"][0]["model"] == "custom/openai/gpt-oss-120b:free"


def test_force_finish_overrides_and_stops(tmp_path):
    # Student tries to keep searching, but force_finish must convert to finish.
    student = json.dumps({
        "thought": "keep searching",
        "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "more", "k": 3}},
        "new_facts_extracted": [],
    })
    teacher = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_continuous": 0.2, "feedback": "ok"},
        "private_diagnosis": {},
        "teacher_decision": "force_finish",
    })
    ctx = _context(tmp_path)
    ctx.metadata["draft_answer"] = "Delhi"
    comp = TeacherGuidedAgentStep(
        config={"step_index": 5, "budget": 5, "force_finish": True}, llm_client=StubLLM(student, teacher)
    )
    result = asyncio.run(comp.execute(ctx))

    assert result.data["verdict"] == "FINISH"
    assert ctx.metadata["done"] is True
    assert ctx.metadata["stop_reason"] == "budget_forced_finish"
    assert ctx.metadata["final_answer"] == "Delhi"
    assert ctx.metadata["teacher_guided_steps"][0]["student_action"]["action"]["tool"] == "finish"


def test_student_repair_retry_on_invalid_action(tmp_path):
    # First student output is invalid (bad tool); a retry returns a valid action.
    bad = json.dumps({"thought": "hmm", "decision": {"category": "need_retrieval"},
                      "action": {"tool": "google_it", "params": {}}, "new_facts_extracted": []})
    good = json.dumps({"thought": "search", "decision": {"category": "need_retrieval"},
                       "action": {"tool": "search", "params": {"query": "Oberoi HQ", "k": 2}},
                       "new_facts_extracted": []})
    teacher = json.dumps({"guidance_level": 3, "student_visible": {"score_continuous": 0.6, "feedback": "ok"},
                          "private_diagnosis": {}, "teacher_decision": "continue"})

    class RepairStub:
        def __init__(self): self.student_calls = 0
        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "teacher evaluating" in prompt:
                return teacher
            self.student_calls += 1
            # invalid first, valid on the correction retry
            return bad if "previous response was not a valid action" not in prompt else good

    ctx = _context(tmp_path)
    stub = RepairStub()
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    step = ctx.metadata["teacher_guided_steps"][0]
    assert step["student_repair_attempts"] == 1
    assert step["student_action"]["action"]["tool"] == "search"  # ended valid
    assert stub.student_calls == 2  # one retry


def test_teacher_eval_repairs_truncated_response(tmp_path):
    student = json.dumps({
        "thought": "search", "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters", "k": 3}},
        "new_facts_extracted": [],
    })
    # Genuinely unparseable (no JSON object at all -- unterminated JSON gets recovered
    # by the json_repair fallback tier now, so it no longer exercises the repair path).
    bad_teacher = "I think the student did fine but I forgot to write JSON, sorry about that."
    good_teacher = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_binary": 1, "score_continuous": 0.7, "feedback": "Good retrieval."},
        "private_diagnosis": {"retrieved_gold_doc": True},
        "teacher_decision": "continue",
    })

    class TeacherRepairStub:
        def __init__(self):
            self.teacher_calls = 0

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "teacher evaluating" in prompt:
                self.teacher_calls += 1
                return bad_teacher if self.teacher_calls == 1 else good_teacher
            return student

    ctx = _context(tmp_path)
    stub = TeacherRepairStub()
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=stub)
    result = asyncio.run(comp.execute(ctx))

    step = ctx.metadata["teacher_guided_steps"][0]
    assert step["teacher_repair_attempts"] == 1
    assert stub.teacher_calls == 2
    assert result.data["student_visible_guidance"]["feedback"] == "Good retrieval."

    # The failed first attempt's raw (truncated) text is preserved in teacher_calls,
    # not just the winning retry -- this is exactly the truncation evidence that would
    # be needed to debug why a repair was triggered.
    calls = step["teacher_calls"]
    assert len(calls) == 2
    assert calls[0]["attempt"] == 1
    assert calls[0]["response_text"] == bad_teacher
    assert calls[1]["attempt"] == 2
    assert calls[1]["response_text"] == good_teacher
    for c in calls:
        assert c["started_at"] <= c["ended_at"]
        assert c["elapsed_ms"] >= 0
    assert step["teacher_call_ms"] == sum(c["elapsed_ms"] for c in calls)


def test_teacher_eval_falls_back_when_repair_exhausted(tmp_path):
    student = json.dumps({
        "thought": "search", "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters", "k": 3}},
        "new_facts_extracted": [],
    })
    bad_teacher = "Sorry, I can't format this as JSON right now."

    class AlwaysBadTeacherStub:
        def __init__(self):
            self.teacher_calls = 0

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "teacher evaluating" in prompt:
                self.teacher_calls += 1
                return bad_teacher
            return student

    ctx = _context(tmp_path)
    stub = AlwaysBadTeacherStub()
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=stub)
    result = asyncio.run(comp.execute(ctx))

    step = ctx.metadata["teacher_guided_steps"][0]
    assert stub.teacher_calls == 2  # base call + 1 retry (default max)
    assert step["teacher_repair_attempts"] == 1
    assert result.data["student_visible_guidance"]["feedback"] == FALLBACK_FEEDBACK
    assert step["leakage_check"]["feedback_fallback_used"] is True


def test_step_record_has_call_and_elapsed_timing(tmp_path):
    student = json.dumps({
        "thought": "search",
        "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters", "k": 3}},
        "new_facts_extracted": [],
    })
    teacher = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_binary": 1, "score_continuous": 0.7, "feedback": "Good retrieval."},
        "private_diagnosis": {"retrieved_gold_doc": True},
        "teacher_decision": "continue",
    })
    ctx = _context(tmp_path)
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=StubLLM(student, teacher))
    asyncio.run(comp.execute(ctx))

    step = ctx.metadata["teacher_guided_steps"][0]
    for key in ("student_call_ms", "teacher_call_ms", "step_elapsed_ms"):
        assert isinstance(step[key], float)
        assert step[key] >= 0
    assert step["step_started_at"] <= step["step_ended_at"]
    for calls_key in ("student_calls", "teacher_calls"):
        calls = step[calls_key]
        assert len(calls) == 1
        assert calls[0]["attempt"] == 1
        assert calls[0]["started_at"] <= calls[0]["ended_at"]
        # StubLLM (test double) doesn't honor return_raw, so raw_response is None --
        # confirms the helper degrades gracefully rather than erroring.
        assert calls[0]["raw_response"] is None


def test_skip_teacher_makes_zero_teacher_calls_and_stops_on_own_finish(tmp_path):
    # No-teacher-guidance ablation: student decides everything, no teacher LLM call at all.
    student_finish = json.dumps({
        "thought": "done", "decision": {"category": "finish", "parametric_knowledge_used": False},
        "action": {"tool": "finish", "params": {"answer": "Delhi", "citations": []}},
        "new_facts_extracted": [],
    })

    class NoTeacherCallStub:
        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "teacher evaluating" in prompt:
                raise AssertionError("teacher should never be called when skip_teacher is set")
            return student_finish

    ctx = _context(tmp_path)
    ctx.metadata["skip_teacher"] = True
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=NoTeacherCallStub())
    result = asyncio.run(comp.execute(ctx))

    step = ctx.metadata["teacher_guided_steps"][0]
    assert step["teacher_calls"] == []
    assert step["teacher_call_ms"] == 0
    assert step["teacher_skipped"] is True
    assert step["student_visible_guidance"] == {}
    assert step["leakage_check"] == {}
    assert result.data["verdict"] == "FINISH"
    assert ctx.metadata["done"] is True
    assert ctx.metadata["stop_reason"] == "teacher_accept"


def test_guidance_does_not_leak_gold_answer(tmp_path):
    student = json.dumps({
        "thought": "finish", "decision": {"category": "finish", "parametric_knowledge_used": False},
        "action": {"tool": "finish", "params": {"answer": "Delhi", "citations": []}},
        "new_facts_extracted": [],
    })
    # Teacher tries to leak the gold answer in visible feedback.
    teacher = json.dumps({
        "guidance_level": 3,
        "student_visible": {"score_continuous": 1.0, "feedback": "Correct, the answer is Delhi."},
        "private_diagnosis": {},
        "teacher_decision": "accept_finish",
    })
    ctx = _context(tmp_path)
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=StubLLM(student, teacher))
    result = asyncio.run(comp.execute(ctx))
    feedback = result.data["student_visible_guidance"]["feedback"]
    assert "Delhi" not in feedback
    assert ctx.metadata["teacher_guided_steps"][0]["leakage_check"]["gold_answer_leaked"] is True


def test_student_and_teacher_calls_request_constrained_output_schemas(tmp_path):
    student = json.dumps({
        "thought": "search", "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters", "k": 3}},
        "new_facts_extracted": [],
    })
    teacher = json.dumps({
        "guidance_level": 3, "student_visible": {"score_continuous": 0.6, "feedback": "ok"},
        "private_diagnosis": {}, "teacher_decision": "continue",
    })

    class SchemaCapturingStub:
        def __init__(self):
            self.schemas = []

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, response_schema=None, **kw):
            self.schemas.append(response_schema)
            return teacher if "teacher evaluating" in prompt else student

    ctx = _context(tmp_path)
    stub = SchemaCapturingStub()
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    # First call is the student action, second is the teacher evaluation.
    assert stub.schemas[0] == STUDENT_ACTION_SCHEMA
    assert stub.schemas[1] == TEACHER_EVALUATION_SCHEMA


def test_force_finish_step_constrains_student_to_finish_only_schema(tmp_path):
    # On the final (force_finish) step the student's generation must be constrained to
    # the finish-only schema so the model commits an answer instead of searching again.
    student = json.dumps({
        "thought": "Based on the retrieved evidence the answer is Delhi.",
        "action": {"tool": "finish", "params": {"answer": "Delhi", "citations": []}},
    })
    teacher = json.dumps({
        "guidance_level": 3, "student_visible": {"score_continuous": 0.9, "feedback": "ok"},
        "private_diagnosis": {}, "teacher_decision": "accept_finish",
    })

    class SchemaCapturingStub:
        def __init__(self):
            self.schemas = []

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, response_schema=None, **kw):
            self.schemas.append(response_schema)
            return teacher if "teacher evaluating" in prompt else student

    ctx = _context(tmp_path)
    stub = SchemaCapturingStub()
    comp = TeacherGuidedAgentStep(
        config={"step_index": 5, "budget": 5, "force_finish": True}, llm_client=stub
    )
    asyncio.run(comp.execute(ctx))

    assert stub.schemas[0] == STUDENT_FINISH_ACTION_SCHEMA
    assert ctx.metadata["final_answer"] == "Delhi"


def test_teacher_sees_raw_text_when_student_totally_unparseable(tmp_path):
    # The student never returns valid JSON, even after the one repair retry --
    # confirms the teacher prompt shows the real raw text instead of a blank/default
    # parsed action (previously the teacher had zero signal about what happened).
    unparseable = "I looked at the docs but I'm not sure how to format my answer as JSON, sorry."
    teacher = json.dumps({
        "guidance_level": 3, "student_visible": {"score_continuous": 0.0, "feedback": "Invalid output."},
        "private_diagnosis": {}, "teacher_decision": "continue",
    })

    class AlwaysUnparseableStudentStub:
        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, **kw):
            if "teacher evaluating" in prompt:
                return teacher
            return unparseable

    ctx = _context(tmp_path)
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=AlwaysUnparseableStudentStub())
    asyncio.run(comp.execute(ctx))

    step = ctx.metadata["teacher_guided_steps"][0]
    assert step["metrics"]["json_valid"] is False
    assert unparseable in step["teacher_prompt"]
    assert "FAILED TO PARSE" in step["teacher_prompt"]
    # the blank default action's empty tool must not silently stand in for the raw text
    assert '"tool": ""' not in step["teacher_prompt"]


def test_student_schema_disabled_by_metadata_flag(tmp_path):
    # student_use_response_schema=False must drop the student's grammar constraint
    # (some models collapse under it) while leaving the teacher's schema intact.
    student = json.dumps({
        "thought": "search", "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
        "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters", "k": 3}},
        "new_facts_extracted": [],
    })
    teacher = json.dumps({
        "guidance_level": 3, "student_visible": {"score_continuous": 0.6, "feedback": "ok"},
        "private_diagnosis": {}, "teacher_decision": "continue",
    })

    class SchemaCapturingStub:
        def __init__(self):
            self.schemas = []

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, response_schema=None, **kw):
            self.schemas.append(response_schema)
            return teacher if "teacher evaluating" in prompt else student

    ctx = _context(tmp_path)
    ctx.metadata["student_use_response_schema"] = False
    stub = SchemaCapturingStub()
    comp = TeacherGuidedAgentStep(config={"step_index": 1, "budget": 5}, llm_client=stub)
    asyncio.run(comp.execute(ctx))

    assert stub.schemas[0] is None  # student unconstrained
    assert stub.schemas[1] == TEACHER_EVALUATION_SCHEMA  # teacher unchanged


def test_force_finish_keeps_finish_schema_even_when_student_schema_disabled(tmp_path):
    student = json.dumps({
        "thought": "Committing my final answer based on the retrieved evidence now.",
        "action": {"tool": "finish", "params": {"answer": "Delhi", "citations": []}},
    })
    teacher = json.dumps({
        "guidance_level": 3, "student_visible": {"score_continuous": 0.9, "feedback": "ok"},
        "private_diagnosis": {}, "teacher_decision": "accept_finish",
    })

    class SchemaCapturingStub:
        def __init__(self):
            self.schemas = []

        async def get_completion(self, prompt, model=None, temperature=0.0, max_tokens=None, response_schema=None, **kw):
            self.schemas.append(response_schema)
            return teacher if "teacher evaluating" in prompt else student

    ctx = _context(tmp_path)
    ctx.metadata["student_use_response_schema"] = False
    stub = SchemaCapturingStub()
    comp = TeacherGuidedAgentStep(
        config={"step_index": 5, "budget": 5, "force_finish": True}, llm_client=stub
    )
    asyncio.run(comp.execute(ctx))

    # The final commit step stays grammar-constrained to finish-only.
    assert stub.schemas[0] == STUDENT_FINISH_ACTION_SCHEMA
