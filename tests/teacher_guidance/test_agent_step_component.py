"""End-to-end test of the teacher_guided_agent_step component with a stub LLM."""

import asyncio
import json

import pytest

from agentsim.workflow.context import WorkflowContext
from agentsim.components.control.teacher_guided_agent_step import TeacherGuidedAgentStep


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
