"""Tests for the deterministic tool executor."""

import json

import pytest

from agentsim.workflow.context import WorkflowContext
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever
from agentsim.teacher_guidance.schemas import StudentAction
from agentsim.teacher_guidance.tool_executor import execute_student_tool


CORPUS = [
    {"doc_id": "q1::doc0", "qid": "q1", "title": "Oberoi family",
     "text": "The Oberoi family runs hotels.", "sentences": ["The Oberoi family runs hotels."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
    {"doc_id": "q1::doc1", "qid": "q1", "title": "The Oberoi Group",
     "text": "The Oberoi Group is headquartered in Delhi.",
     "sentences": ["The Oberoi Group is headquartered in Delhi."],
     "is_gold_doc": True, "gold_sent_ids": [0], "source": "hotpotqa", "split": "validation"},
]


@pytest.fixture
def retriever(tmp_path):
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in CORPUS:
            f.write(json.dumps(row) + "\n")
    return HotpotLocalRetriever(str(path))


@pytest.fixture
def context():
    ctx = WorkflowContext(
        task_id="q1",
        query="Where is the Oberoi Group headquartered?",
        metadata={"retrieval_scope": {"qid": "q1", "candidate_doc_ids": ["q1::doc0", "q1::doc1"]}},
    )
    return ctx


def _action(tool, params, facts=None):
    return StudentAction.from_dict(
        {"action": {"tool": tool, "params": params}, "new_facts_extracted": facts or []}
    )


def test_search_adds_evidence_and_previews(context, retriever):
    obs = execute_student_tool(context, _action("search", {"query": "Delhi headquarters", "k": 2}), retriever)
    assert obs["status"] == "ok"
    assert context.metadata["retrieved_doc_ids"]
    # full text is available as evidence for later extraction
    assert context.get_evidence_by_id("q1::doc1").text.startswith("The Oberoi Group")


def test_extract_accepts_verbatim_span(context, retriever):
    execute_student_tool(context, _action("search", {"query": "Delhi", "k": 2}), retriever)
    obs = execute_student_tool(
        context,
        _action("extract", {"doc_ids": ["q1::doc1"], "target_facts": ["headquartered in Delhi"]}),
        retriever,
    )
    assert obs["status"] == "ok"
    assert obs["extracted"][0]["doc_id"] == "q1::doc1"
    assert any(f["span"] == "headquartered in Delhi" for f in context.metadata["extracted_facts"])


def test_extract_rejects_paraphrase(context, retriever):
    execute_student_tool(context, _action("search", {"query": "Delhi", "k": 2}), retriever)
    obs = execute_student_tool(
        context,
        _action("extract", {"doc_ids": ["q1::doc1"], "target_facts": ["it is based in New Delhi"]}),
        retriever,
    )
    assert obs["status"] == "error"
    assert obs["invalid_spans"][0]["reason"] == "span_not_found"


def test_decompose_stores_sub_questions(context, retriever):
    obs = execute_student_tool(
        context, _action("decompose", {"sub_questions": ["Who is the Oberoi Group?", "Where is its HQ?"]}), retriever
    )
    assert obs["sub_questions"][0].startswith("Who is")
    assert context.metadata["sub_questions"]


def test_finish_sets_final_answer(context, retriever):
    obs = execute_student_tool(
        context, _action("finish", {"answer": "Delhi", "citations": [{"doc_id": "q1::doc1", "span": "Delhi"}]}), retriever
    )
    assert obs["answer"] == "Delhi"
    assert context.metadata["final_answer"] == "Delhi"


def test_new_facts_validated_against_evidence(context, retriever):
    execute_student_tool(context, _action("search", {"query": "Delhi", "k": 2}), retriever)
    # valid span via new_facts_extracted on a non-extract tool (synthesize)
    execute_student_tool(
        context,
        _action("synthesize", {}, facts=[{"doc_id": "q1::doc1", "span": "headquartered in Delhi", "fact": "HQ in Delhi"}]),
        retriever,
    )
    assert any(f["fact"] == "HQ in Delhi" for f in context.metadata["extracted_facts"])
    # invalid span is dropped
    execute_student_tool(
        context,
        _action("synthesize", {}, facts=[{"doc_id": "q1::doc1", "span": "not present", "fact": "bogus"}]),
        retriever,
    )
    assert not any(f["fact"] == "bogus" for f in context.metadata["extracted_facts"])
