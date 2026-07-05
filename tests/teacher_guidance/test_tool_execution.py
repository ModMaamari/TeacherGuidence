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


def test_extract_tolerates_wrapping_quotes(context, retriever):
    # The exact failure from the local run: the student wrapped the span in quotes.
    execute_student_tool(context, _action("search", {"query": "Delhi", "k": 2}), retriever)
    obs = execute_student_tool(
        context,
        _action("extract", {"doc_ids": ["q1::doc1"], "target_facts": ['"headquartered in Delhi"']}),
        retriever,
    )
    assert obs["status"] == "ok"
    # stored span has the quotes stripped
    assert obs["extracted"][0]["span"] == "headquartered in Delhi"


def test_extract_tolerates_case_and_whitespace(context, retriever):
    execute_student_tool(context, _action("search", {"query": "Delhi", "k": 2}), retriever)
    obs = execute_student_tool(
        context,
        _action("extract", {"doc_ids": ["q1::doc1"], "target_facts": ["  HEADQUARTERED   in   Delhi "]}),
        retriever,
    )
    assert obs["status"] == "ok"
    assert obs["extracted"][0]["span"] == "HEADQUARTERED in Delhi"


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


def test_finish_never_empty_uses_draft(context, retriever):
    context.metadata["draft_answer"] = "Delhi"
    obs = execute_student_tool(context, _action("finish", {}), retriever)
    assert obs["answer"] == "Delhi"
    assert context.metadata["final_answer"] == "Delhi"


def test_finish_never_empty_last_resort(context, retriever):
    # no answer, no draft, no facts -> still non-empty
    obs = execute_student_tool(context, _action("finish", {"citations": []}), retriever)
    assert obs["answer"] and obs["answer"] != ""
    assert context.metadata["final_answer"]


def test_forced_finish_reuses_prior_committed_answer(context, retriever):
    # The BBC case: the student committed "the BBC, in London" at an earlier finish,
    # the teacher asked it to keep going, budget ran out -> the forced finish (empty
    # params) must reuse that committed answer, not fall back to "unknown".
    execute_student_tool(context, _action("finish", {"answer": "the BBC, in London"}), retriever)
    assert context.metadata["candidate_final_answer"] == "the BBC, in London"
    obs = execute_student_tool(context, _action("finish", {}), retriever)
    assert obs["answer"] == "the BBC, in London"
    assert context.metadata["final_answer"] == "the BBC, in London"


def test_derive_final_answer_priority_and_unknown_not_resurfaced():
    from agentsim.teacher_guidance.tool_executor import derive_final_answer

    class _Ctx:
        def __init__(self, md):
            self.metadata = md

    # explicit params answer wins
    assert derive_final_answer(_Ctx({"candidate_final_answer": "old"}), {"answer": "new"}) == "new"
    # a prior committed answer beats draft/facts
    assert derive_final_answer(
        _Ctx({"candidate_final_answer": "committed", "draft_answer": "draft"})
    ) == "committed"
    # a prior "unknown" is never resurfaced -- fall through to the draft
    assert derive_final_answer(
        _Ctx({"candidate_final_answer": "unknown", "draft_answer": "draft"})
    ) == "draft"
    # nothing at all -> the placeholder
    assert derive_final_answer(_Ctx({})) == "unknown"


def test_clean_forced_answer_normalizes_and_rejects_placeholders():
    from agentsim.teacher_guidance.tool_executor import clean_forced_answer

    # bare phrase, quoted / whitespace -> cleaned
    assert clean_forced_answer('  "August 1973"  ') == "August 1973"
    # code-fenced JSON with an answer field -> pulled out
    assert clean_forced_answer('```json\n{"answer": "Delhi"}\n```') == "Delhi"
    # nested action/params shape (model ignored the "no JSON" instruction)
    assert clean_forced_answer('{"action": {"params": {"answer": "London"}}}') == "London"
    # multi-line prose -> first non-empty line
    assert clean_forced_answer("The Beatles\nsome trailing note") == "The Beatles"
    # placeholders / refusals are rejected so they never become the final answer
    assert clean_forced_answer("unknown") == ""
    assert clean_forced_answer("I don't know the answer") == ""
    assert clean_forced_answer("") == ""


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
