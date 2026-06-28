"""
Deterministic tool executor for the Teacher Guidance environment.

Executes the tool the student selected and updates the shared workflow context.
Tool execution is deterministic (no LLM): the environment owns the corpus, validates
extracted spans against retrieved text, and records all state on ``context.metadata``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentsim.workflow.context import EvidenceSpan
from agentsim.teacher_guidance.schemas import StudentAction, ToolObservation


def _scope(context: Any) -> Dict[str, Any]:
    return context.metadata.get("retrieval_scope", {}) or {}


def _qid(context: Any) -> str:
    return _scope(context).get("qid") or context.metadata.get("sample_id") or context.task_id


def _candidate_doc_ids(context: Any) -> Optional[List[str]]:
    return _scope(context).get("candidate_doc_ids")


def _record_action(context: Any, action: StudentAction) -> None:
    summary = {"tool": action.action.tool, "params": action.action.params}
    context.metadata.setdefault("previous_actions", []).append(summary)


def _ingest_facts(context: Any, action: StudentAction) -> List[Dict[str, Any]]:
    """Validate student-declared new_facts against retrieved evidence and store valid ones."""
    stored: List[Dict[str, Any]] = []
    for fact in action.new_facts_extracted:
        evidence = context.get_evidence_by_id(fact.doc_id)
        if evidence and fact.span and fact.span.strip() and fact.span.strip() in evidence.text:
            entry = {"doc_id": fact.doc_id, "span": fact.span, "fact": fact.fact}
            context.metadata.setdefault("extracted_facts", []).append(entry)
            stored.append(entry)
    return stored


def _do_search(context: Any, params: Dict[str, Any], retriever) -> ToolObservation:
    query = str(params.get("query", "") or context.query)
    k = int(params.get("k", 5) or 5)
    qid = _qid(context)
    results = retriever.search(qid, query, k=k, candidate_doc_ids=_candidate_doc_ids(context))

    retrieved_docs = context.metadata.setdefault("retrieved_docs", [])
    retrieved_ids = context.metadata.setdefault("retrieved_doc_ids", [])
    seen = set(retrieved_ids)
    for r in results:
        doc_id = r["doc_id"]
        if doc_id not in seen:
            retrieved_docs.append(r)
            retrieved_ids.append(doc_id)
            seen.add(doc_id)
            full = retriever.get_doc(doc_id)
            if full:
                context.add_evidence(
                    EvidenceSpan(
                        source="hotpot_local",
                        id=doc_id,
                        text=full.get("text", ""),
                        doc_meta={"title": full.get("title", "")},
                    )
                )
    return ToolObservation(tool="search", status="ok", data={"query": query, "k": k, "results": results})


def _do_extract(context: Any, params: Dict[str, Any]) -> ToolObservation:
    doc_ids = params.get("doc_ids", []) or []
    target_facts = params.get("target_facts", []) or []
    extracted: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    for span in target_facts:
        span_text = str(span)
        matched_doc = None
        # Prefer the explicitly named docs, else any retrieved doc.
        search_ids = doc_ids if doc_ids else context.metadata.get("retrieved_doc_ids", [])
        for doc_id in search_ids:
            evidence = context.get_evidence_by_id(doc_id)
            if evidence and span_text.strip() and span_text.strip() in evidence.text:
                matched_doc = doc_id
                break
        if matched_doc:
            entry = {"doc_id": matched_doc, "span": span_text, "fact": span_text}
            context.metadata.setdefault("extracted_facts", []).append(entry)
            extracted.append(entry)
        else:
            invalid.append({"doc_id": doc_ids[0] if doc_ids else None, "span": span_text, "reason": "span_not_found"})

    status = "ok" if not invalid else ("partial_error" if extracted else "error")
    return ToolObservation(
        tool="extract", status=status, data={"extracted": extracted, "invalid_spans": invalid}
    )


def _do_verify(context: Any, params: Dict[str, Any], retriever) -> ToolObservation:
    claim = str(params.get("claim", ""))
    query = str(params.get("query", "") or claim)
    k = int(params.get("k", 5) or 5)
    qid = _qid(context)
    results = retriever.search(qid, query, k=k, candidate_doc_ids=_candidate_doc_ids(context))
    supported = False
    for r in results:
        full = retriever.get_doc(r["doc_id"])
        if full and claim.strip() and claim.strip().lower() in full.get("text", "").lower():
            supported = True
            break
    return ToolObservation(
        tool="verify", status="ok", data={"claim": claim, "supported": supported, "results": results}
    )


def _do_synthesize(context: Any) -> ToolObservation:
    facts = context.metadata.get("extracted_facts", [])
    draft = " ".join(f.get("fact", "") for f in facts).strip()
    context.metadata["draft_answer"] = draft
    return ToolObservation(tool="synthesize", status="ok", data={"draft_answer": draft})


def _do_finish(context: Any, params: Dict[str, Any]) -> ToolObservation:
    answer = str(params.get("answer", "") or context.metadata.get("draft_answer", "") or "")
    citations = params.get("citations", []) or []
    context.metadata["candidate_final_answer"] = answer
    context.metadata["final_answer"] = answer
    context.metadata["final_citations"] = citations
    return ToolObservation(
        tool="finish", status="ok", data={"answer": answer, "citations": citations}
    )


def execute_student_tool(context: Any, action: StudentAction, retriever) -> Dict[str, Any]:
    """Execute the student's selected tool and return the observation as a dict."""
    tool = action.action.tool
    params = action.action.params or {}
    _record_action(context, action)

    if tool == "decompose":
        sub_questions = params.get("sub_questions", []) or []
        context.metadata["sub_questions"] = sub_questions
        obs = ToolObservation(tool="decompose", status="ok", data={"sub_questions": sub_questions})
    elif tool == "reformulate":
        queries = params.get("queries", []) or []
        context.metadata["reformulated_queries"] = queries
        obs = ToolObservation(
            tool="reformulate",
            status="ok",
            data={"queries": queries, "reformulation_type": params.get("reformulation_type")},
        )
    elif tool == "search":
        obs = _do_search(context, params, retriever)
    elif tool == "extract":
        obs = _do_extract(context, params)
    elif tool == "verify":
        obs = _do_verify(context, params, retriever)
    elif tool == "synthesize":
        obs = _do_synthesize(context)
    elif tool == "finish":
        obs = _do_finish(context, params)
    else:
        obs = ToolObservation(tool=tool or "unknown", status="invalid", data={"reason": "unknown_tool"})

    # Ingest any student-declared facts (validated against retrieved evidence) for
    # tools that do not themselves manage extraction.
    if tool not in {"extract"}:
        _ingest_facts(context, action)

    return obs.to_dict()
