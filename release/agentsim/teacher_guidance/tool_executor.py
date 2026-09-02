"""
Deterministic tool executor for the Teacher Guidance environment.

Executes the tool the student selected and updates the shared workflow context.
Tool execution is deterministic (no LLM): the environment owns the corpus, validates
extracted spans against retrieved text, and records all state on ``context.metadata``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agentsim.workflow.context import EvidenceSpan
from agentsim.teacher_guidance.schemas import StudentAction, ToolObservation

_QUOTE_CHARS = "\"'“”‘’`"


def clean_span(span: str) -> str:
    """Strip wrapping quotes and collapse whitespace from a student-provided span.

    Small models often wrap extract targets in literal quotes (e.g. '"English stage and
    film director"'), which breaks naive substring validation. We strip those and
    normalise whitespace without otherwise altering the text, so grounding is preserved.
    """
    s = (span or "").strip()
    while len(s) >= 2 and s[0] in _QUOTE_CHARS and s[-1] in _QUOTE_CHARS:
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def span_in_text(span: str, doc_text: str) -> bool:
    """True if the cleaned span appears in the doc text (whitespace- and case-tolerant)."""
    cleaned = clean_span(span)
    return bool(cleaned) and _norm(cleaned) in _norm(doc_text)


def _scope(context: Any) -> Dict[str, Any]:
    return context.metadata.get("retrieval_scope", {}) or {}


def _qid(context: Any) -> str:
    return _scope(context).get("qid") or context.metadata.get("sample_id") or context.task_id


def _candidate_doc_ids(context: Any) -> Optional[List[str]]:
    return _scope(context).get("candidate_doc_ids")


def _record_action(context: Any, action: StudentAction) -> None:
    summary = {"tool": action.action.tool, "params": action.action.params}
    context.metadata.setdefault("previous_actions", []).append(summary)
    # A wiki_read's content is surfaced in the *next* step's student prompt only
    # (one-shot); executing any new action retires the pending read. If this action
    # is itself a wiki_read, _do_wiki_read re-arms it afterwards.
    context.metadata.pop("wiki_just_read", None)


def _ingest_facts(context: Any, action: StudentAction) -> List[Dict[str, Any]]:
    """Validate student-declared new_facts against retrieved evidence and store valid ones."""
    stored: List[Dict[str, Any]] = []
    for fact in action.new_facts_extracted:
        evidence = context.get_evidence_by_id(fact.doc_id)
        if evidence and span_in_text(fact.span, evidence.text):
            cleaned = clean_span(fact.span)
            entry = {"doc_id": fact.doc_id, "span": cleaned, "fact": fact.fact or cleaned}
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
        cleaned = clean_span(span_text)
        matched_doc = None
        # Prefer the explicitly named docs, else any retrieved doc.
        search_ids = doc_ids if doc_ids else context.metadata.get("retrieved_doc_ids", [])
        for doc_id in search_ids:
            evidence = context.get_evidence_by_id(doc_id)
            if evidence and span_in_text(span_text, evidence.text):
                matched_doc = doc_id
                break
        if matched_doc:
            entry = {"doc_id": matched_doc, "span": cleaned, "fact": cleaned}
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


def _do_wiki_read(context: Any) -> ToolObservation:
    """Read the episode's wiki notes (wiki.md). The content is returned in the
    observation and stashed on the context so the next step's student prompt shows it."""
    content = str(context.metadata.get("wiki", "") or "")
    context.metadata["wiki_just_read"] = content
    return ToolObservation(tool="wiki_read", status="ok", data={"content": content, "chars": len(content)})


def _do_wiki_write(context: Any, params: Dict[str, Any]) -> ToolObservation:
    """Replace the episode's wiki notes with the given content (full rewrite -- the
    student is instructed to keep the file minimal, so it re-emits what it keeps)."""
    content = str(params.get("content", "") or "").strip()
    context.metadata["wiki"] = content
    return ToolObservation(tool="wiki_write", status="ok", data={"chars": len(content)})


def _do_synthesize(context: Any) -> ToolObservation:
    facts = context.metadata.get("extracted_facts", [])
    draft = " ".join(f.get("fact", "") for f in facts).strip()
    context.metadata["draft_answer"] = draft
    return ToolObservation(tool="synthesize", status="ok", data={"draft_answer": draft})


def _substantive(text: Any) -> str:
    """Return the stripped text if it is a real answer, else "". Treats the
    "unknown" placeholder as non-substantive so it never gets resurfaced as if it were
    a genuine answer by a later fallback."""
    s = str(text or "").strip()
    return "" if s.lower() == "unknown" else s


def clean_forced_answer(raw: str) -> str:
    """Turn a free-text (or accidentally-JSON) forced-finish answer reply into a clean
    answer string, or "" if there is nothing substantive.

    The forced-answer prompt asks for a bare phrase, but small models sometimes still wrap
    it in JSON or code fences, so we defensively pull ``answer`` out of a JSON object when
    present, strip fences/quotes, and reject the "unknown"/"I don't know" placeholders via
    ``_substantive``."""
    s = str(raw or "").strip()
    if not s:
        return ""
    # Strip a leading/trailing markdown code fence if the model added one.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    # If it came back as a JSON object, prefer its "answer" field.
    if s.startswith("{"):
        try:
            import json as _json
            obj = _json.loads(s)
            if isinstance(obj, dict):
                cand = obj.get("answer") or (obj.get("action", {}) or {}).get("params", {}).get("answer")
                if isinstance(cand, str):
                    s = cand.strip()
        except Exception:
            pass
    # Collapse to the first non-empty line and strip wrapping quotes.
    s = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    s = clean_span(s)
    lowered = s.lower()
    if lowered.startswith(("i don't know", "i do not know", "i'm not sure", "i am not sure")):
        return ""
    return _substantive(s)


def clean_wiki_content(raw: str, max_chars: int = 2000) -> str:
    """Normalize a free-text wiki-update reply into storable wiki.md content: strip a
    wrapping markdown code fence if the model added one, and hard-cap the length so a
    rambling model can never blow up every subsequent step's prompt."""
    s = str(raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def wiki_answer_candidate(wiki_text: Any) -> str:
    """Extract the ANSWER line's value from a templated wiki.md, or "".

    The auto-wiki template keeps the student's running best guess on an
    ``ANSWER: <candidate>`` line; that self-committed candidate is a better final
    answer than the "unknown" placeholder. Placeholder values ("?", "unknown") are
    rejected via ``_substantive``."""
    for line in str(wiki_text or "").splitlines():
        if line.strip().lower().startswith("answer:"):
            value = line.split(":", 1)[1].strip()
            return "" if value == "?" else _substantive(value)
    return ""


def derive_final_answer(context: Any, params: Optional[Dict[str, Any]] = None) -> str:
    """Best non-empty final answer, in priority order:

    1. the answer in *this* finish call's params,
    2. a previously-committed finish answer (``candidate_final_answer``) -- e.g. the
       student answered "the BBC, in London" at an earlier step, the teacher asked it to
       keep going, and then budget ran out: that committed answer must not be thrown away
       and replaced with "unknown" on the forced finish,
    3. the synthesized draft,
    4. the wiki's ANSWER line (wiki-enabled runs: the student's own best guess),
    5. the concatenated extracted facts,
    6. the "unknown" placeholder (never empty).
    """
    params = params or {}
    answer = _substantive(params.get("answer"))
    if not answer:
        answer = _substantive(context.metadata.get("candidate_final_answer"))
    if not answer:
        answer = _substantive(context.metadata.get("draft_answer"))
    if not answer:
        answer = wiki_answer_candidate(context.metadata.get("wiki"))
    if not answer:
        facts = context.metadata.get("extracted_facts", []) or []
        answer = " ".join(str(f.get("fact", "")).strip() for f in facts).strip()
    return answer or "unknown"


def _do_finish(context: Any, params: Dict[str, Any]) -> ToolObservation:
    answer = derive_final_answer(context, params)
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
    elif tool == "wiki_read":
        obs = _do_wiki_read(context)
    elif tool == "wiki_write":
        obs = _do_wiki_write(context, params)
    elif tool == "finish":
        obs = _do_finish(context, params)
    else:
        obs = ToolObservation(tool=tool or "unknown", status="invalid", data={"reason": "unknown_tool"})

    # Ingest any student-declared facts (validated against retrieved evidence) for
    # tools that do not themselves manage extraction.
    if tool not in {"extract"}:
        _ingest_facts(context, action)

    return obs.to_dict()
