"""
Leakage detection and sanitization for student-visible teacher guidance.

The teacher holds privileged information (gold answer, hidden gold titles/doc ids,
hidden supporting spans). This module enforces — in code, not by prompt — that none of
that leaks into the student-visible guidance, except for titles/spans the student has
already retrieved (which are fair game). It is meant to run *after* the guidance
renderer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def _boundary_pattern(needle: str) -> str:
    """Match ``needle`` only when it is not flanked by word characters.

    This prevents a short token like ``no`` from matching inside ``noted`` while still
    matching the standalone word ``no`` (including ``No.``), and works for phrases and
    doc ids alike.
    """
    return r"(?<!\w)" + re.escape(needle.strip()) + r"(?!\w)"


def _contains(haystack: str, needle: str) -> bool:
    if not needle or not needle.strip():
        return False
    return re.search(_boundary_pattern(needle), haystack, flags=re.IGNORECASE) is not None


def detect_leakage(
    text_or_obj: Any,
    gold_answer: str,
    hidden_titles: List[str],
    hidden_doc_ids: List[str],
    hidden_spans: List[str],
) -> Dict[str, Any]:
    """Return a leakage report for any string/dict/list payload."""
    combined = " \n ".join(_iter_strings(text_or_obj))

    matched: List[str] = []
    report = {
        "gold_answer_leaked": False,
        "hidden_doc_id_leaked": False,
        "hidden_title_leaked": False,
        "hidden_span_leaked": False,
        "matched_strings": matched,
        "sanitizations_applied": [],
    }

    if _contains(combined, gold_answer):
        report["gold_answer_leaked"] = True
        matched.append(gold_answer)
    for doc_id in hidden_doc_ids or []:
        if _contains(combined, doc_id):
            report["hidden_doc_id_leaked"] = True
            matched.append(doc_id)
    for title in hidden_titles or []:
        if _contains(combined, title):
            report["hidden_title_leaked"] = True
            matched.append(title)
    for span in hidden_spans or []:
        if _contains(combined, span):
            report["hidden_span_leaked"] = True
            matched.append(span)

    return report


def _sanitize_string(
    text: str,
    replacements: List[Tuple[str, str]],
    applied: List[str],
) -> str:
    for needle, placeholder in replacements:
        if needle and needle.strip() and _contains(text, needle):
            text = re.sub(_boundary_pattern(needle), placeholder, text, flags=re.IGNORECASE)
            applied.append(placeholder)
    return text


def _sanitize_obj(obj: Any, replacements, applied):
    if isinstance(obj, str):
        return _sanitize_string(obj, replacements, applied)
    if isinstance(obj, dict):
        return {k: _sanitize_obj(v, replacements, applied) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_obj(v, replacements, applied) for v in obj]
    return obj


def sanitize_rendered_guidance(
    rendered: Dict[str, Any],
    visibility: Dict[str, Any],
    config: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sanitize the rendered guidance and return ``(clean_rendered, leakage_report)``.

    ``visibility`` provides the gold/hidden values and the already-retrieved
    titles/doc-ids that are allowed to remain.
    """
    gold_answer = visibility.get("gold_answer", "") or ""
    retrieved_titles = {t.lower() for t in visibility.get("retrieved_titles", [])}
    retrieved_doc_ids = {d.lower() for d in visibility.get("retrieved_doc_ids", [])}

    # Hidden = gold values not yet retrieved by the student.
    hidden_titles = [
        t for t in visibility.get("gold_titles", []) if t.lower() not in retrieved_titles
    ]
    hidden_doc_ids = [
        d for d in visibility.get("gold_doc_ids", []) if d.lower() not in retrieved_doc_ids
    ]
    hidden_spans = visibility.get("hidden_spans", [])

    report = detect_leakage(rendered, gold_answer, hidden_titles, hidden_doc_ids, hidden_spans)

    leak_policy = getattr(config, "leak_policy", "strict")
    if leak_policy != "strict":
        return rendered, report

    replacements: List[Tuple[str, str]] = []
    if not getattr(config, "expose_gold_answer_hint", False) and gold_answer:
        replacements.append((gold_answer, "[answer hidden]"))
    for title in hidden_titles:
        replacements.append((title, "[title hidden]"))
    for doc_id in hidden_doc_ids:
        replacements.append((doc_id, "[doc hidden]"))
    for span in hidden_spans:
        replacements.append((span, "[span hidden]"))

    applied: List[str] = []
    clean = _sanitize_obj(rendered, replacements, applied)
    report["sanitizations_applied"] = applied
    return clean, report
