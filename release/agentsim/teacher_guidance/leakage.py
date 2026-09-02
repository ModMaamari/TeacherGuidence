"""
Leakage detection and sanitization for student-visible teacher guidance.

The teacher holds privileged information (gold answer, hidden gold titles/doc ids,
hidden supporting spans). The only thing that must never reach the student is the
**gold answer itself** — and even that is fair game when it already appears in the
question (a comparison/boolean question whose answer is one of the entities the student
was handed). Gold titles, doc ids, and supporting spans are *not* redacted: they are
routinely the very entities named in the question, or docs the student has already
retrieved, so hiding them mangled legitimate teacher feedback (e.g. restating the
question turned into ``[title hidden]``). This module still *detects* those mentions for
telemetry, but only the gold answer is ever removed. It runs *after* the guidance
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


def asserts_gold_answer(text: str, gold_answer: str) -> bool:
    """True when ``text`` states the gold answer, as opposed to merely naming its format.

    For a boolean answer the value space is public -- the student can see it is a yes/no
    question -- so the secret is *which* one, not the words themselves. A verdict saying
    "you didn't answer the yes/no question" asserts nothing, and treating it as a leak
    both inflates the reported leak rate and throws away sound training examples.
    """
    if not gold_answer or not gold_answer.strip():
        return False
    if gold_answer.strip().lower() in BOOLEAN_ANSWERS:
        text = _YESNO_DISJUNCTION.sub(" ", text)
    return _contains(text, gold_answer)


def detect_leakage(
    text_or_obj: Any,
    gold_answer: str,
    hidden_titles: List[str],
    hidden_doc_ids: List[str],
    hidden_spans: List[str],
    question: str = "",
) -> Dict[str, Any]:
    """Return a leakage report for any string/dict/list payload.

    ``question`` (when supplied) suppresses a gold-answer "leak": if the answer already
    appears in the question (a comparison/boolean question whose answer is one of the
    entities the student was handed), the teacher echoing it reveals nothing new.
    """
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

    answer_in_question = _contains(question, gold_answer)
    if asserts_gold_answer(combined, gold_answer) and not answer_in_question:
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


#: Answers whose value space is public knowledge: the student can see the question is a
#: yes/no one, so the word itself is not the secret -- only an assertion of which.
BOOLEAN_ANSWERS = {"yes", "no", "true", "false"}

#: "yes/no question", "yes or no" -- these name the answer FORMAT and say nothing about
#: which one is correct. Redacting inside them is doubly wrong: it destroys legitimate
#: feedback, and the surviving half ("[answer hidden]/no") gives the answer away by
#: position. Observed live: a Kimi-K3 verdict "doesn't answer the yes/no comparison
#: question" was rewritten to "the [answer hidden]/no comparison question".
_YESNO_DISJUNCTION = re.compile(
    r"(?<!\w)(?:yes|no|true|false)\s*(?:/|-|\s+or\s+|\s*/\s*)\s*(?:yes|no|true|false)(?!\w)",
    flags=re.IGNORECASE,
)
_DISJUNCTION_GUARD = "\x00DISJ{}\x00"


def _sanitize_string(
    text: str,
    replacements: List[Tuple[str, str]],
    applied: List[str],
) -> str:
    for needle, placeholder in replacements:
        if not (needle and needle.strip() and _contains(text, needle)):
            continue
        # For a boolean answer, shield yes/no disjunctions from redaction first.
        guarded: List[str] = []
        if needle.strip().lower() in BOOLEAN_ANSWERS:
            def _guard(match: "re.Match[str]") -> str:
                guarded.append(match.group(0))
                return _DISJUNCTION_GUARD.format(len(guarded) - 1)

            text = _YESNO_DISJUNCTION.sub(_guard, text)
            if not _contains(text, needle):
                # Every mention was part of a disjunction -- nothing was ever asserted.
                for i, original in enumerate(guarded):
                    text = text.replace(_DISJUNCTION_GUARD.format(i), original)
                continue

        text = re.sub(_boundary_pattern(needle), placeholder, text, flags=re.IGNORECASE)
        applied.append(placeholder)
        for i, original in enumerate(guarded):
            text = text.replace(_DISJUNCTION_GUARD.format(i), original)
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
    question = visibility.get("question", "") or ""
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

    report = detect_leakage(
        rendered, gold_answer, hidden_titles, hidden_doc_ids, hidden_spans, question=question
    )

    leak_policy = getattr(config, "leak_policy", "strict")
    if leak_policy != "strict":
        return rendered, report

    # Only the gold answer is ever redacted -- and not even that when it already appears
    # in the question (the teacher echoing a question entity reveals nothing). Titles,
    # doc ids, and spans are still reported above for telemetry, but are considered fair
    # game (question entities / already-retrieved docs) and left in place.
    answer_in_question = _contains(question, gold_answer)
    replacements: List[Tuple[str, str]] = []
    if not getattr(config, "expose_gold_answer_hint", False) and gold_answer and not answer_in_question:
        replacements.append((gold_answer, "[answer hidden]"))

    applied: List[str] = []
    clean = _sanitize_obj(rendered, replacements, applied)
    report["sanitizations_applied"] = applied
    return clean, report
