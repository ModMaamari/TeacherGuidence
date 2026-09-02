"""
Metrics for Teacher Guidance episodes.

Answer scoring uses the standard SQuAD/HotpotQA normalization (lowercase, strip
punctuation, drop articles, collapse whitespace). Retrieval and evidence metrics are
deterministic set/recall computations so they can be used for filtering independent of
the teacher's interpretive judgement.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Set


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    def normalize_ampersand(s: str) -> str:
        # Treat "&" as the word "and" so "Medicare & Medicaid" == "Medicare and Medicaid".
        return s.replace("&", " and ")

    return white_space_fix(remove_articles(remove_punc(normalize_ampersand((text or "").lower()))))


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def _is_contiguous_span(needle: list, hay: list) -> bool:
    if not needle or len(needle) > len(hay):
        return False
    for i in range(len(hay) - len(needle) + 1):
        if hay[i : i + len(needle)] == needle:
            return True
    return False


def _has_number(tokens: list) -> bool:
    return any(any(ch.isdigit() for ch in tok) for tok in tokens)


def _looks_like_entity(raw_text: str) -> bool:
    """A short (<=3 char) gold token normally must *lead* the prediction to count
    (see the ``no`` / ``There is no clear winner`` guard below) -- that avoids
    incidental matches of common short words deep in an unrelated answer. But a short
    proper noun or acronym (``Ana``, ``CBS``) is distinctive enough to accept anywhere
    in the prediction: ordinary English words that short are essentially never
    capitalized when a gold answer is written as a bare entity name."""
    word = (raw_text or "").strip()
    return bool(word) and " " not in word and word[:1].isupper()


_FILLER_TOKENS = {
    # Pronoun + copula lead-ins ("He is the younger brother of X" vs a prediction that
    # names the actual subject instead of using a pronoun) and short name-linking
    # words dropped between components of a multi-word proper noun in another
    # language ("Club Atlético de Madrid" -> "...Atlético Madrid..."). None of these
    # carry the answer's actual content, so they shouldn't count against coverage.
    "he", "she", "it", "they", "is", "was", "are", "were", "has", "have", "been", "being", "be",
    "of", "de", "da", "van", "von", "der", "la", "le", "el", "di",
}

_COVERAGE_THRESHOLD = 0.6


def _content_tokens(tokens: list) -> list:
    filtered = [t for t in tokens if t not in _FILLER_TOKENS]
    return filtered if filtered else tokens


def _token_coverage(needle: list, hay: list) -> float:
    """Fraction of `needle`'s tokens found anywhere in `hay` (a bag/multiset match --
    order-independent, each hay token satisfies at most one needle token)."""
    if not needle:
        return 0.0
    hay_counts = Counter(hay)
    matched = 0
    for tok in needle:
        if hay_counts.get(tok, 0) > 0:
            hay_counts[tok] -= 1
            matched += 1
    return matched / len(needle)


def cover_match(pred: str, gold: str) -> bool:
    """Robust-but-cheap correctness, in both directions, without an LLM.

    Accepts when:

    * exact normalized equality, or
    * the gold answer appears as a contiguous token span in the prediction
      (the "answer + explanation" pattern: gold ``no`` vs ``No. Roger Donaldson ...``;
      a short yes/no-style gold must *lead* the prediction, unless it looks like a
      proper noun/acronym -- see ``_looks_like_entity`` -- in which case it may appear
      anywhere), or
    * most (>= 60%) of the gold's content tokens appear anywhere in the prediction,
      ignoring a small set of pronoun/copula/name-linking filler words (see
      ``_FILLER_TOKENS``) that often differ between a gold answer phrased as a full
      sentence or foreign-language name and a prediction that rephrases it -- this
      catches a dropped middle name (``Kelly Lee Osbourne`` vs ``Kelly Osbourne``), an
      inserted filler word (``born October 25, 1931`` vs ``born on October 25, 1931``),
      or a reordered generic word (``Club Atlético de Madrid`` vs ``Atlético
      Madrid ... the club``), or
    * the prediction is the salient core of a longer gold (gold ``22 episodes`` vs
      prediction ``22``): the prediction is a contiguous token span of the gold and is
      "salient" — it contains a number or covers at least half the gold tokens. This
      avoids accepting a partial entity (``York`` vs ``New York City``).
    """
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not g:
        return False
    if p == g:
        return True
    pt = p.split()
    gt = g.split()
    if not pt or not gt:
        return False

    # Direction 1: gold contained in the prediction.
    if len(gt) <= len(pt):
        if len(gt) == 1 and len(gt[0]) <= 3 and not _looks_like_entity(gold):
            if pt[0] == gt[0]:
                return True
        elif len(gt) == 1:
            if _is_contiguous_span(gt, pt):
                return True
        else:
            if _is_contiguous_span(gt, pt):
                return True
            gt_content = _content_tokens(gt)
            if len(gt_content) >= 2 and _token_coverage(gt_content, pt) >= _COVERAGE_THRESHOLD:
                return True

    # Direction 2: prediction is the salient core of a longer gold.
    if len(pt) < len(gt) and _is_contiguous_span(pt, gt):
        if _has_number(pt) or 2 * len(pt) >= len(gt):
            return True

    # Direction 3: a single-token gold is a strong prefix of a prediction token
    # (gold "KXII" vs prediction token "KXII-TV" -> "kxiitv"). Requires the shorter to
    # be a prefix of the longer and to cover >= 60% of it, so "KXII"/"KXII-TV" matches
    # but "Bart"/"Bartholomew" does not.
    if len(gt) == 1 and len(gt[0]) >= 4:
        gtok = gt[0]
        for tok in pt:
            shorter, longer = sorted([gtok, tok], key=len)
            if longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
                return True

    return False


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_doc_recall(retrieved_doc_ids: Set[str], gold_doc_ids: Set[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return len(gold & set(retrieved_doc_ids)) / len(gold)


def supporting_fact_recall(
    extracted_spans: List[Dict[str, Any]],
    gold_facts: List[Dict[str, Any]],
    corpus: Dict[str, Dict[str, Any]],
) -> float:
    """Fraction of gold (title, sent_id) facts whose sentence text appears verbatim in
    an extracted span.

    ``corpus`` maps doc_id -> doc dict (with ``title`` and ``sentences``).
    """
    gold_facts = gold_facts or []
    if not gold_facts:
        return 0.0

    # Build title -> {sent_id: sentence_text}
    sentences_by_title: Dict[str, Dict[int, str]] = {}
    for doc in corpus.values():
        title = doc.get("title", "")
        sentences_by_title.setdefault(title, {})
        for idx, sent in enumerate(doc.get("sentences", [])):
            sentences_by_title[title][idx] = sent

    extracted_text = " \n ".join(s.get("span", "") for s in (extracted_spans or []))

    covered = 0
    for fact in gold_facts:
        title = fact.get("title", "")
        sent_id = fact.get("sent_id", 0)
        sent_text = sentences_by_title.get(title, {}).get(sent_id)
        if sent_text and sent_text.strip() and sent_text.strip() in extracted_text:
            covered += 1
    return covered / len(gold_facts)


def answer_grounding(
    final_answer: str,
    evidence_text: str,
    *,
    supporting_doc_recall: float = 0.0,
    has_extracted_facts: bool = False,
    threshold: float = 0.5,
) -> tuple:
    """Is the final answer supported by evidence the student actually gathered, rather
    than produced from parametric memory? Returns ``(score, grounded_bool)``.

    For a lexical answer (entities/dates/phrases) the score is the fraction of the
    answer's *content* tokens (fillers dropped) that appear in ``evidence_text`` (the
    concatenation of the student's extracted spans and retrieved doc text). A grounded
    answer is one the retrieved evidence could actually justify -- exactly the behavior
    we want the fine-tuned student to learn, and a general signal (it keys on token
    overlap with *whatever* was retrieved, never on the specific gold answer).

    For a non-lexical answer with no substantive content tokens (yes/no and other short
    verdicts, where token overlap is meaningless) grounding instead requires that the
    evidence base was genuinely built -- the gold supporting docs were retrieved or at
    least some facts were extracted -- so such traces are neither auto-passed nor unfairly
    rejected.
    """
    norm = normalize_answer(final_answer)
    content = [t for t in _content_tokens(norm.split()) if t not in _FILLER_TOKENS]
    substantive = [t for t in content if len(t) > 1]
    # Boolean/verdict answers ("yes"/"no") carry no lexical content to match against the
    # evidence, so token overlap is meaningless -- fall back to the evidence base.
    if substantive and norm not in {"yes", "no"}:
        score = _token_coverage(substantive, normalize_answer(evidence_text).split())
        return round(score, 4), score >= threshold
    grounded = supporting_doc_recall >= 0.999 or has_extracted_facts
    return (1.0 if grounded else 0.0), grounded


def binary_from_continuous(score_continuous: float, threshold: float = 0.75) -> int:
    return 1 if score_continuous >= threshold else 0


def compute_step_metrics(
    student_action: Dict[str, Any],
    tool_observation: Dict[str, Any],
    parse_info: Optional[Dict[str, Any]] = None,
    retrieved_doc_ids: Optional[Set[str]] = None,
    gold_doc_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Deterministic per-step labels."""
    parse_info = parse_info or {}
    observation = tool_observation or {}
    action = (student_action or {}).get("action", {})

    metrics: Dict[str, Any] = {
        "json_valid": bool(parse_info.get("json_valid", False)),
        "action_schema_valid": bool(parse_info.get("action_valid", False)),
        "tool": action.get("tool"),
        "tool_status": observation.get("status"),
        "invalid_action": observation.get("status") in {"error", "invalid"},
        "span_validation_failed": bool(observation.get("invalid_spans")),
        "parametric_knowledge_used": bool(
            (student_action or {}).get("decision", {}).get("parametric_knowledge_used", False)
        ),
    }
    if retrieved_doc_ids is not None and gold_doc_ids is not None:
        metrics["retrieved_gold_doc"] = bool(set(retrieved_doc_ids) & set(gold_doc_ids))
        metrics["supporting_doc_recall_so_far"] = supporting_doc_recall(
            set(retrieved_doc_ids), set(gold_doc_ids)
        )
    return metrics


def compute_final_metrics(
    final_answer: str,
    gold_answer: str,
    retrieved_doc_ids: Set[str],
    gold_doc_ids: Set[str],
    extracted_spans: List[Dict[str, Any]],
    gold_facts: List[Dict[str, Any]],
    corpus: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    doc_recall = supporting_doc_recall(set(retrieved_doc_ids), set(gold_doc_ids))

    # Evidence the student actually gathered: its extracted spans plus the text of the
    # docs it retrieved (never gold docs it failed to retrieve). Used only for grounding.
    evidence_parts = [str(s.get("span", "") or "") for s in (extracted_spans or [])]
    for did in retrieved_doc_ids:
        doc = corpus.get(did, {}) or {}
        evidence_parts.append(doc.get("text", "") or " ".join(doc.get("sentences", []) or []))
    grounded_score, grounded = answer_grounding(
        final_answer, " \n ".join(evidence_parts),
        supporting_doc_recall=doc_recall, has_extracted_facts=bool(extracted_spans),
    )

    return {
        "exact_match": exact_match(final_answer, gold_answer),
        "answer_correct": cover_match(final_answer, gold_answer),
        "f1": round(f1_score(final_answer, gold_answer), 4),
        "supporting_doc_recall": round(doc_recall, 4),
        "supporting_fact_recall": round(
            supporting_fact_recall(extracted_spans, gold_facts, corpus), 4
        ),
        "answer_grounded": grounded,
        "answer_grounded_score": grounded_score,
    }
