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

    return white_space_fix(remove_articles(remove_punc((text or "").lower())))


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


def cover_match(pred: str, gold: str) -> bool:
    """Robust-but-cheap correctness, in both directions, without an LLM.

    Accepts when:

    * exact normalized equality, or
    * the gold answer appears as a contiguous token span in the prediction
      (the "answer + explanation" pattern: gold ``no`` vs ``No. Roger Donaldson ...``;
      a short yes/no-style gold must *lead* the prediction), or
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
        if len(gt) == 1 and len(gt[0]) <= 3:
            if pt[0] == gt[0]:
                return True
        elif _is_contiguous_span(gt, pt):
            return True

    # Direction 2: prediction is the salient core of a longer gold.
    if len(pt) < len(gt) and _is_contiguous_span(pt, gt):
        if _has_number(pt) or 2 * len(pt) >= len(gt):
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
    return {
        "exact_match": exact_match(final_answer, gold_answer),
        "answer_correct": cover_match(final_answer, gold_answer),
        "f1": round(f1_score(final_answer, gold_answer), 4),
        "supporting_doc_recall": round(
            supporting_doc_recall(set(retrieved_doc_ids), set(gold_doc_ids)), 4
        ),
        "supporting_fact_recall": round(
            supporting_fact_recall(extracted_spans, gold_facts, corpus), 4
        ),
    }
