"""
Convert HotpotQA examples into Teacher Guidance question + corpus rows.

The core functions are dependency-free (no ``datasets`` import) so they can be unit
tested offline. ``scripts/prepare_hotpot_teacher_guidance.py`` is responsible for
loading examples (from the HF ``datasets`` library or a local raw JSON file) and
streaming the results to JSONL.

Two HotpotQA shapes are supported:

* HF columnar (``hotpotqa/hotpot_qa``)::

      context = {"title": [...], "sentences": [[...], ...]}
      supporting_facts = {"title": [...], "sent_id": [...]}

* Raw distractor JSON (official release)::

      context = [[title, [sent, ...]], ...]
      supporting_facts = [[title, sent_id], ...]
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _normalize_context(context: Any) -> List[Tuple[str, List[str]]]:
    """Return ``[(title, [sentence, ...]), ...]`` for either HotpotQA shape."""
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [
            (str(titles[i]), [str(s) for s in (sentences[i] if i < len(sentences) else [])])
            for i in range(len(titles))
        ]
    # Raw list shape: [[title, [sent, ...]], ...]
    result: List[Tuple[str, List[str]]] = []
    for entry in context or []:
        if not entry:
            continue
        title = str(entry[0])
        sents = [str(s) for s in (entry[1] if len(entry) > 1 and entry[1] else [])]
        result.append((title, sents))
    return result


def _normalize_supporting_facts(supporting_facts: Any) -> List[Tuple[str, int]]:
    """Return ``[(title, sent_id), ...]`` for either HotpotQA shape."""
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])
        return [
            (str(titles[i]), int(sent_ids[i]))
            for i in range(min(len(titles), len(sent_ids)))
        ]
    result: List[Tuple[str, int]] = []
    for entry in supporting_facts or []:
        if not entry:
            continue
        title = str(entry[0])
        sent_id = int(entry[1]) if len(entry) > 1 else 0
        result.append((title, sent_id))
    return result


def convert_example(
    example: Dict[str, Any], split: str = "validation"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert one HotpotQA example into (question_row, corpus_rows)."""
    qid = str(example.get("id") or example.get("_id") or "")
    question = str(example.get("question", ""))
    answer = str(example.get("answer", ""))
    qtype = str(example.get("type", ""))
    level = str(example.get("level", ""))

    context = _normalize_context(example.get("context"))
    facts = _normalize_supporting_facts(example.get("supporting_facts"))

    # Map title -> set of supporting sentence ids.
    gold_sent_by_title: Dict[str, List[int]] = {}
    for title, sent_id in facts:
        gold_sent_by_title.setdefault(title, [])
        if sent_id not in gold_sent_by_title[title]:
            gold_sent_by_title[title].append(sent_id)

    supporting_titles = list(gold_sent_by_title.keys())

    corpus_rows: List[Dict[str, Any]] = []
    candidate_doc_ids: List[str] = []
    gold_doc_ids: List[str] = []

    for idx, (title, sentences) in enumerate(context):
        doc_id = f"{qid}::doc{idx}"
        candidate_doc_ids.append(doc_id)
        is_gold = title in gold_sent_by_title
        gold_sent_ids = sorted(gold_sent_by_title.get(title, []))
        if is_gold:
            gold_doc_ids.append(doc_id)
        corpus_rows.append(
            {
                "doc_id": doc_id,
                "qid": qid,
                "title": title,
                "text": " ".join(sentences),
                "sentences": sentences,
                "is_gold_doc": is_gold,
                "gold_sent_ids": gold_sent_ids,
                "source": "hotpotqa",
                "split": split,
            }
        )

    question_row = {
        "id": qid,
        "query": question,
        "answer": answer,
        "type": qtype,
        "level": level,
        "gold": {
            "answer": answer,
            "supporting_titles": supporting_titles,
            "supporting_facts": [{"title": t, "sent_id": s} for t, s in facts],
            "gold_doc_ids": gold_doc_ids,
        },
        "retrieval_scope": {
            "backend": "hotpot_local",
            "qid": qid,
            "candidate_doc_ids": candidate_doc_ids,
        },
    }
    return question_row, corpus_rows


def convert_examples(
    examples: List[Dict[str, Any]], split: str = "validation"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert many examples into (question_rows, corpus_rows)."""
    question_rows: List[Dict[str, Any]] = []
    corpus_rows: List[Dict[str, Any]] = []
    for example in examples:
        q_row, c_rows = convert_example(example, split=split)
        question_rows.append(q_row)
        corpus_rows.extend(c_rows)
    return question_rows, corpus_rows
