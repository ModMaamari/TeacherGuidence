"""HotpotQA -> canonical teacher-guidance rows.

Sentence-level gold: HotpotQA annotates supporting *sentences*, so
``gold_granularity == "sentence"`` and ``supporting_facts`` is populated.

Two input shapes are supported:

* HF columnar (``hotpotqa/hotpot_qa``)::

      context = {"title": [...], "sentences": [[...], ...]}
      supporting_facts = {"title": [...], "sent_id": [...]}

* Raw distractor JSON (official release)::

      context = [[title, [sent, ...]], ...]
      supporting_facts = [[title, sent_id], ...]
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentsim.teacher_guidance.converters.base import build_corpus_row, build_question_row

SOURCE = "hotpotqa"

# HotpotQA is 2-hop by construction (bridge or comparison over two gold paragraphs).
_NUM_HOPS = 2


def normalize_context(context: Any) -> List[Tuple[str, List[str]]]:
    """Return ``[(title, [sentence, ...]), ...]`` for either HotpotQA shape."""
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [
            (str(titles[i]), [str(s) for s in (sentences[i] if i < len(sentences) else [])])
            for i in range(len(titles))
        ]
    result: List[Tuple[str, List[str]]] = []
    for entry in context or []:
        if not entry:
            continue
        title = str(entry[0])
        sents = [str(s) for s in (entry[1] if len(entry) > 1 and entry[1] else [])]
        result.append((title, sents))
    return result


def normalize_supporting_facts(supporting_facts: Any) -> List[Tuple[str, int]]:
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
        result.append((str(entry[0]), int(entry[1]) if len(entry) > 1 else 0))
    return result


def convert_example(
    example: Dict[str, Any], split: str = "validation"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert one HotpotQA example into ``(question_row, corpus_rows)``."""
    qid = str(example.get("id") or example.get("_id") or "")
    context = normalize_context(example.get("context"))
    facts = normalize_supporting_facts(example.get("supporting_facts"))

    gold_sent_by_title: Dict[str, List[int]] = {}
    for title, sent_id in facts:
        gold_sent_by_title.setdefault(title, [])
        if sent_id not in gold_sent_by_title[title]:
            gold_sent_by_title[title].append(sent_id)

    corpus_rows = [
        build_corpus_row(
            qid=qid,
            index=idx,
            title=title,
            sentences=sentences,
            is_gold_doc=title in gold_sent_by_title,
            gold_sent_ids=gold_sent_by_title.get(title, []),
            source=SOURCE,
            split=split,
        )
        for idx, (title, sentences) in enumerate(context)
    ]

    # Only keep facts that resolve to a real sentence in the shipped context; HotpotQA has
    # a handful of examples whose supporting_facts reference an out-of-range sentence.
    by_title = {d["title"]: d for d in corpus_rows}
    resolvable = [
        {"title": t, "sent_id": s}
        for t, s in facts
        if t in by_title and 0 <= s < len(by_title[t]["sentences"])
    ]

    question_row = build_question_row(
        qid=qid,
        query=str(example.get("question", "")),
        answer=example.get("answer", ""),
        source=SOURCE,
        split=split,
        corpus_rows=corpus_rows,
        gold_granularity="sentence",
        qtype=str(example.get("type", "")),
        level=str(example.get("level", "")),
        num_hops=_NUM_HOPS,
        supporting_facts=resolvable,
    )
    return question_row, corpus_rows
