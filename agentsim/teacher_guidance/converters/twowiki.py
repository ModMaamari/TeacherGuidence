"""2WikiMultihopQA -> canonical teacher-guidance rows.

Input shape (official release JSON)::

    {"_id", "type", "question", "answer",
     "context": [[title, [sent, ...]], ...],
     "supporting_facts": [[title, sent_id], ...],
     "evidences": [[subject, relation, object], ...]}

Sentence-level gold, like HotpotQA, so ``supporting_facts`` is populated.

Two things distinguish it from HotpotQA and are preserved here:

* **Deeper reasoning types.** ``type`` is one of ``comparison``, ``inference``,
  ``compositional``, ``bridge_comparison``. ``bridge_comparison`` composes a bridge with a
  comparison and is 4-hop by construction; the others are 2-hop. ``num_hops`` is derived
  from the type on that basis (the source does not state a hop count per example).
* **Structured evidence triples.** ``evidences`` gives (subject, relation, object) triples
  behind each answer -- a second, symbolic gold signal that HotpotQA lacks. They are
  carried through on the question row as ``evidence_triples`` for downstream research;
  nothing in the simulation engine reads them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentsim.teacher_guidance.converters.base import build_corpus_row, build_question_row
from agentsim.teacher_guidance.converters.hotpot import (
    normalize_context,
    normalize_supporting_facts,
)

SOURCE = "2wikimultihopqa"

#: Hop count implied by the composition pattern each type is built from.
_HOPS_BY_TYPE = {
    "comparison": 2,
    "inference": 2,
    "compositional": 2,
    "bridge_comparison": 4,
}


def _evidence_triples(example: Dict[str, Any]) -> List[List[str]]:
    triples: List[List[str]] = []
    for ev in example.get("evidences") or []:
        if isinstance(ev, (list, tuple)) and len(ev) >= 3:
            triples.append([str(ev[0]), str(ev[1]), str(ev[2])])
    return triples


def convert_example(
    example: Dict[str, Any], split: str = "train"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert one 2WikiMultihopQA example into ``(question_row, corpus_rows)``."""
    qid = str(example.get("_id") or example.get("id") or "")
    # The context/supporting_facts encodings are identical to HotpotQA's raw shape.
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

    by_title = {d["title"]: d for d in corpus_rows}
    resolvable = [
        {"title": t, "sent_id": s}
        for t, s in facts
        if t in by_title and 0 <= s < len(by_title[t]["sentences"])
    ]

    qtype = str(example.get("type", ""))
    question_row = build_question_row(
        qid=qid,
        query=str(example.get("question", "")),
        answer=example.get("answer", ""),
        source=SOURCE,
        split=split,
        corpus_rows=corpus_rows,
        gold_granularity="sentence",
        qtype=qtype,
        level="",  # 2Wiki ships no difficulty label
        num_hops=_HOPS_BY_TYPE.get(qtype),
        supporting_facts=resolvable,
    )
    triples = _evidence_triples(example)
    if triples:
        question_row["evidence_triples"] = triples
    return question_row, corpus_rows
