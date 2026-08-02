"""StrategyQA -> canonical teacher-guidance rows.

Input shape (official release)::

    {"qid", "term", "description", "question", "answer": true|false,
     "facts": [...], "decomposition": [...],
     "evidence": [ annotator -> [ step -> [ evidence_set -> ["para_id", ...] | "operation" ] ] ]}

plus a separate paragraph corpus ``strategyqa_*_paragraphs.json``::

    {"para_id": {"title", "section", "headers", "content"}}

StrategyQA differs from the other sources in three ways that this converter makes explicit:

* **Boolean answers.** ``answer`` is a bool and becomes the string ``"yes"``/``"no"``
  (``answer_type == "boolean"``) so the shared SQuAD-style normalization applies.
* **Paragraph-level gold**, so ``gold_granularity == "paragraph"`` and no
  ``supporting_facts`` are emitted (same rule as MuSiQue).
* **No distractors ship with the dataset.** Every other source gives a fixed candidate set
  per question; StrategyQA gives only the gold evidence paragraphs. To place it in the same
  retrieval setting we *construct* a candidate set: the gold paragraphs plus
  ``num_distractors`` paragraphs sampled from the corpus, seeded by the qid so the result is
  deterministic and reproducible. **This is a transformation of the source dataset and must
  be disclosed in the dataset card** -- it is the one place where our corpus is not simply a
  re-encoding of the original.

Questions whose evidence contains no usable paragraph ids (annotators marked the step
``operation`` or ``no_evidence`` throughout) are rejected: without gold documents the
episode is unanswerable under this protocol.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agentsim.teacher_guidance.converters.base import (
    build_corpus_row,
    build_question_row,
    split_sentences,
)

SOURCE = "strategyqa"

#: Sentinel values that appear in the evidence tree in place of paragraph ids.
_NON_EVIDENCE = {"operation", "no_evidence"}

DEFAULT_NUM_DISTRACTORS = 8


def collect_evidence_ids(evidence: Any) -> List[str]:
    """Flatten StrategyQA's nested evidence tree into unique paragraph ids, in order.

    The tree is annotator -> decomposition step -> evidence set -> paragraph ids, but the
    nesting depth varies and leaves may be the ``operation``/``no_evidence`` sentinels, so
    this walks the structure generically rather than assuming a fixed depth.
    """
    found: List[str] = []
    seen = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if node and node not in _NON_EVIDENCE and node not in seen:
                seen.add(node)
                found.append(node)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(evidence)
    return found


def _paragraph_sentences(para: Dict[str, Any]) -> List[str]:
    return split_sentences(str(para.get("content", "") or ""))


def convert_example(
    example: Dict[str, Any],
    paragraphs: Dict[str, Dict[str, Any]],
    split: str = "train",
    num_distractors: int = DEFAULT_NUM_DISTRACTORS,
    distractor_pool: Optional[Sequence[str]] = None,
    seed: int = 13,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Convert one StrategyQA example, or ``None`` when it has no usable gold evidence.

    ``paragraphs`` is the full paragraph corpus; ``distractor_pool`` defaults to all of its
    ids and should normally be passed in pre-computed by the caller (it is the same for
    every question, and re-deriving it per question is wasteful).
    """
    qid = str(example.get("qid") or example.get("id") or "")
    gold_ids = [pid for pid in collect_evidence_ids(example.get("evidence")) if pid in paragraphs]
    gold_ids = [pid for pid in gold_ids if _paragraph_sentences(paragraphs[pid])]
    if not gold_ids:
        return None

    pool = list(distractor_pool if distractor_pool is not None else paragraphs.keys())
    rng = random.Random(f"{seed}:{qid}")
    gold_set = set(gold_ids)
    distractors: List[str] = []
    if num_distractors > 0 and pool:
        # Sample with rejection rather than filtering the whole pool per question: the pool
        # has tens of thousands of ids and the gold set is tiny.
        attempts = 0
        max_attempts = num_distractors * 50
        while len(distractors) < num_distractors and attempts < max_attempts:
            attempts += 1
            pid = pool[rng.randrange(len(pool))]
            if pid in gold_set or pid in distractors:
                continue
            if not _paragraph_sentences(paragraphs.get(pid, {})):
                continue
            distractors.append(pid)

    # Interleave deterministically so gold documents are not always first.
    candidates = gold_ids + distractors
    rng.shuffle(candidates)

    corpus_rows = []
    for idx, pid in enumerate(candidates):
        para = paragraphs[pid]
        sentences = _paragraph_sentences(para)
        corpus_rows.append(
            build_corpus_row(
                qid=qid,
                index=idx,
                title=str(para.get("title", "") or pid),
                sentences=sentences,
                is_gold_doc=pid in gold_set,
                gold_sent_ids=[],
                source=SOURCE,
                split=split,
                text=str(para.get("content", "") or ""),
            )
        )
        # Preserve the original paragraph id so a row can be traced back to the source.
        corpus_rows[-1]["source_doc_id"] = pid

    question_row = build_question_row(
        qid=qid,
        query=str(example.get("question", "")),
        answer=example.get("answer"),
        source=SOURCE,
        split=split,
        corpus_rows=corpus_rows,
        gold_granularity="paragraph",
        qtype="strategy",
        level="",
        # StrategyQA's decomposition length is the number of implicit reasoning steps.
        num_hops=len(example.get("decomposition") or []) or None,
        supporting_facts=None,
    )
    if example.get("facts"):
        question_row["gold_facts_text"] = [str(f) for f in example["facts"]]
    if example.get("decomposition"):
        question_row["question_decomposition"] = [
            {"question": str(d), "answer": "", "paragraph_support_idx": None}
            for d in example["decomposition"]
        ]
    question_row["constructed_candidates"] = True
    return question_row, corpus_rows
