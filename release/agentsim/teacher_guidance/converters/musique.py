"""MuSiQue -> canonical teacher-guidance rows.

Input shape (MuSiQue-Ans / MuSiQue-Full JSONL)::

    {"id": "2hop__12345_6789",
     "paragraphs": [{"idx", "title", "paragraph_text", "is_supporting"}, ...],
     "question", "answer", "answer_aliases": [...],
     "question_decomposition": [{"id", "question", "answer", "paragraph_support_idx"}, ...],
     "answerable": bool}

Three differences from HotpotQA/2Wiki, all handled explicitly:

* **Paragraph-level gold.** ``is_supporting`` marks whole paragraphs; MuSiQue never says
  which sentence carries the fact. We therefore set ``gold_granularity="paragraph"`` and
  emit NO ``supporting_facts`` rather than inventing sentence ids -- fabricating them
  would silently corrupt ``supporting_fact_recall``.
* **Paragraphs arrive as one string**, so they are sentence-split deterministically
  (``base.split_sentences``) to populate the ``sentences`` field the retriever and the
  ``extract`` tool expect.
* **Answer aliases** are provided and are carried into ``gold.answer_aliases``.

Unanswerable items (``answerable == False``, present in MuSiQue-Full) are rejected by
:func:`convert_example` returning ``None`` -- the teacher-guidance protocol assumes a
reachable gold answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agentsim.teacher_guidance.converters.base import (
    build_corpus_row,
    build_question_row,
    split_sentences,
)

SOURCE = "musique"

#: MuSiQue ids look like "2hop__12345_6789" / "3hop1__..." / "4hop3__...".
_ID_PATTERN = re.compile(r"^(?P<hops>\d+)hop(?P<variant>\d*)__")


def _hops_and_type(qid: str, decomposition: Any) -> Tuple[Optional[int], str]:
    """Hop count and composition pattern.

    Prefer the explicit decomposition length; fall back to the id prefix. The ``type`` is
    the id's composition tag (e.g. ``2hop``, ``3hop1``), which encodes *how* the single-hop
    questions were composed -- a genuine reasoning-structure label worth keeping.
    """
    match = _ID_PATTERN.match(qid or "")
    qtype = ""
    hops: Optional[int] = None
    if match:
        qtype = f"{match.group('hops')}hop{match.group('variant')}"
        hops = int(match.group("hops"))
    if isinstance(decomposition, list) and decomposition:
        hops = len(decomposition)
    return hops, qtype


def convert_example(
    example: Dict[str, Any], split: str = "train"
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Convert one MuSiQue example, or return ``None`` if it is unanswerable."""
    if example.get("answerable") is False:
        return None

    qid = str(example.get("id") or "")
    paragraphs = example.get("paragraphs") or []

    corpus_rows = []
    for idx, para in enumerate(paragraphs):
        text = str(para.get("paragraph_text", "") or "")
        sentences = split_sentences(text)
        if not sentences:
            continue
        corpus_rows.append(
            build_corpus_row(
                qid=qid,
                index=idx,
                title=str(para.get("title", "") or ""),
                sentences=sentences,
                is_gold_doc=bool(para.get("is_supporting")),
                # paragraph-level annotation: no sentence ids exist to record
                gold_sent_ids=[],
                source=SOURCE,
                split=split,
                text=text,
            )
        )

    decomposition = example.get("question_decomposition")
    num_hops, qtype = _hops_and_type(qid, decomposition)

    question_row = build_question_row(
        qid=qid,
        query=str(example.get("question", "")),
        answer=example.get("answer", ""),
        source=SOURCE,
        split=split,
        corpus_rows=corpus_rows,
        gold_granularity="paragraph",
        qtype=qtype,
        level="",
        num_hops=num_hops,
        answer_aliases=example.get("answer_aliases") or [],
        supporting_facts=None,
    )
    # The gold decomposition is a strong supervision signal in its own right (each hop's
    # sub-question and answer); keep it on the question row for downstream research.
    if isinstance(decomposition, list) and decomposition:
        question_row["question_decomposition"] = [
            {
                "question": str(d.get("question", "")),
                "answer": str(d.get("answer", "")),
                "paragraph_support_idx": d.get("paragraph_support_idx"),
            }
            for d in decomposition
        ]
    return question_row, corpus_rows
