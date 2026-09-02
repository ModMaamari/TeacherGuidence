"""Canonical question/corpus schema shared by every source QA dataset.

Every converter emits exactly these two row shapes, so the simulation engine, the local
retriever, the metrics and the release packager never need to know which dataset a
question came from. Adding a dataset means writing a converter that produces these rows --
nothing downstream changes.

Question row::

    {
      "id", "query", "answer",            # answer is ALWAYS a string ("yes"/"no" for boolean)
      "type", "level",                    # source-specific reasoning type / difficulty ("" if absent)
      "source", "split",                  # provenance -- which dataset and split this came from
      "num_hops",                         # int when the source states it, else None
      "answer_type",                      # "span" | "boolean"
      "gold_granularity",                 # "sentence" | "paragraph"  (see note below)
      "gold": {
        "answer", "answer_aliases",
        "supporting_titles",
        "supporting_facts": [{"title", "sent_id"}],   # EMPTY when gold_granularity == "paragraph"
        "gold_doc_ids"
      },
      "retrieval_scope": {"backend", "qid", "candidate_doc_ids"}
    }

Corpus row::

    {"doc_id", "qid", "title", "text", "sentences",
     "is_gold_doc", "gold_sent_ids", "source", "split"}

**Gold granularity is a real difference between datasets, not a defect.** HotpotQA and
2WikiMultihopQA annotate supporting *sentences*; MuSiQue and StrategyQA annotate supporting
*paragraphs*. We record which one applies rather than fabricating sentence ids that the
source never provided -- inventing them would silently corrupt ``supporting_fact_recall``.
For paragraph-level datasets ``supporting_facts`` is empty and that metric is reported as
``None`` (not 0.0, which would read as "recalled nothing").

``retrieval_scope.backend`` is always ``hotpot_local``: it names the retrieval *mechanism*
(per-question BM25 over that question's candidate documents), not the dataset.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Bump when the emitted row shapes change in a backward-incompatible way.
DATASET_SCHEMA_VERSION = "1.0"

#: Retrieval mechanism id (per-question local distractor retrieval), not a dataset name.
LOCAL_BACKEND = "hotpot_local"

ANSWER_TYPES = ("span", "boolean")
GOLD_GRANULARITIES = ("sentence", "paragraph")

_QUESTION_REQUIRED = (
    "id", "query", "answer", "type", "level", "source", "split",
    "num_hops", "answer_type", "gold_granularity", "gold", "retrieval_scope",
)
_GOLD_REQUIRED = (
    "answer", "answer_aliases", "supporting_titles", "supporting_facts", "gold_doc_ids",
)
_CORPUS_REQUIRED = (
    "doc_id", "qid", "title", "text", "sentences",
    "is_gold_doc", "gold_sent_ids", "source", "split",
)

# Sentence splitter: split after . ! ? when followed by whitespace + an uppercase/quote/digit
# start. Deliberately dependency-free and deterministic so converter output is reproducible
# without nltk. Common abbreviations are protected to avoid splitting "Dr. Who".
_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co",
    "no", "vol", "op", "fig", "al", "ca", "approx", "dept", "est", "gen", "gov", "capt",
)
_ABBREV_RE = re.compile(r"\b(" + "|".join(_ABBREV) + r")\.$", re.IGNORECASE)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+(?=[\"'“‘(\[]?[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Deterministic sentence split for sources that ship paragraphs as one string.

    Never returns empty strings; returns ``[]`` for blank input. A fragment ending in a
    known abbreviation is re-joined with the next fragment so titles like "Dr." don't
    create spurious sentences.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_BOUNDARY.split(text)
    merged: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def normalize_answer_text(answer: Any) -> Tuple[str, str]:
    """Return ``(answer_string, answer_type)``.

    Booleans (StrategyQA) become the strings ``"yes"``/``"no"`` so that every dataset's
    answer is a string and the standard SQuAD-style normalization in ``metrics.py`` applies
    uniformly. Everything else is treated as a span.
    """
    if isinstance(answer, bool):
        return ("yes" if answer else "no"), "boolean"
    text = str(answer if answer is not None else "").strip()
    if text.lower() in ("true", "false"):
        return ("yes" if text.lower() == "true" else "no"), "boolean"
    if text.lower() in ("yes", "no"):
        return text.lower(), "boolean"
    return text, "span"


def build_corpus_row(
    *,
    qid: str,
    index: int,
    title: str,
    sentences: Sequence[str],
    is_gold_doc: bool,
    gold_sent_ids: Sequence[int],
    source: str,
    split: str,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """One canonical corpus row. ``doc_id`` is always ``<qid>::doc<index>``."""
    sents = [str(s) for s in sentences if str(s).strip()]
    return {
        "doc_id": f"{qid}::doc{index}",
        "qid": str(qid),
        "title": str(title),
        "text": text if text is not None else " ".join(sents),
        "sentences": sents,
        "is_gold_doc": bool(is_gold_doc),
        "gold_sent_ids": sorted({int(i) for i in gold_sent_ids}),
        "source": str(source),
        "split": str(split),
    }


def build_question_row(
    *,
    qid: str,
    query: str,
    answer: Any,
    source: str,
    split: str,
    corpus_rows: Sequence[Dict[str, Any]],
    gold_granularity: str,
    qtype: str = "",
    level: str = "",
    num_hops: Optional[int] = None,
    answer_aliases: Optional[Iterable[str]] = None,
    supporting_facts: Optional[Sequence[Dict[str, Any]]] = None,
    answer_type: Optional[str] = None,
) -> Dict[str, Any]:
    """One canonical question row, derived from its own corpus rows.

    ``gold_doc_ids``, ``supporting_titles`` and ``candidate_doc_ids`` are computed from
    ``corpus_rows`` so a question can never disagree with its own corpus.
    """
    if gold_granularity not in GOLD_GRANULARITIES:
        raise ValueError(f"gold_granularity must be one of {GOLD_GRANULARITIES}")
    answer_text, inferred_type = normalize_answer_text(answer)
    answer_type = answer_type or inferred_type
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"answer_type must be one of {ANSWER_TYPES}")

    gold_docs = [d for d in corpus_rows if d.get("is_gold_doc")]
    facts = [dict(f) for f in (supporting_facts or [])]
    if gold_granularity == "paragraph" and facts:
        raise ValueError(
            "paragraph-level datasets must not emit supporting_facts "
            "(sentence ids the source never provided)"
        )
    return {
        "id": str(qid),
        "query": str(query),
        "answer": answer_text,
        "type": str(qtype or ""),
        "level": str(level or ""),
        "source": str(source),
        "split": str(split),
        "num_hops": int(num_hops) if num_hops is not None else None,
        "answer_type": answer_type,
        "gold_granularity": gold_granularity,
        "gold": {
            "answer": answer_text,
            "answer_aliases": sorted({str(a).strip() for a in (answer_aliases or []) if str(a).strip()}),
            "supporting_titles": [d["title"] for d in gold_docs],
            "supporting_facts": facts,
            "gold_doc_ids": [d["doc_id"] for d in gold_docs],
        },
        "retrieval_scope": {
            "backend": LOCAL_BACKEND,
            "qid": str(qid),
            "candidate_doc_ids": [d["doc_id"] for d in corpus_rows],
        },
    }


# ---------------------------------------------------------------------------
# Validation -- run over every converted example before anything is written
# ---------------------------------------------------------------------------
class ConversionError(ValueError):
    """A converted example violates the canonical schema."""


def validate_example(question: Dict[str, Any], corpus: Sequence[Dict[str, Any]]) -> None:
    """Raise :class:`ConversionError` if the pair is not a valid canonical example.

    Checks structure *and* internal consistency: doc ids unique and matching the question,
    gold docs non-empty and referenced correctly, supporting facts resolvable to real
    sentences, and the answer non-empty. These are exactly the mistakes that would
    otherwise only surface after an expensive generation run.
    """
    for key in _QUESTION_REQUIRED:
        if key not in question:
            raise ConversionError(f"question missing field {key!r}")
    for key in _GOLD_REQUIRED:
        if key not in question["gold"]:
            raise ConversionError(f"question.gold missing field {key!r}")

    qid = question["id"]
    if not qid:
        raise ConversionError("question id is empty")
    if not str(question["query"]).strip():
        raise ConversionError(f"{qid}: empty query")
    if not str(question["answer"]).strip():
        raise ConversionError(f"{qid}: empty answer")
    if question["gold_granularity"] not in GOLD_GRANULARITIES:
        raise ConversionError(f"{qid}: bad gold_granularity {question['gold_granularity']!r}")
    if question["answer_type"] not in ANSWER_TYPES:
        raise ConversionError(f"{qid}: bad answer_type {question['answer_type']!r}")

    if not corpus:
        raise ConversionError(f"{qid}: no corpus documents")
    doc_ids = [d["doc_id"] for d in corpus]
    if len(set(doc_ids)) != len(doc_ids):
        raise ConversionError(f"{qid}: duplicate doc_ids")
    for doc in corpus:
        for key in _CORPUS_REQUIRED:
            if key not in doc:
                raise ConversionError(f"{qid}: corpus row missing field {key!r}")
        if doc["qid"] != qid:
            raise ConversionError(f"{qid}: corpus row {doc['doc_id']} has qid {doc['qid']!r}")
        if not doc["sentences"]:
            raise ConversionError(f"{qid}: corpus row {doc['doc_id']} has no sentences")
        for sid in doc["gold_sent_ids"]:
            if not 0 <= sid < len(doc["sentences"]):
                raise ConversionError(
                    f"{qid}: {doc['doc_id']} gold_sent_id {sid} out of range "
                    f"(has {len(doc['sentences'])} sentences)"
                )

    scope_ids = question["retrieval_scope"]["candidate_doc_ids"]
    if list(scope_ids) != doc_ids:
        raise ConversionError(f"{qid}: candidate_doc_ids do not match the corpus rows")

    gold_ids = question["gold"]["gold_doc_ids"]
    if not gold_ids:
        raise ConversionError(f"{qid}: no gold documents -- the question is unanswerable")
    unknown = set(gold_ids) - set(doc_ids)
    if unknown:
        raise ConversionError(f"{qid}: gold_doc_ids not in corpus: {sorted(unknown)}")

    facts = question["gold"]["supporting_facts"]
    if question["gold_granularity"] == "paragraph" and facts:
        raise ConversionError(f"{qid}: paragraph-level dataset emitted supporting_facts")
    if question["gold_granularity"] == "sentence" and not facts:
        raise ConversionError(f"{qid}: sentence-level dataset emitted no supporting_facts")
    titles = {d["title"]: d for d in corpus}
    for fact in facts:
        title, sent_id = fact.get("title"), fact.get("sent_id")
        doc = titles.get(title)
        if doc is None:
            raise ConversionError(f"{qid}: supporting fact title {title!r} not in corpus")
        if not 0 <= int(sent_id) < len(doc["sentences"]):
            raise ConversionError(
                f"{qid}: supporting fact ({title!r}, {sent_id}) out of sentence range"
            )
