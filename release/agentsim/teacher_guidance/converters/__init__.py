"""Source-dataset converters for teacher-guidance data collection.

Every supported multi-hop QA dataset is converted into ONE canonical pair of row shapes
(see :mod:`.base`), so the simulation engine, retriever, metrics and release packager are
dataset-agnostic. Adding a dataset = adding a converter module + a :data:`CONVERTERS` entry.

Conversion always runs through :func:`convert_dataset`, which validates every example
against the canonical schema and *drops* (with a counted reason) anything malformed. That
turns silent data corruption into an up-front, visible rejection count -- the whole point
being to catch mistakes before an expensive generation run, not after.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

from agentsim.teacher_guidance.converters import hotpot, musique, strategyqa, twowiki
from agentsim.teacher_guidance.converters.base import (
    DATASET_SCHEMA_VERSION,
    ConversionError,
    validate_example,
)

__all__ = [
    "CONVERTERS",
    "DATASET_SCHEMA_VERSION",
    "ConversionError",
    "DatasetSpec",
    "convert_dataset",
    "dataset_names",
    "get_spec",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to convert and to document one source dataset."""

    name: str
    source: str
    convert: Callable[..., Any]
    gold_granularity: str
    #: Answer types that occur in this source. Several datasets mix them: HotpotQA and
    #: 2WikiMultihopQA comparison questions answer yes/no while bridge questions answer
    #: with a span, so the type is inferred per question and this records the full set.
    answer_types: Tuple[str, ...]
    license: str
    homepage: str
    #: Extra keyword arguments ``convert`` requires beyond ``(example, split)``.
    requires: Sequence[str] = field(default_factory=tuple)
    notes: str = ""


CONVERTERS: Dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        source=hotpot.SOURCE,
        convert=hotpot.convert_example,
        gold_granularity="sentence",
        answer_types=("span", "boolean"),
        license="CC BY-SA 4.0",
        homepage="https://hotpotqa.github.io/",
        notes="2-hop bridge/comparison; distractor setting ships 10 paragraphs per question.",
    ),
    "2wikimultihopqa": DatasetSpec(
        name="2wikimultihopqa",
        source=twowiki.SOURCE,
        convert=twowiki.convert_example,
        gold_granularity="sentence",
        answer_types=("span", "boolean"),
        license="Apache-2.0",
        homepage="https://github.com/Alab-NII/2wikimultihop",
        notes="2-4 hop; adds symbolic evidence triples; bridge_comparison is 4-hop.",
    ),
    "musique": DatasetSpec(
        name="musique",
        source=musique.SOURCE,
        convert=musique.convert_example,
        gold_granularity="paragraph",
        answer_types=("span",),
        license="CC BY 4.0",
        homepage="https://github.com/StonyBrookNLP/musique",
        notes="2-4 hop, shortcut-resistant; paragraph-level gold; ships answer aliases.",
    ),
    "strategyqa": DatasetSpec(
        name="strategyqa",
        source=strategyqa.SOURCE,
        convert=strategyqa.convert_example,
        gold_granularity="paragraph",
        answer_types=("boolean",),
        license="MIT",
        homepage="https://allenai.org/data/strategyqa",
        requires=("paragraphs",),
        notes=(
            "Implicit multi-hop, boolean answers. Candidate sets are CONSTRUCTED "
            "(gold + seeded distractors) because the source ships no distractors."
        ),
    ),
}


def dataset_names() -> List[str]:
    return sorted(CONVERTERS)


def get_spec(name: str) -> DatasetSpec:
    try:
        return CONVERTERS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {dataset_names()}") from None


def convert_dataset(
    name: str,
    examples: Sequence[Dict[str, Any]],
    split: str,
    *,
    validate: bool = True,
    strict: bool = False,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Convert many examples of one dataset into canonical rows.

    Returns ``(question_rows, corpus_rows, stats)``. ``stats`` records how many examples
    were converted, how many were skipped and why -- rejection reasons are counted rather
    than hidden, because a sudden jump in skips is the earliest signal that a source file
    or a converter is wrong.

    With ``strict=True`` the first invalid example raises instead of being skipped.
    """
    spec = get_spec(name)
    missing = [r for r in spec.requires if r not in kwargs]
    if missing:
        raise TypeError(f"{name} requires converter argument(s): {missing}")

    question_rows: List[Dict[str, Any]] = []
    corpus_rows: List[Dict[str, Any]] = []
    skipped: collections.Counter = collections.Counter()
    seen_qids: set = set()

    for example in examples:
        try:
            converted = spec.convert(example, split=split, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- one bad row must not kill the batch
            if strict:
                raise
            skipped[f"convert_error:{type(exc).__name__}"] += 1
            continue
        if converted is None:
            skipped["unconvertible"] += 1
            continue
        question, corpus = converted

        if question["id"] in seen_qids:
            skipped["duplicate_qid"] += 1
            continue
        if validate:
            try:
                validate_example(question, corpus)
            except ConversionError as exc:
                if strict:
                    raise
                skipped[f"invalid:{str(exc).split(':')[-1].strip()[:40]}"] += 1
                continue

        seen_qids.add(question["id"])
        question_rows.append(question)
        corpus_rows.extend(corpus)

    stats = {
        "dataset": name,
        "source": spec.source,
        "split": split,
        "schema_version": DATASET_SCHEMA_VERSION,
        "input_examples": len(examples),
        "converted": len(question_rows),
        "corpus_docs": len(corpus_rows),
        "skipped_total": sum(skipped.values()),
        "skipped_reasons": dict(skipped),
        "gold_granularity": spec.gold_granularity,
        "license": spec.license,
    }
    return question_rows, corpus_rows, stats
