"""Backward-compatible shim for the HotpotQA converter.

The implementation moved to :mod:`agentsim.teacher_guidance.converters.hotpot` when the
converter package was generalized to several multi-hop QA datasets. This module stays so
existing callers (``scripts/prepare_hotpot_teacher_guidance.py`` and its tests) keep
working, and it *delegates* rather than duplicating -- one implementation, so HotpotQA rows
can never drift from the rows every other dataset produces.

Rows now carry the canonical extra fields (``source``, ``split``, ``num_hops``,
``answer_type``, ``gold_granularity``, ``gold.answer_aliases``); every previously emitted
field is unchanged.

New code should use :func:`agentsim.teacher_guidance.converters.convert_dataset`, which
also validates every converted example against the canonical schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentsim.teacher_guidance.converters.hotpot import convert_example
from agentsim.teacher_guidance.converters.hotpot import (  # noqa: F401 -- legacy re-export
    normalize_context as _normalize_context,
)
from agentsim.teacher_guidance.converters.hotpot import (  # noqa: F401 -- legacy re-export
    normalize_supporting_facts as _normalize_supporting_facts,
)

__all__ = ["convert_example", "convert_examples"]


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
