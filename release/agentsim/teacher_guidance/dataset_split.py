"""
Question-level, stratified train/val/test split.

Fine-tuning on traces and then evaluating on traces of the *same* questions would measure
memorization, not generalization. This splits by ``qid`` so no question ever appears in
more than one split, and stratifies by answer type (boolean / numeric / entity) so each
split has a comparable mix. Assignment is a deterministic hash of ``(seed, qid)`` -- stable
across runs and independent of input order -- so the split is reproducible.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from agentsim.teacher_guidance.metrics import normalize_answer

SPLIT_NAMES = ("train", "val", "test")


def question_stratum(episode: Dict[str, Any]) -> str:
    """A general, always-available class of the question, derived only from the gold
    answer's shape (never its value): boolean (yes/no), numeric (contains a digit), or
    entity (everything else)."""
    ans = normalize_answer(episode.get("gold_answer", "") or "")
    if ans in {"yes", "no"}:
        return "boolean"
    if any(ch.isdigit() for ch in ans):
        return "numeric"
    return "entity"


def _rank(seed: int, qid: str) -> str:
    return hashlib.md5(f"{seed}:{qid}".encode("utf-8")).hexdigest()


def split_by_question(
    episodes: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 2026,
) -> Dict[str, str]:
    """Return ``{qid: split_name}``. Splitting is per-qid (a question never crosses
    splits) and stratified: within each answer-type stratum the qids are ordered by a
    stable hash and cut at the given ratios."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    stratum_of: Dict[str, str] = {}
    for ep in episodes:
        qid = ep.get("qid")
        if qid not in stratum_of:
            stratum_of[qid] = question_stratum(ep)

    by_stratum: Dict[str, List[str]] = defaultdict(list)
    for qid, stratum in stratum_of.items():
        by_stratum[stratum].append(qid)

    assignment: Dict[str, str] = {}
    for stratum, qids in by_stratum.items():
        ordered = sorted(qids, key=lambda q: _rank(seed, q))
        n = len(ordered)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, qid in enumerate(ordered):
            if i < n_train:
                assignment[qid] = "train"
            elif i < n_train + n_val:
                assignment[qid] = "val"
            else:
                assignment[qid] = "test"
    return assignment
