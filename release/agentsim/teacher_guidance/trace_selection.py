"""
Canonical trace selection: one optimal trace per question.

The same HotpotQA question is typically solved many times over the generation matrix
(different student models, settings, temperatures). If all correct traces went into the
SFT set, the data would over-represent easy questions and bake in one model's idioms.
Instead we keep, per qid, the single *most optimal* accepted trace: the shortest path,
then the most efficient, then the most precise answer. That yields an optimal-path
dataset with one example per question -- fewer, cleaner, and balanced across questions,
which helps the student generalize rather than memorize a particular model's quirks.

Ranking uses only trajectory shape and answer quality (length, efficiency, F1), never the
specific question, so it never biases toward particular content.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _rank_key(episode: Dict[str, Any]):
    """Lower is better. Prefer: fewer steps, then higher efficiency, then fewer wasted
    steps, then higher F1; break ties deterministically for reproducible datasets."""
    po = episode.get("path_optimality", {}) or {}
    fm = episode.get("final_metrics", {}) or {}
    used_steps = episode.get("used_steps")
    if used_steps is None:
        used_steps = len(episode.get("steps", []) or [])
    return (
        used_steps,
        -float(po.get("step_efficiency", 0.0) or 0.0),
        int(po.get("wasted_step_count", 0) or 0),
        -float(fm.get("f1", 0.0) or 0.0),
        str(episode.get("student_model", "")),
        str(episode.get("episode_id", "")),
    )


def select_canonical(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one episode per qid -- the most optimal by ``_rank_key`` -- sorted by qid
    for deterministic output."""
    by_qid: Dict[Any, List[Dict[str, Any]]] = {}
    for ep in episodes:
        by_qid.setdefault(ep.get("qid"), []).append(ep)
    return [min(group, key=_rank_key) for _, group in sorted(by_qid.items(), key=lambda kv: str(kv[0]))]
