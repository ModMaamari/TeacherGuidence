"""
Diversity metrics and near-duplicate capping for the SFT set.

If a handful of trajectory shapes (say ``search>extract>finish``) dominate the training
data, the student overfits to those patterns instead of learning flexible search. This
module measures how varied the accepted traces are and caps how many traces may share the
same tool-sequence signature, keeping the shortest/cleanest representatives. It also
offers a Jaccard near-duplicate check for reasoning text. All of it is content-agnostic --
it looks at tool sequences and token shingles, never the question -- so it broadens
coverage without biasing toward any topic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional


def tool_sequence(episode: Dict[str, Any]) -> List[str]:
    return [
        ((s.get("student_action") or {}).get("action") or {}).get("tool", "") or ""
        for s in (episode.get("steps") or [])
    ]


def signature(episode: Dict[str, Any]) -> str:
    return ">".join(t for t in tool_sequence(episode) if t)


def _default_rank(episode: Dict[str, Any]):
    """Prefer shorter, more efficient traces when capping; stable by qid."""
    po = episode.get("path_optimality", {}) or {}
    used = episode.get("used_steps")
    if used is None:
        used = len(episode.get("steps", []) or [])
    return (used, -float(po.get("step_efficiency", 0.0) or 0.0), str(episode.get("qid", "")))


def diversity_report(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    sigs = [signature(e) for e in episodes]
    counts = Counter(sigs)
    total = len(sigs)
    if total == 0:
        return {"total": 0, "unique_signatures": 0, "top_signatures": [], "normalized_entropy": 0.0}
    probs = [n / total for n in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
    return {
        "total": total,
        "unique_signatures": len(counts),
        "top_signatures": counts.most_common(10),
        "normalized_entropy": round(entropy / max_entropy, 4) if max_entropy > 0 else 1.0,
    }


def cap_by_signature(
    episodes: List[Dict[str, Any]],
    max_per_signature: int,
    rank: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> List[Dict[str, Any]]:
    """Keep at most ``max_per_signature`` episodes per tool-sequence signature, preferring
    the best-ranked (shortest/cleanest) ones. Deterministic."""
    if max_per_signature <= 0:
        return list(episodes)
    rank = rank or _default_rank
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in episodes:
        groups[signature(e)].append(e)
    kept: List[Dict[str, Any]] = []
    for sig in sorted(groups):
        kept.extend(sorted(groups[sig], key=rank)[:max_per_signature])
    return kept


def _shingles(text: str, n: int = 3) -> set:
    tokens = (text or "").lower().split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    sa, sb = _shingles(a, n), _shingles(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
