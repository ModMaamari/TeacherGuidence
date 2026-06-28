"""
Data-access layer for the trajectory explorer.

Pure functions that scan an output root for Teacher Guidance episode files and return
runs, per-episode summaries, and full episodes. No HTTP, no mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _run_dir_for(episode_file: Path) -> Path:
    """Run directory = the <run_uuid> dir (sample/dataset/run). Fall back gracefully."""
    parents = episode_file.parents
    if len(parents) >= 3:
        return parents[2]
    return episode_file.parent


def _run_id(output_root: Path, run_dir: Path) -> str:
    try:
        rel = run_dir.resolve().relative_to(output_root.resolve())
    except ValueError:
        rel = Path(run_dir.name)
    return rel.as_posix()


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _episode_summary(ep: Dict[str, Any]) -> Dict[str, Any]:
    fm = ep.get("final_metrics", {}) or {}
    return {
        "qid": ep.get("qid"),
        "query": ep.get("query", ""),
        "gold_answer": ep.get("gold_answer", ""),
        "final_answer": ep.get("final_answer", ""),
        "exact_match": bool(fm.get("exact_match", False)),
        "f1": float(fm.get("f1", 0.0) or 0.0),
        "supporting_doc_recall": float(fm.get("supporting_doc_recall", 0.0) or 0.0),
        "stop_reason": ep.get("stop_reason", ""),
        "num_steps": len(ep.get("steps", []) or []),
        "plan_review_enabled": bool((ep.get("plan_review") or {}).get("enabled", False)),
        "guidance_level": ep.get("guidance_level"),
    }


def find_runs(output_root: str | Path) -> List[Dict[str, Any]]:
    """Group all episode files by run and return one summary per run."""
    output_root = Path(output_root)
    if not output_root.exists():
        return []

    runs: Dict[str, Dict[str, Any]] = {}
    for episode_file in output_root.rglob(EPISODE_FILENAME):
        run_dir = _run_dir_for(episode_file)
        run_id = _run_id(output_root, run_dir)
        episodes = _read_jsonl(episode_file)
        bucket = runs.setdefault(
            run_id,
            {"run_id": run_id, "label": run_id, "episodes": []},
        )
        bucket["episodes"].extend(episodes)

    result: List[Dict[str, Any]] = []
    for run_id, bucket in runs.items():
        episodes = bucket["episodes"]
        if not episodes:
            continue
        em = [1.0 if (e.get("final_metrics", {}) or {}).get("exact_match") else 0.0 for e in episodes]
        f1 = [float((e.get("final_metrics", {}) or {}).get("f1", 0.0) or 0.0) for e in episodes]
        doc = [float((e.get("final_metrics", {}) or {}).get("supporting_doc_recall", 0.0) or 0.0) for e in episodes]
        stop_reasons: Dict[str, int] = {}
        for e in episodes:
            sr = e.get("stop_reason", "?")
            stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
        first = episodes[0]
        result.append(
            {
                "run_id": run_id,
                "label": run_id,
                "num_episodes": len(episodes),
                "guidance_level": first.get("guidance_level"),
                "student_model": first.get("student_model", ""),
                "teacher_model": first.get("teacher_model", ""),
                "mean_exact_match": _mean(em),
                "mean_f1": _mean(f1),
                "mean_doc_recall": _mean(doc),
                "stop_reasons": stop_reasons,
            }
        )
    result.sort(key=lambda r: r["run_id"])
    return result


def _resolve_run_dir(output_root: Path, run_id: str) -> Optional[Path]:
    """Resolve run_id to a directory under output_root, rejecting traversal."""
    output_root = output_root.resolve()
    candidate = (output_root / run_id).resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def _episode_files_for_run(output_root: Path, run_id: str) -> List[Path]:
    run_dir = _resolve_run_dir(output_root, run_id)
    if run_dir is None:
        return []
    return sorted(run_dir.rglob(EPISODE_FILENAME))


def get_run_episodes(output_root: str | Path, run_id: str) -> List[Dict[str, Any]]:
    """Return per-episode summaries for a run, ordered by sample path."""
    output_root = Path(output_root)
    summaries: List[Dict[str, Any]] = []
    for episode_file in _episode_files_for_run(output_root, run_id):
        for ep in _read_jsonl(episode_file):
            summaries.append(_episode_summary(ep))
    return summaries


def get_episode(output_root: str | Path, run_id: str, qid: str) -> Optional[Dict[str, Any]]:
    """Return the full episode for a qid within a run, or None."""
    output_root = Path(output_root)
    for episode_file in _episode_files_for_run(output_root, run_id):
        for ep in _read_jsonl(episode_file):
            if str(ep.get("qid")) == str(qid):
                return ep
    return None
