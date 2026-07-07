"""
Data-access layer for the trajectory explorer.

Pure functions that scan an output root for Teacher Guidance episode files and return
runs, per-episode summaries, and full episodes. No HTTP, no mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentsim.teacher_guidance.metrics import cover_match

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"

# The explorer's correct/incorrect signal is the teacher's verdict: the teacher (which can
# see the gold answer) scores the student's final answer on a 0.0-1.0 scale
# (``teacher_answer_score``, exported from its ``final_answer_score`` field). Any answer at
# or above this threshold counts as correct -- a more forgiving, semantics-aware signal
# than deterministic cover-match, which the teacher judgment supersedes when present.
TEACHER_CORRECT_THRESHOLD = 0.40


def _answer_correct(ep: Dict[str, Any]) -> bool:
    """Correctness for an episode.

    Primary signal is the teacher verdict: ``teacher_answer_score >=
    TEACHER_CORRECT_THRESHOLD``. Falls back to the stored ``answer_correct`` flag, then to
    deterministic cover-match, for runs with no teacher verdict (e.g. skip_teacher runs, or
    a teacher that didn't return the score)."""
    fm = ep.get("final_metrics", {}) or {}
    score = fm.get("teacher_answer_score")
    if score is not None:
        try:
            return float(score) >= TEACHER_CORRECT_THRESHOLD
        except (TypeError, ValueError):
            pass
    if "answer_correct" in fm:
        return bool(fm["answer_correct"])
    return cover_match(ep.get("final_answer", ""), ep.get("gold_answer", ""))


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
        "answer_correct": _answer_correct(ep),
        "teacher_answer_correct": fm.get("teacher_answer_correct"),
        "teacher_answer_score": fm.get("teacher_answer_score"),
        "f1": float(fm.get("f1", 0.0) or 0.0),
        "supporting_doc_recall": float(fm.get("supporting_doc_recall", 0.0) or 0.0),
        "stop_reason": ep.get("stop_reason", ""),
        "num_steps": len(ep.get("steps", []) or []),
        "plan_review_enabled": bool((ep.get("plan_review") or {}).get("enabled", False)),
        "guidance_level": ep.get("guidance_level"),
        "wiki_mode": (ep.get("wiki_mode") or "tools") if ep.get("wiki_enabled") else None,
        "student_model": ep.get("student_model", ""),
        "teacher_model": ep.get("teacher_model", ""),
    }


def _teacher_source(model: str) -> str:
    """Classify a per-call teacher model id into a coarse provider/source bucket."""
    m = (model or "").lower()
    if m.startswith("fau/"):
        return "FAU"
    if m.startswith("custom/") or "openrouter" in m:
        return "OpenRouter free" if m.endswith(":free") else "OpenRouter paid"
    if not m:
        return "unknown"
    return model


def _iter_teacher_call_models(ep: Dict[str, Any]):
    pr = ep.get("plan_review") or {}
    for call in (pr.get("plan_calls") or []):
        yield call.get("model")
    for rnd in (pr.get("rounds") or []):
        for call in (rnd.get("review_calls") or []):
            yield call.get("model")
    for step in (ep.get("steps") or []):
        for call in (step.get("teacher_calls") or []):
            yield call.get("model")


# Two-level cache:
#   _FILE_CACHE  path -> parsed per-episode summaries + per-file aggregates, keyed by the
#                file's (mtime_ns, size). Incremental: when one episode file changes (a
#                run in progress appending samples), ONLY that file is re-parsed.
#   _RUNS_CACHE  output_root -> fully aggregated /api/runs payload, keyed by the stat
#                signature of every episode file (a stat scan, no reads/parses).
_FILE_CACHE: Dict[str, Dict[str, Any]] = {}
_RUNS_CACHE: Dict[str, Any] = {}


def _runs_signature(output_root: Path) -> tuple:
    sig = []
    for f in output_root.rglob(EPISODE_FILENAME):
        try:
            st = f.stat()
            sig.append((str(f), st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    sig.sort()
    return tuple(sig)


def _file_entry(episode_file: Path) -> Optional[Dict[str, Any]]:
    """Per-file cache entry: episode summaries + teacher-source counts, re-parsed only
    when the file's (mtime_ns, size) signature changes."""
    try:
        st = episode_file.stat()
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    key = str(episode_file)
    cached = _FILE_CACHE.get(key)
    if cached is not None and cached["sig"] == sig:
        return cached
    episodes = _read_jsonl(episode_file)
    source_counts: Dict[str, int] = {}
    for ep in episodes:
        for model in _iter_teacher_call_models(ep):
            src = _teacher_source(model)
            source_counts[src] = source_counts.get(src, 0) + 1
    entry = {
        "sig": sig,
        "mtime": st.st_mtime,
        "summaries": [_episode_summary(ep) for ep in episodes],
        "source_counts": source_counts,
    }
    _FILE_CACHE[key] = entry
    return entry


def _run_summary(run_id: str, summaries: List[Dict[str, Any]],
                 source_counts: Dict[str, int], mtime: float) -> Dict[str, Any]:
    em = [1.0 if s["exact_match"] else 0.0 for s in summaries]
    correct = [1.0 if s["answer_correct"] else 0.0 for s in summaries]
    f1 = [float(s["f1"] or 0.0) for s in summaries]
    doc = [float(s["supporting_doc_recall"] or 0.0) for s in summaries]
    steps = [int(s["num_steps"] or 0) for s in summaries]
    stop_reasons: Dict[str, int] = {}
    for s in summaries:
        sr = s.get("stop_reason") or "?"
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
    total_calls = sum(source_counts.values()) or 1
    teacher_source_pct = {
        k: round(100.0 * v / total_calls, 1)
        for k, v in sorted(source_counts.items(), key=lambda kv: -kv[1])
    }
    first = summaries[0]
    return {
        "run_id": run_id,
        "label": run_id,
        "num_episodes": len(summaries),
        "mtime": mtime,
        "guidance_level": first.get("guidance_level"),
        # Agent-wiki runs: "tools" | "auto" (None when the wiki was disabled).
        "wiki_mode": first.get("wiki_mode"),
        "student_model": first.get("student_model", ""),
        "teacher_model": first.get("teacher_model", ""),
        "mean_exact_match": _mean(em),
        "mean_correct": _mean(correct),
        "mean_f1": _mean(f1),
        "mean_doc_recall": _mean(doc),
        "mean_steps": _mean([float(s) for s in steps]),
        "stop_reasons": stop_reasons,
        "teacher_calls": sum(source_counts.values()),
        "teacher_source_pct": teacher_source_pct,
    }


def find_runs(output_root: str | Path) -> List[Dict[str, Any]]:
    """Group all episode files by run and return one summary per run, newest first.

    Incremental: aggregates are memoized per output_root on a cheap stat signature, and
    when the signature changes only the episode files that actually changed are
    re-parsed (per-file cache) -- a run appending samples doesn't invalidate the rest.
    """
    output_root = Path(output_root)
    if not output_root.exists():
        return []

    signature = _runs_signature(output_root)
    cached = _RUNS_CACHE.get(str(output_root))
    if cached is not None and cached[0] == signature:
        return cached[1]

    runs: Dict[str, Dict[str, Any]] = {}
    seen_files = set()
    for episode_file in output_root.rglob(EPISODE_FILENAME):
        entry = _file_entry(episode_file)
        if entry is None or not entry["summaries"]:
            continue
        seen_files.add(str(episode_file))
        run_id = _run_id(output_root, _run_dir_for(episode_file))
        bucket = runs.setdefault(
            run_id, {"summaries": [], "source_counts": {}, "mtime": 0.0}
        )
        bucket["summaries"].extend(entry["summaries"])
        for k, v in entry["source_counts"].items():
            bucket["source_counts"][k] = bucket["source_counts"].get(k, 0) + v
        bucket["mtime"] = max(bucket["mtime"], entry["mtime"])

    # Drop cache entries for files that no longer exist (deleted/moved runs).
    for stale in [k for k in _FILE_CACHE if k.startswith(str(output_root)) and k not in seen_files]:
        _FILE_CACHE.pop(stale, None)

    result = [
        _run_summary(run_id, b["summaries"], b["source_counts"], b["mtime"])
        for run_id, b in runs.items()
    ]
    # Newest run first (by most-recent episode file mtime).
    result.sort(key=lambda r: r["mtime"], reverse=True)
    _RUNS_CACHE[str(output_root)] = (signature, result)
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
    """Return per-episode summaries for a run, ordered by sample path.

    Served from the per-file summary cache -- unchanged files are never re-parsed."""
    output_root = Path(output_root)
    summaries: List[Dict[str, Any]] = []
    for episode_file in _episode_files_for_run(output_root, run_id):
        entry = _file_entry(episode_file)
        if entry is not None:
            summaries.extend(entry["summaries"])
    return summaries


def _load_sft_index(path: Path) -> Dict[int, Dict[str, str]]:
    """step -> {"input", "output"} from a student_sft.jsonl/teacher_sft.jsonl file."""
    idx: Dict[int, Dict[str, str]] = {}
    if not path.exists():
        return idx
    for row in _read_jsonl(path):
        step = (row.get("metadata") or {}).get("step")
        if step is not None:
            idx[step] = {"input": row.get("input", ""), "output": row.get("output", "")}
    return idx


def _backfill_raw_io(ep: Dict[str, Any], episode_dir: Path) -> None:
    """Older runs never wrote student_prompt/teacher_prompt into the episode row (fixed
    in a later exporter version), but the sibling student_sft.jsonl/teacher_sft.jsonl
    files have always carried the same input/output per step. Recover it from there so
    the viewer can show raw I/O without requiring the run to be regenerated. Flags each
    backfilled step since per-call timing and the raw provider response can't be
    recovered this way — only the prompt/response text."""
    steps = ep.get("steps") or []
    if any(not s.get("student_prompt") for s in steps):
        idx = _load_sft_index(episode_dir / "student_sft.jsonl")
        for s in steps:
            hit = idx.get(s.get("t"))
            if not s.get("student_prompt") and hit:
                s["student_prompt"] = hit["input"]
                s["student_raw"] = hit["output"]
                s["student_io_backfilled"] = True
    if any(not s.get("teacher_prompt") for s in steps):
        idx = _load_sft_index(episode_dir / "teacher_sft.jsonl")
        for s in steps:
            hit = idx.get(s.get("t"))
            if not s.get("teacher_prompt") and hit:
                s["teacher_prompt"] = hit["input"]
                s["teacher_raw"] = hit["output"]
                s["teacher_io_backfilled"] = True


def get_episode(output_root: str | Path, run_id: str, qid: str) -> Optional[Dict[str, Any]]:
    """Return the full episode for a qid within a run, or None."""
    output_root = Path(output_root)
    for episode_file in _episode_files_for_run(output_root, run_id):
        for ep in _read_jsonl(episode_file):
            if str(ep.get("qid")) == str(qid):
                fm = ep.setdefault("final_metrics", {})
                # Recompute so the episode-detail view uses the same teacher-verdict signal
                # as the list/aggregate (not the stored deterministic cover-match value).
                fm["answer_correct"] = _answer_correct(ep)
                _backfill_raw_io(ep, episode_file.parent)
                return ep
    return None
