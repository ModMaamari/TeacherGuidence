"""
Live progress watcher for a batched trace-collection run.

Polls the run's output root for finished episodes and maintains three files inside it:

    results_live.jsonl   one line per finished episode, appended as soon as the episode
                         file lands (qid, worker, correctness, teacher score, steps,
                         stop reason, timestamp) -- append-only, safe to tail
    stats_live.json      rolling aggregates (done/expected, correct rates, throughput,
                         ETA, per-worker episode counts) -- rewritten atomically
    missing_ids.txt      expected qids with no episode yet -- rewritten each refresh

Episodes are found both in the per-worker shard dirs (w*/...) during the run and in the
consolidated run/ dir afterwards; qids are deduplicated, so consolidation moving files
does not double-count. Exits on its own once every expected qid is seen and a
RUN_DONE marker file exists (written by the orchestration script), or runs until killed.

Usage:
    python scripts/watch_trace_collection.py \
        --out-root data/simulation_output/traces_g3b_3000 \
        --expected-ids data/simulation_output/traces_g3b_3000/used_sample_ids.txt \
        --interval 30
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

TEACHER_CORRECT_THRESHOLD = 0.40


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _episode_row(path: str, ep: dict) -> dict:
    fm = ep.get("final_metrics") or {}
    ts = fm.get("teacher_answer_score")
    worker = next((p for p in Path(path).parts if p.startswith("w") and p[1:].isdigit()), "run")
    return {
        "seen_at": _now(),
        "qid": ep.get("qid"),
        "worker": worker,
        "final_answer": ep.get("final_answer"),
        "gold_answer": ep.get("gold_answer"),
        "answer_correct": bool(fm.get("answer_correct")),
        "teacher_answer_score": ts,
        "teacher_correct": (None if ts is None else bool(float(ts) >= TEACHER_CORRECT_THRESHOLD)),
        "f1": fm.get("f1"),
        "steps": len(ep.get("steps") or []),
        "stop_reason": ep.get("stop_reason"),
        "unknown": str(ep.get("final_answer", "")).strip().lower() == "unknown",
    }


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--expected-ids", required=True)
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.out_root)
    expected = [ln.strip() for ln in open(args.expected_ids, encoding="utf-8") if ln.strip()]
    expected_set = set(expected)

    results_path = root / "results_live.jsonl"
    stats_path = root / "stats_live.json"
    missing_path = root / "missing_ids.txt"
    done_marker = root / "RUN_DONE"

    # Resume support: a restarted watcher re-reads what it already logged.
    seen: dict[str, dict] = {}
    if results_path.exists():
        for ln in open(results_path, encoding="utf-8"):
            try:
                row = json.loads(ln)
                seen[row["qid"]] = row
            except (json.JSONDecodeError, KeyError):
                continue

    scanned_files: set[str] = set()
    started = time.time()
    first_seen_t: float | None = None

    while True:
        new_rows = []
        patterns = [
            str(root / "w*" / "**" / "teacher_guidance_episodes.jsonl"),
            str(root / "run" / "**" / "teacher_guidance_episodes.jsonl"),
        ]
        for pat in patterns:
            for f in glob.glob(pat, recursive=True):
                if f in scanned_files:
                    continue
                try:
                    lines = open(f, encoding="utf-8").read().splitlines()
                except OSError:
                    continue  # mid-move during consolidation; retry next tick
                complete = True
                for ln in lines:
                    if not ln.strip():
                        continue
                    try:
                        ep = json.loads(ln)
                    except json.JSONDecodeError:
                        complete = False  # partial write; re-scan file next tick
                        continue
                    qid = ep.get("qid")
                    if qid and qid not in seen:
                        row = _episode_row(f, ep)
                        seen[qid] = row
                        new_rows.append(row)
                if complete:
                    scanned_files.add(f)

        if new_rows:
            if first_seen_t is None:
                first_seen_t = time.time()
            with open(results_path, "a", encoding="utf-8") as fh:
                for row in new_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        done = len(seen)
        rows = list(seen.values())
        scored = [r for r in rows if r["teacher_correct"] is not None]
        elapsed_h = (time.time() - (first_seen_t or started)) / 3600
        eph = (done / elapsed_h) if elapsed_h > 0 and done else 0.0
        remaining = len(expected_set) - len(expected_set & set(seen))
        stats = {
            "updated_at": _now(),
            "expected": len(expected),
            "done": done,
            "missing": remaining,
            "answer_correct": sum(r["answer_correct"] for r in rows),
            "answer_correct_rate": round(sum(r["answer_correct"] for r in rows) / done, 4) if done else None,
            "teacher_scored": len(scored),
            "teacher_correct": sum(r["teacher_correct"] for r in scored),
            "teacher_correct_rate": round(sum(r["teacher_correct"] for r in scored) / len(scored), 4) if scored else None,
            "unknown": sum(r["unknown"] for r in rows),
            "mean_steps": round(sum(r["steps"] for r in rows) / done, 2) if done else None,
            "stop_reasons": {sr: sum(1 for r in rows if r["stop_reason"] == sr)
                             for sr in sorted({r["stop_reason"] for r in rows})} if rows else {},
            "episodes_per_hour": round(eph, 1),
            "eta_hours": round(remaining / eph, 2) if eph > 0 else None,
            "per_worker": {w: sum(1 for r in rows if r["worker"] == w)
                           for w in sorted({r["worker"] for r in rows})} if rows else {},
        }
        _write_atomic(stats_path, json.dumps(stats, indent=2) + "\n")
        _write_atomic(missing_path, "\n".join(q for q in expected if q not in seen) + "\n")

        if done_marker.exists() and (remaining == 0 or done_marker.stat().st_mtime + 2 * args.interval < time.time()):
            # Run is over: one last sweep already happened above; stop.
            print(f"[{_now()}] watcher exiting: done={done}/{len(expected)} missing={remaining}", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
