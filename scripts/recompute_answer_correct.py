"""
Recompute `final_metrics.answer_correct` in every stored Teacher Guidance episode
file, using the current `cover_match` (agentsim/teacher_guidance/metrics.py).

Why this is needed: `answer_correct` is computed once at export time and stored in
`teacher_guidance_episodes.jsonl`; the viewer and report both read that stored value
directly rather than recomputing it live (see
`agentsim/teacher_guidance/viewer/data_access.py:_answer_correct`, which only
recomputes when the field is *missing*). A `cover_match` bug fix therefore doesn't
retroactively change anything already on disk -- this script re-scores every stored
episode in place so the explorer and any report built from the data reflect the fix.

Usage:
    python scripts/recompute_answer_correct.py [--root data/simulation_output] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.teacher_guidance.metrics import cover_match  # noqa: E402

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"


def recompute_episode(episode: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, bool]:
    """Return (episode, changed, regressed). `regressed` is True iff a previously
    True answer_correct flipped to False -- cover_match's fix is purely additive
    (it only accepts cases the old strict check rejected), so this should never
    happen; if it does, treat it as a bug to investigate rather than apply silently.
    """
    fm = episode.get("final_metrics")
    if not isinstance(fm, dict) or "answer_correct" not in fm:
        return episode, False, False

    old_value = bool(fm["answer_correct"])
    new_value = cover_match(episode.get("final_answer", ""), episode.get("gold_answer", ""))
    if new_value == old_value:
        return episode, False, False

    regressed = old_value and not new_value
    fm["answer_correct"] = new_value
    return episode, True, regressed


def process_file(path: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """Return (num_records, num_changed, num_regressed) for one episode file."""
    records: List[Dict[str, Any]] = []
    num_changed = 0
    num_regressed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode, changed, regressed = recompute_episode(json.loads(line))
            records.append(episode)
            if changed:
                num_changed += 1
            if regressed:
                num_regressed += 1

    if num_changed and not dry_run:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for episode in records:
                f.write(json.dumps(episode) + "\n")
        os.replace(tmp_path, path)

    return len(records), num_changed, num_regressed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/simulation_output")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing")
    args = parser.parse_args()

    root = Path(args.root)
    total_files = total_records = total_changed = total_regressed = 0
    regressed_files = []

    for path in sorted(root.rglob(EPISODE_FILENAME)):
        num_records, num_changed, num_regressed = process_file(path, dry_run=args.dry_run)
        total_files += 1
        total_records += num_records
        total_changed += num_changed
        total_regressed += num_regressed
        if num_regressed:
            regressed_files.append(str(path))

    verb = "Would update" if args.dry_run else "Updated"
    print(f"Scanned {total_files} episode file(s), {total_records} episode record(s).")
    print(f"{verb} {total_changed} record(s) (answer_correct flipped, almost always False -> True).")
    if total_regressed:
        print(f"WARNING: {total_regressed} record(s) flipped True -> False in: {regressed_files}")
    else:
        print("No regressions (no True -> False flips), as expected.")


if __name__ == "__main__":
    main()
