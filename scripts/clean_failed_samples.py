"""Remove sample dirs that ran but produced no episode, and un-mark them in checkpoints.

When the teacher endpoint fails, a sample still gets a directory, a `_SUCCESS` marker and a
checkpoint entry -- but no `teacher_guidance_episodes.jsonl`, because the workflow never
completed a trajectory. That combination is worse than a plain crash: the runner's resume
logic trusts the checkpoint, so those questions would be skipped forever and the corpus
would carry silent holes.

This deletes such directories and removes them from `completed_samples`, so the next
resume re-collects exactly the questions that produced nothing.

Usage::

    python scripts/clean_failed_samples.py --run-root data/simulation_output/tgv1 --dry-run
    python scripts/clean_failed_samples.py --run-root data/simulation_output/tgv1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EPISODE = "teacher_guidance_episodes.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="data/simulation_output/tgv1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_root)
    removed = uncheckpointed = 0
    for shard in sorted(p for p in root.glob("tgv1_*") if p.is_dir()):
        empties = [s for s in shard.glob("*/*/sample_*")
                   if s.is_dir() and not (s / EPISODE).exists()]
        if not empties:
            continue
        names = {s.name for s in empties}
        for s in empties:
            if not args.dry_run:
                shutil.rmtree(s)
            removed += 1
        for cp in shard.glob("checkpoint_*.json"):
            state = json.loads(cp.read_text(encoding="utf-8"))
            before = state.get("completed_samples", [])
            after = [s for s in before if s not in names]
            if len(after) != len(before):
                uncheckpointed += len(before) - len(after)
                state["completed_samples"] = after
                # The run is being resumed, not finished; keep the status honest.
                state["status"] = "running"
                if not args.dry_run:
                    cp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"  {shard.name:<30} {len(empties):>4} empty sample dir(s)")
    verb = "would remove" if args.dry_run else "removed"
    print(f"\n{verb} {removed} empty sample dirs; "
          f"{'would un-mark' if args.dry_run else 'un-marked'} {uncheckpointed} checkpoint entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
