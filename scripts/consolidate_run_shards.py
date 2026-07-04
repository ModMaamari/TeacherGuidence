"""
Consolidate a sharded parallel run into a single run for the trajectory explorer.

The parallel orchestrator writes each GPU worker's episodes under its own
``<shard_root>/w{i}/<run_uuid>/hotpot_questions/sample_XXX/`` tree, so the explorer
(which groups by the ``<run_uuid>`` directory) shows one run per worker. This moves every
sample directory into a single ``<shard_root>/<run_name>/hotpot_questions/`` run (samples
renumbered sequentially, all sibling files preserved) and removes the now-empty worker
dirs, so all episodes appear as one run.

Usage:
    python scripts/consolidate_run_shards.py \
        --shard-root data/simulation_output/fau_run --run-name fau_run100
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"
_WORKER_DIR_RE = re.compile(r"^w\d+$")


def find_sample_dirs(shard_root: Path, exclude_name: str) -> List[Path]:
    """Directories that directly contain an episode file, excluding the target run dir
    (so re-running is idempotent)."""
    dirs = []
    for ep in shard_root.rglob(EPISODE_FILENAME):
        sample_dir = ep.parent
        if exclude_name in sample_dir.parts:
            continue
        dirs.append(sample_dir)
    return sorted(dirs, key=lambda p: str(p))


def plan_moves(sample_dirs: List[Path], dest_hotpot: Path) -> List[Tuple[Path, Path]]:
    """Map each source sample dir to sample_0001.. under the destination, deterministically."""
    return [
        (src, dest_hotpot / f"sample_{i:04d}")
        for i, src in enumerate(sample_dirs, start=1)
    ]


def _remove_empty_worker_dirs(shard_root: Path, run_name: str) -> List[str]:
    """Remove only the emptied ``w<N>`` shard dirs -- never other siblings like ``logs``."""
    removed = []
    for child in sorted(shard_root.iterdir()):
        if not child.is_dir() or child.name == run_name or not _WORKER_DIR_RE.match(child.name):
            continue
        if not list(child.rglob(EPISODE_FILENAME)):
            shutil.rmtree(child)
            removed.append(child.name)
    return removed


def consolidate(shard_root: Path, run_name: str) -> Tuple[int, List[str]]:
    """Move all shard sample dirs into a single ``<shard_root>/<run_name>`` run and drop
    the emptied worker dirs. Returns ``(num_episodes_moved, removed_dir_names)``. A no-op
    (returns ``(0, [])``) when nothing is left to consolidate, so it's safe to re-run."""
    dest_hotpot = shard_root / run_name / "hotpot_questions"
    sample_dirs = find_sample_dirs(shard_root, run_name)
    if not sample_dirs:
        return 0, []
    dest_hotpot.mkdir(parents=True, exist_ok=True)
    for src, dest in plan_moves(sample_dirs, dest_hotpot):
        shutil.move(str(src), str(dest))
    return len(sample_dirs), _remove_empty_worker_dirs(shard_root, run_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--run-name", default="combined")
    args = ap.parse_args()

    shard_root = Path(args.shard_root)
    if not shard_root.exists():
        sys.exit(f"shard root not found: {shard_root}")

    moved, removed = consolidate(shard_root, args.run_name)
    if not moved:
        sys.exit("no sample directories with episodes found (already consolidated?)")
    print(f"consolidated {moved} episodes -> run '{shard_root.name}/{args.run_name}'")
    print(f"removed empty shard dirs: {removed}")


if __name__ == "__main__":
    main()
