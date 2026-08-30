"""Re-run the episodes a collection pass lost to transient gateway failures.

Each tg_v1 template pins one teacher with no fallback, so cost and provenance stay
attributable -- the cost of that choice is that a single ReadTimeout or 500 ends the
episode with ``stop_reason: "error"``. Those episodes are written to disk, so the
collection runner's resume logic (which counts exported episodes) treats them as done and
never retries them. This closes that gap.

The retried episodes are collected with the SAME teacher and config, so the corpus stays
single-teacher; only the failures are re-attempted. Superseded records are left in place
and filtered by qid at consolidation -- deleting from a JSONL mid-run risks losing good
episodes if the retry itself fails.

Usage::

    python scripts/retry_failed_episodes.py --run-root data/simulation_output/tgv1 --list
    python scripts/retry_failed_episodes.py --run-root data/simulation_output/tgv1 --shards 4
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_tgv1_templates import DATASETS, build  # noqa: E402

RETRY_ROOT = "./data/simulation_output/tgv1_retry"
SHARD_DIR = REPO_ROOT / "data" / "datasets" / "tg_v1_retry"
TEMPLATE_DIR = REPO_ROOT / "templates" / "simulations"
#: Episodes in these states carry no usable trajectory and are worth re-attempting.
FAILED_STATES = {"error"}


def failed_qids(run_root: Path) -> Dict[str, List[str]]:
    """dataset -> qids whose only episode ended in a failed state."""
    best: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for p in run_root.rglob("teacher_guidance_episodes.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            ds, qid, stop = e.get("dataset"), e.get("qid"), e.get("stop_reason")
            # A later good episode for the same qid supersedes an earlier failure.
            if best[ds].get(qid) not in (None,) and best[ds].get(qid) not in FAILED_STATES:
                continue
            best[ds][qid] = stop
    return {ds: sorted(q for q, s in m.items() if s in FAILED_STATES) for ds, m in best.items()}


def question_index(dataset: str) -> Dict[str, str]:
    d, stem = DATASETS[dataset]
    path = REPO_ROOT / "data" / "datasets" / "tg_v1" / d / f"{stem}_questions.jsonl"
    idx = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            idx[str(json.loads(line)["id"])] = line
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="data/simulation_output/tgv1")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="report failures, run nothing")
    args = ap.parse_args()

    failures = failed_qids(Path(args.run_root))
    total = sum(len(v) for v in failures.values())
    print(f"failed episodes needing retry: {total}")
    for ds, qids in sorted(failures.items()):
        print(f"  {ds:<18} {len(qids)}")
    if args.list or total == 0:
        return 0

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    log_dir = REPO_ROOT / "data" / "simulation_output" / "tgv1_retry" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for ds, qids in sorted(failures.items()):
        if not qids:
            continue
        idx = question_index(ds)
        lines = [idx[q] for q in qids if q in idx]
        n_shards = max(1, min(args.shards, len(lines)))
        for s in range(n_shards):
            shard_lines = lines[s::n_shards]
            if not shard_lines:
                continue
            qpath = SHARD_DIR / f"{ds}_retry_s{s}.jsonl"
            qpath.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")
            tid = f"tgv1_retry_{ds}_s{s}"
            tpl = build(ds, len(shard_lines), tid, RETRY_ROOT)
            tpl["datasets"][0]["path"] = "./" + str(qpath.relative_to(REPO_ROOT))
            (TEMPLATE_DIR / f"{tid}.yaml").write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
            log = (log_dir / f"{tid}.log").open("w", encoding="utf-8")
            procs.append((tid, subprocess.Popen(
                [sys.executable, "-m", "agentsim.cli", "simulate", tid],
                cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT), log))
    print(f"launched {len(procs)} retry workers")
    bad = []
    for tid, proc, log in procs:
        rc = proc.wait(); log.close()
        print(f"  [{'ok' if rc == 0 else 'FAILED'}] {tid} rc={rc}")
        if rc:
            bad.append(tid)
    print(f"retry finished; failed workers: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
