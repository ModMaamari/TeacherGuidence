"""Verify a four-arm eval experiment is complete before analysis (PLAN.md G-9).

Reports the episode count of every per-run metrics.json and the judged count, and flags any
run short of --expect episodes (a circuit-breaker-tripped teacher seed must be refilled and
re-judged, never reported at a partial count).

Usage:
    .venv_train/bin/python training_methods/exp_teacher_only/check_eval.py \
        --exp-dir training_methods/exp_unseen100/runs/<ts>_exp_teacheronly_<model> [--expect 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--expect", type=int, default=100)
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    runs = sorted(exp.glob("*/metrics.json"))
    short = []
    total = 0
    print(f"=== {exp.name}: {len(runs)} runs ===")
    for mf in runs:
        m = json.loads(mf.read_text())
        tag = mf.parent.name
        n = int(m.get("n", 0))
        total += n
        flag = "" if n >= args.expect else "  <-- SHORT"
        print(f"  {tag:40} n={n:4} arm={m.get('arm','?')}{flag}")
        if n < args.expect:
            short.append((tag, n))

    jf = exp / "judge" / "summary.json"
    judged = None
    if jf.exists():
        js = json.loads(jf.read_text())
        judged = sum(v.get("judged", 0) for v in js.values() if isinstance(v, dict))
    print(f"total episodes across runs: {total} | judged: {judged}")
    if short:
        print(f"SHORT RUNS ({len(short)}) -- refill + re-judge before reporting: {short}")
    else:
        print("all runs complete.")


if __name__ == "__main__":
    main()
