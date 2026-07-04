"""
Filter generated episodes down to training-worthy traces.

Scans one or more simulation-output roots for teacher_guidance_episodes.jsonl, applies
the trace-quality gate (agentsim.teacher_guidance.trace_quality), and writes the accepted
episodes plus a funnel/rejection report. This is stage 1 of the trace -> SFT pipeline:
keep only correct, grounded, clean, efficient, naturally-finished, leak-free trajectories.

Usage:
    python scripts/filter_traces.py --roots data/simulation_output/exp_matrix \
        --out data/sft/accepted_traces.jsonl
    python scripts/filter_traces.py --allow-forced-finish --min-step-efficiency 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.trace_quality import TraceCriteria, evaluate_trace  # noqa: E402

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"


# ---------------------------------------------------------------------------
# Pure (unit-tested)
# ---------------------------------------------------------------------------
def partition_episodes(
    episodes: Iterable[Dict[str, Any]], criteria: TraceCriteria
) -> Tuple[List[Dict[str, Any]], int, Counter]:
    """Return ``(accepted, num_rejected, rejection_reason_counts)``."""
    accepted: List[Dict[str, Any]] = []
    num_rejected = 0
    reasons: Counter = Counter()
    for ep in episodes:
        verdict = evaluate_trace(ep, criteria)
        if verdict["accepted"]:
            accepted.append(ep)
        else:
            num_rejected += 1
            reasons.update(verdict["reasons"])
    return accepted, num_rejected, reasons


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def iter_episodes(roots: List[Path]) -> Iterable[Dict[str, Any]]:
    for root in roots:
        for path in sorted(root.rglob(EPISODE_FILENAME)):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep.setdefault("_source", str(path.relative_to(root.parent if root.parent else root)))
                yield ep


def _criteria_from_args(args) -> TraceCriteria:
    return TraceCriteria(
        require_grounded=not args.no_require_grounded,
        require_natural_finish=not args.allow_forced_finish,
        min_step_efficiency=args.min_step_efficiency,
        max_wasted_steps=args.max_wasted_steps,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=["data/simulation_output"])
    ap.add_argument("--out", default="data/sft/accepted_traces.jsonl")
    ap.add_argument("--no-require-grounded", action="store_true")
    ap.add_argument("--allow-forced-finish", action="store_true")
    ap.add_argument("--min-step-efficiency", type=float, default=0.5)
    ap.add_argument("--max-wasted-steps", type=int, default=1)
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    criteria = _criteria_from_args(args)

    episodes = list(iter_episodes(roots))
    accepted, num_rejected, reasons = partition_episodes(episodes, criteria)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in accepted:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    total = len(episodes)
    report = {
        "total_episodes": total,
        "accepted": len(accepted),
        "rejected": num_rejected,
        "acceptance_rate": round(len(accepted) / total, 4) if total else 0.0,
        "rejection_reasons": dict(reasons.most_common()),
        "unique_qids_accepted": len({ep.get("qid") for ep in accepted}),
        "criteria": criteria.__dict__,
    }
    report_path = out_path.with_name("filter_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"episodes scanned: {total}")
    print(f"accepted:        {len(accepted)} ({report['acceptance_rate']:.1%})")
    print(f"rejected:        {num_rejected}")
    print(f"rejection reasons: {report['rejection_reasons']}")
    print(f"accepted traces -> {out_path}")
    print(f"report          -> {report_path}")


if __name__ == "__main__":
    main()
