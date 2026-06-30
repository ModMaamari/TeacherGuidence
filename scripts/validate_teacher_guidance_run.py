"""
Validate a Teacher Guidance run directory.

Walks a run directory, finds every teacher_guidance_episodes.jsonl, and checks:

* episode JSON validity and presence of required fields,
* gold answer / hidden title / hidden doc id leakage flags,
* invalid span extraction,
* presence of final metrics and a stop reason.

Exits non-zero if any leakage is detected.

Usage:
    python scripts/validate_teacher_guidance_run.py --run-dir data/simulation_output/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_FIELDS = ["episode_id", "qid", "query", "final_metrics", "stop_reason", "steps"]


def _iter_episode_files(run_dir: Path):
    yield from run_dir.rglob("teacher_guidance_episodes.jsonl")


def validate_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    leaks = 0
    fallbacks_used = 0
    for field in REQUIRED_FIELDS:
        if field not in ep:
            issues.append(f"missing_field:{field}")
    if not ep.get("stop_reason"):
        issues.append("empty_stop_reason")

    for step in ep.get("steps", []) or []:
        leak = step.get("leakage_check", {}) or {}
        for key in ("gold_answer_leaked", "hidden_doc_id_leaked", "hidden_title_leaked", "hidden_span_leaked"):
            if leak.get(key):
                leaks += 1
                issues.append(f"leak:t{step.get('t')}:{key}")
        if leak.get("feedback_fallback_used"):
            fallbacks_used += 1
        obs = step.get("tool_observation", {}) or {}
        if obs.get("invalid_spans"):
            issues.append(f"invalid_spans:t{step.get('t')}")

    plan_review = ep.get("plan_review", {}) or {}
    if plan_review.get("enabled") and plan_review.get("planner") == "student":
        for round_rec in plan_review.get("rounds", []) or []:
            feedback = (round_rec.get("student_visible_plan_feedback") or {})
            # "feedback" key absent (e.g. guidance level 0/1) is fine; present-but-blank
            # is the regression this guards against (see guidance_policy.FALLBACK_FEEDBACK,
            # which should make this unreachable post-fix — flag it if it ever recurs).
            if "feedback" in feedback and not (feedback.get("feedback") or "").strip():
                issues.append(f"blank_plan_feedback:round{round_rec.get('round')}")
            if (round_rec.get("leakage_check") or {}).get("feedback_fallback_used"):
                fallbacks_used += 1

    return {"qid": ep.get("qid"), "issues": issues, "leaks": leaks, "fallbacks_used": fallbacks_used}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    total_eps = 0
    total_leaks = 0
    total_fallbacks = 0
    flagged: List[Dict[str, Any]] = []

    for path in _iter_episode_files(run_dir):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_eps += 1
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    flagged.append({"qid": "?", "issues": ["unparseable_episode"], "leaks": 0})
                    continue
                report = validate_episode(ep)
                total_leaks += report["leaks"]
                total_fallbacks += report.get("fallbacks_used", 0)
                if report["issues"]:
                    flagged.append(report)

    print(f"Episodes checked: {total_eps}")
    print(f"Total leakage flags: {total_leaks}")
    print(f"Plan-review feedback fallbacks used: {total_fallbacks}")
    print(f"Episodes with issues: {len(flagged)}")
    for report in flagged[:50]:
        print(f"  - qid={report['qid']}: {', '.join(report['issues'])}")

    sys.exit(1 if total_leaks > 0 else 0)


if __name__ == "__main__":
    main()
