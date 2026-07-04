"""
Build the SFT dataset report + leakage healthcheck (final pipeline stage).

Summarizes the exported training set and asserts it is free of gold-answer leakage.
Exits non-zero if the leakage healthcheck fails, so it can gate a training run in CI.

Usage:
    python scripts/build_sft_report.py \
        --accepted data/sft/accepted_traces.jsonl \
        --examples data/sft/sft_examples.jsonl --out data/sft/SFT_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.sft_report import build_report  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _render_md(report: Dict[str, Any]) -> str:
    hc = report["leakage_healthcheck"]
    lines = ["# SFT dataset report", ""]
    lines.append(f"- Episodes (accepted traces): **{report['episodes']}**")
    lines.append(f"- Unique questions: **{report['unique_questions']}**")
    lines.append(f"- Training examples: **{report['examples']}** {report['example_kinds']}")
    lines.append(f"- Step count: {report['step_count_distribution']}")
    lines.append("")
    lines.append("## Tool distribution (action examples)")
    for tool, n in report["tool_distribution"].items():
        lines.append(f"- `{tool}`: {n}")
    lines.append("")
    lines.append("## Question-type balance")
    for stratum, n in report["question_type_balance"].items():
        lines.append(f"- {stratum}: {n}")
    lines.append("")
    lines.append("## Diversity")
    lines.append(f"- Unique tool-sequences: {report['diversity']['unique_signatures']}")
    lines.append(f"- Normalized entropy: {report['diversity']['normalized_entropy']}")
    lines.append(f"- Top signatures: {report['diversity']['top_signatures']}")
    lines.append("")
    lines.append("## Leakage healthcheck")
    lines.append(f"- Clean: **{hc['clean']}**")
    lines.append(f"- Steps flagged gold_answer_leaked: {hc['flagged_gold_answer_leaked_steps']}")
    lines.append(f"- Internalized-reflection leaks: {len(hc['reflection_leaks'])}")
    if hc["reflection_leaks"]:
        for leak in hc["reflection_leaks"][:20]:
            lines.append(f"  - qid={leak['qid']} step={leak['step']}: {leak['reflection']!r}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted", default="data/sft/accepted_traces.jsonl")
    ap.add_argument("--examples", default="data/sft/sft_examples.jsonl")
    ap.add_argument("--out", default="data/sft/SFT_REPORT.md")
    args = ap.parse_args()

    episodes = _read_jsonl(Path(args.accepted))
    examples = _read_jsonl(Path(args.examples))
    report = build_report(episodes, examples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_md(report), encoding="utf-8")
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    hc = report["leakage_healthcheck"]
    print(f"episodes={report['episodes']} examples={report['examples']} "
          f"unique_questions={report['unique_questions']}")
    print(f"leakage healthcheck clean: {hc['clean']}")
    print(f"report -> {out_path}")
    if not hc["clean"]:
        print("LEAKAGE DETECTED -- dataset is NOT safe to train on.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
