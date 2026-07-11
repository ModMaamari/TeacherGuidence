"""Aggregate eval metrics.json files into a markdown results table.

Usage:
    python training_methods/common/compare_evals.py --out results.md \
        label1=path/to/metrics.json label2=path/to/metrics.json ...

Each metrics.json is the output of common/eval_agent.py. The newest run directory
matching a glob may be selected with '@latest', e.g.:
    m1_golden=training_methods/m1_sft/runs/eval_golden100/@latest
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

COLS = ["n", "budget", "em", "f1", "cover_match", "doc_recall", "mean_steps",
        "invalid_action_steps"]


def resolve(path: str) -> Path:
    if path.endswith("@latest"):
        base = path[: -len("@latest")].rstrip("/")
        candidates = sorted(glob.glob(base + "/*/metrics.json"))
        if not candidates:
            raise FileNotFoundError(f"no metrics.json under {base}")
        return Path(candidates[-1])
    p = Path(path)
    return p if p.name == "metrics.json" else p / "metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Evaluation results")
    ap.add_argument("entries", nargs="+", help="label=path pairs")
    args = ap.parse_args()

    rows = []
    for entry in args.entries:
        label, path = entry.split("=", 1)
        p = resolve(path)
        m = json.load(open(p))
        rows.append((label, m, str(p)))

    lines = [f"# {args.title}",
             f"\n_Generated {datetime.now(timezone.utc).isoformat()}_\n",
             "| run | " + " | ".join(COLS) + " |",
             "|---|" + "---|" * len(COLS)]
    for label, m, _ in rows:
        lines.append("| " + label + " | " + " | ".join(str(m.get(c, "")) for c in COLS) + " |")
    lines.append("\nSources:")
    for label, _, p in rows:
        lines.append(f"- {label}: `{p}`")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
