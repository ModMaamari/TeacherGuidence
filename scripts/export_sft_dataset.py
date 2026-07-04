"""
Export accepted traces to a chat-format SFT dataset (stage 2 of the pipeline).

Reads the accepted_traces.jsonl produced by filter_traces.py, optionally reduces to one
canonical (most optimal) trace per question, then emits chat-format training examples
(system/user/assistant) with the teacher guidance internalized into the student's thought.

Usage:
    python scripts/export_sft_dataset.py \
        --accepted data/sft/accepted_traces.jsonl \
        --out data/sft/sft_examples.jsonl --canonical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.sft_export import build_dataset  # noqa: E402
from agentsim.teacher_guidance.trace_selection import select_canonical  # noqa: E402
from agentsim.teacher_guidance.sft_diversity import cap_by_signature, diversity_report  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted", default="data/sft/accepted_traces.jsonl")
    ap.add_argument("--out", default="data/sft/sft_examples.jsonl")
    ap.add_argument("--canonical", action="store_true",
                    help="Keep only one optimal trace per question before exporting.")
    ap.add_argument("--max-per-tool-sequence", type=int, default=0,
                    help="Cap how many traces may share the same tool-sequence signature "
                         "(0 = no cap), keeping the shortest/cleanest -- prevents a few "
                         "patterns from dominating and overfitting the student.")
    args = ap.parse_args()

    episodes = _read_jsonl(Path(args.accepted))
    if args.canonical:
        episodes = select_canonical(episodes)
    diversity = diversity_report(episodes)
    if args.max_per_tool_sequence > 0:
        episodes = cap_by_signature(episodes, args.max_per_tool_sequence)

    examples = build_dataset(episodes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    by_kind: Dict[str, int] = {}
    for ex in examples:
        by_kind[ex["metadata"]["kind"]] = by_kind.get(ex["metadata"]["kind"], 0) + 1
    print(f"episodes used:  {len(episodes)}")
    print(f"examples:       {len(examples)} ({by_kind})")
    print(f"diversity:      {diversity['unique_signatures']} unique tool-sequences, "
          f"normalized entropy {diversity['normalized_entropy']}")
    print(f"sft examples -> {out_path}")


if __name__ == "__main__":
    main()
