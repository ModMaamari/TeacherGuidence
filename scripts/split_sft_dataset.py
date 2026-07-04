"""
Split an SFT dataset into train/val/test by question (stage 3 of the pipeline).

Computes a per-qid, stratified split from the accepted episodes (which carry the gold
answer needed for stratification), then partitions the sft_examples.jsonl by each example's
qid so no question crosses splits.

Usage:
    python scripts/split_sft_dataset.py \
        --accepted data/sft/accepted_traces.jsonl \
        --examples data/sft/sft_examples.jsonl --out-dir data/sft
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.dataset_split import SPLIT_NAMES, split_by_question  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted", default="data/sft/accepted_traces.jsonl")
    ap.add_argument("--examples", default="data/sft/sft_examples.jsonl")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    episodes = _read_jsonl(Path(args.accepted))
    examples = _read_jsonl(Path(args.examples))
    assignment = split_by_question(episodes, tuple(args.ratios), args.seed)

    buckets: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLIT_NAMES}
    dropped = 0
    for ex in examples:
        qid = ex.get("metadata", {}).get("qid")
        split = assignment.get(qid)
        if split is None:
            dropped += 1
            continue
        buckets[split].append(ex)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in buckets.items():
        with open(out_dir / f"sft_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    q_by_split = Counter(assignment.values())
    print(f"questions: {dict(q_by_split)} (total {len(assignment)})")
    for split in SPLIT_NAMES:
        print(f"  {split}: {len(buckets[split])} examples -> {out_dir / f'sft_{split}.jsonl'}")
    if dropped:
        print(f"  (dropped {dropped} examples whose qid had no split)")


if __name__ == "__main__":
    main()
