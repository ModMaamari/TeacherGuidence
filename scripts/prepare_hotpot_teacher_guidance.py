"""
Prepare HotpotQA data for the Teacher Guidance pipeline.

Loads HotpotQA examples and writes two JSONL files:

    <out_dir>/hotpot_<subset>_<split>_questions.jsonl
    <out_dir>/hotpot_<subset>_<split>_corpus.jsonl

Examples can be loaded either from the Hugging Face ``datasets`` library
(``hotpotqa/hotpot_qa``) or from a local raw HotpotQA JSON file via ``--input``.

Usage:
    python scripts/prepare_hotpot_teacher_guidance.py \
        --subset distractor --split validation --limit 20 \
        --out_dir data/datasets/hotpot_teacher_guidance

    # offline, from a downloaded raw HotpotQA json:
    python scripts/prepare_hotpot_teacher_guidance.py \
        --input hotpot_dev_distractor_v1.json --split validation --limit 20 \
        --out_dir data/datasets/hotpot_teacher_guidance
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.teacher_guidance.hotpot_converter import convert_examples  # noqa: E402


def _load_from_huggingface(subset: str, split: str, limit: int) -> list:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise SystemExit(
            "The 'datasets' library is required to load from Hugging Face.\n"
            "Install it (pip install datasets) or pass --input <raw_hotpot.json>."
        ) from exc

    ds = load_dataset("hotpotqa/hotpot_qa", subset, split=split, trust_remote_code=True)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def _load_from_file(path: Path, limit: int) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # some dumps wrap rows under a key
        data = data.get("data", data.get("examples", []))
    if limit and limit > 0:
        data = data[:limit]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="distractor", help="HotpotQA subset (distractor|fullwiki)")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--limit", type=int, default=20, help="Max examples (<=0 for all)")
    parser.add_argument("--input", help="Local raw HotpotQA JSON file (offline mode)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit")
    parser.add_argument("--seed", type=int, default=13, help="Shuffle seed")
    parser.add_argument(
        "--out_dir",
        default="data/datasets/hotpot_teacher_guidance",
        help="Output directory",
    )
    args = parser.parse_args()

    if args.input:
        examples = _load_from_file(Path(args.input), 0 if args.shuffle else args.limit)
    else:
        examples = _load_from_huggingface(
            args.subset, args.split, 0 if args.shuffle else args.limit
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(examples)
        if args.limit and args.limit > 0:
            examples = examples[: args.limit]

    question_rows, corpus_rows = convert_examples(examples, split=args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / f"hotpot_{args.subset}_{args.split}_questions.jsonl"
    c_path = out_dir / f"hotpot_{args.subset}_{args.split}_corpus.jsonl"

    with open(q_path, "w", encoding="utf-8") as f:
        for row in question_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(c_path, "w", encoding="utf-8") as f:
        for row in corpus_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(question_rows)} questions -> {q_path}")
    print(f"Wrote {len(corpus_rows)} corpus docs -> {c_path}")


if __name__ == "__main__":
    main()
