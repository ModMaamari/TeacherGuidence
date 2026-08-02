"""Prepare any supported multi-hop QA dataset into canonical teacher-guidance files.

One CLI for every source, so all datasets go through the same conversion, the same
validation, and the same manifest. Writes:

    <out>/<dataset>_<split>_questions.jsonl
    <out>/<dataset>_<split>_corpus.jsonl
    <out>/<dataset>_<split>_manifest.json     conversion stats, license, provenance

Examples::

    # HotpotQA from HuggingFace
    .venv/bin/python scripts/prepare_dataset.py --dataset hotpotqa \
        --hf hotpotqa/hotpot_qa --hf-config distractor --split validation \
        --limit 3000 --out data/datasets/tg_v1/hotpotqa

    # 2WikiMultihopQA from the official release JSON
    .venv/bin/python scripts/prepare_dataset.py --dataset 2wikimultihopqa \
        --input ~/data/2wiki/train.json --split train --limit 3000 \
        --out data/datasets/tg_v1/2wikimultihopqa

    # MuSiQue from the official JSONL
    .venv/bin/python scripts/prepare_dataset.py --dataset musique \
        --input ~/data/musique/musique_ans_v1.0_train.jsonl --split train \
        --limit 3000 --out data/datasets/tg_v1/musique

    # StrategyQA needs its separate paragraph corpus
    .venv/bin/python scripts/prepare_dataset.py --dataset strategyqa \
        --input ~/data/strategyqa/strategyqa_train.json \
        --paragraphs ~/data/strategyqa/strategyqa_train_paragraphs.json \
        --split train --out data/datasets/tg_v1/strategyqa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.converters import (  # noqa: E402
    convert_dataset,
    dataset_names,
    get_spec,
)
from agentsim.teacher_guidance.provenance import framework_commit  # noqa: E402


def load_records(path: Path) -> Any:
    """Load a .json (list or dict) or .jsonl file."""
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_from_hf(hf_id: str, hf_config: str | None, split: str, limit: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        raise SystemExit("`datasets` is not installed; use --input with a local file instead")
    ds = load_dataset(hf_id, hf_config, split=split, trust_remote_code=True)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, choices=dataset_names())
    ap.add_argument("--split", default="train", help="split label recorded on every row")
    ap.add_argument("--input", help="local source file (.json or .jsonl)")
    ap.add_argument("--hf", help="HuggingFace dataset id (alternative to --input)")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--hf-split", default=None, help="defaults to --split")
    ap.add_argument("--paragraphs", help="StrategyQA paragraph corpus JSON")
    ap.add_argument("--limit", type=int, default=0, help="max examples (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle before --limit")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--num-distractors", type=int, default=8,
                    help="StrategyQA only: constructed distractors per question")
    ap.add_argument("--strict", action="store_true",
                    help="fail on the first invalid example instead of skipping it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    if not args.input and not args.hf:
        raise SystemExit("provide --input <file> or --hf <dataset-id>")

    # ---- load source examples ------------------------------------------------
    if args.input:
        examples = load_records(Path(args.input).expanduser())
        if isinstance(examples, dict):
            raise SystemExit(f"{args.input} is a JSON object; expected a list of examples")
    else:
        examples = load_from_hf(args.hf, args.hf_config, args.hf_split or args.split,
                                0 if args.shuffle else args.limit)

    if args.shuffle:
        random.Random(args.seed).shuffle(examples)
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    # ---- converter-specific extras -------------------------------------------
    kwargs: Dict[str, Any] = {}
    if "paragraphs" in spec.requires:
        if not args.paragraphs:
            raise SystemExit(f"{args.dataset} requires --paragraphs <corpus.json>")
        paragraphs = load_records(Path(args.paragraphs).expanduser())
        if not isinstance(paragraphs, dict):
            raise SystemExit("--paragraphs must be a JSON object mapping para_id -> paragraph")
        kwargs["paragraphs"] = paragraphs
        kwargs["distractor_pool"] = sorted(paragraphs.keys())
        kwargs["num_distractors"] = args.num_distractors
        kwargs["seed"] = args.seed

    print(f"[prepare] {args.dataset} split={args.split} examples={len(examples)}")
    questions, corpus, stats = convert_dataset(
        args.dataset, examples, args.split, strict=args.strict, **kwargs
    )
    if not questions:
        raise SystemExit(f"no examples converted -- stats: {stats}")

    # ---- write ----------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q_path = out / f"{args.dataset}_{args.split}_questions.jsonl"
    c_path = out / f"{args.dataset}_{args.split}_corpus.jsonl"
    with open(q_path, "w", encoding="utf-8") as w:
        for row in questions:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(c_path, "w", encoding="utf-8") as w:
        for row in corpus:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        **stats,
        "questions_file": q_path.name,
        "corpus_file": c_path.name,
        "questions_sha256": _sha256_file(q_path),
        "corpus_sha256": _sha256_file(c_path),
        "framework_commit": framework_commit(),
        "license": spec.license,
        "homepage": spec.homepage,
        "notes": spec.notes,
        "source_input": str(args.input or f"hf:{args.hf}"),
        "limit": args.limit,
        "shuffle": args.shuffle,
        "seed": args.seed,
        **({"num_distractors": args.num_distractors} if "paragraphs" in spec.requires else {}),
    }
    (out / f"{args.dataset}_{args.split}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    skipped = stats["skipped_total"]
    print(f"[prepare] converted {stats['converted']} questions, {stats['corpus_docs']} docs "
          f"({skipped} skipped)")
    if stats["skipped_reasons"]:
        for reason, n in sorted(stats["skipped_reasons"].items(), key=lambda x: -x[1]):
            print(f"           skip: {reason} x{n}")
    print(f"[prepare] -> {q_path}")
    print(f"[prepare] -> {c_path}")


if __name__ == "__main__":
    main()
