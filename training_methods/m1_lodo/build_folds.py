"""Build the leave-one-dataset-out training/test splits for m1-LODO.

Design (see PLAN.md): every qid is hashed once into a 10% ``heldin_test`` pool that is
NEVER trained on, and a 90% ``trainable`` pool. For fold k the model trains on the
trainable, *correct-only* episodes of the three datasets other than k, and is tested on
(a) each training dataset's held-in questions -- in-distribution but unseen -- and
(b) every question of dataset k, which the model has never seen in any form.

Splitting by qid hash rather than by row means a question can never leak between train and
test through a second episode, and the assignment is identical on every re-run.

Only episodes with ``final_metrics.answer_correct`` become training examples: a wrong
trajectory teaches the student to reproduce a wrong trajectory. Test sets, by contrast,
use ALL questions of the dataset regardless of collection-time correctness -- the model is
being asked to solve them itself.

Outputs under ``--out``::

    fold_<k>/train.jsonl dev.jsonl          SFT examples (prompt/completion messages)
    fold_<k>/manifest.json                  qid counts, example counts, disjointness proof
    tests/heldin_<ds>_questions.jsonl       held-in test questions (per dataset)
    tests/unseen_<ds>_questions.jsonl       full-dataset test questions (per dataset)
    splits.json                             the qid -> pool assignment, for auditing
    stats.json                              per-fold and per-dataset counts

Usage::

    python training_methods/m1_lodo/build_folds.py \
        --episodes data/datasets/tg_v1_episodes/episodes.jsonl \
        --data-root data/datasets/tg_v1 --out training_methods/m1_lodo/data
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.episode_lib import build_episode_examples  # noqa: E402

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
#: Gold values where a bare string test carries no information (yes/no datasets).
BOOLEAN_GOLD = {"yes", "no", "true", "false"}
#: Required target keys per example kind. A target missing them is malformed -- usually a
#: generation that was truncated mid-JSON -- and would teach the model an incomplete shape.
REQUIRED_KEYS = {"action": {"thought", "action"}, "plan": {"plan_summary", "steps"}}
#: prepare_dataset.py file stems, for locating the question/corpus files per dataset.
STEMS = {
    "hotpotqa": ("hotpotqa", "hotpotqa_train"),
    "2wikimultihopqa": ("2wikimultihopqa", "2wikimultihopqa_train"),
    "musique": ("musique", "musique_train"),
    "strategyqa": ("strategyqa", "strategyqa_train"),
}


def pool_of(qid: str, heldin_fraction: float, salt: str = "m1lodo") -> str:
    """Deterministic 10%/90% assignment. Salted so it is independent of any other
    qid-hash split in the project (the collection sharding also hashes qids)."""
    h = int(hashlib.sha256(f"{salt}:{qid}".encode()).hexdigest(), 16) % 10_000
    return "heldin_test" if h < heldin_fraction * 10_000 else "trainable"


def dev_of(qid: str, dev_fraction: float, salt: str = "m1lodo-dev") -> bool:
    h = int(hashlib.sha256(f"{salt}:{qid}".encode()).hexdigest(), 16) % 10_000
    return h < dev_fraction * 10_000


def _text(part: Any) -> str:
    if isinstance(part, list):
        return " ".join(m.get("content", "") for m in part)
    if isinstance(part, dict):
        return part.get("content", "")
    return str(part or "")


def asserts_ungrounded_gold(ex: Dict[str, Any], gold: str) -> bool:
    """True when the target states the gold answer but the prompt never showed it.

    These come from episodes that were *correct but not grounded*: the student committed
    the right answer without the evidence being visible (often a forced finish, sometimes
    with the teacher explicitly saying the evidence was still missing). Training on them
    teaches the model to assert an answer it cannot derive -- the opposite of what this
    corpus is for -- so they are dropped even though their episode is "correct".

    Boolean golds are exempt: "yes"/"no" appearing in a target says nothing.
    """
    if not gold or gold.lower() in BOOLEAN_GOLD:
        return False
    pat = re.compile(r"\b" + re.escape(gold) + r"\b", re.I)
    return bool(pat.search(_text(ex.get("completion")))) and not pat.search(_text(ex.get("prompt")))


def target_wellformed(ex: Dict[str, Any]) -> bool:
    """The completion must parse as JSON and carry the keys its kind requires."""
    try:
        obj = json.loads(_text(ex.get("completion")))
    except Exception:
        return False
    kind = (ex.get("metadata") or {}).get("kind", "action")
    return REQUIRED_KEYS.get(kind, REQUIRED_KEYS["action"]) <= set(obj)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/datasets/tg_v1_episodes/episodes.jsonl")
    ap.add_argument("--data-root", default="data/datasets/tg_v1")
    ap.add_argument("--out", default="training_methods/m1_lodo/data")
    ap.add_argument("--heldin-fraction", type=float, default=0.10)
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap episodes read")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- pass 1: read episodes, assign pools, build examples for correct ones ---
    pools: Dict[str, str] = {}
    examples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    n = 0
    for ep in read_jsonl(Path(args.episodes)):
        if args.limit and n >= args.limit:
            break
        n += 1
        ds, qid = ep.get("dataset"), str(ep.get("qid"))
        if ds not in STEMS:
            continue
        pool = pool_of(qid, args.heldin_fraction)
        pools[f"{ds}/{qid}"] = pool
        counts[ds][pool] += 1
        correct = bool((ep.get("final_metrics") or {}).get("answer_correct"))
        counts[ds]["correct"] += correct
        if not correct or pool != "trainable":
            continue
        gold = (ep.get("gold_answer") or "").strip()
        built = []
        for ex in build_episode_examples(ep, run="tg_v1"):
            if asserts_ungrounded_gold(ex, gold):
                counts[ds]["dropped_ungrounded_gold"] += 1
                continue
            if not target_wellformed(ex):
                counts[ds]["dropped_malformed_target"] += 1
                continue
            ex.setdefault("metadata", {})
            ex["metadata"]["dataset"] = ds
            ex["metadata"]["qid"] = qid
            built.append(ex)
        if not built:
            counts[ds]["episodes_emptied"] += 1
            continue
        examples[ds].extend(built)
        counts[ds]["train_examples"] += len(built)
        counts[ds]["train_episodes"] += 1

    # --- pass 2: test question files, straight from the prepared datasets ---
    tests_dir = out / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_counts = collections.Counter()
    for ds, (d, stem) in STEMS.items():
        qpath = Path(args.data_root) / d / f"{stem}_questions.jsonl"
        rows = list(read_jsonl(qpath))
        heldin = [r for r in rows if pools.get(f"{ds}/{r['id']}") == "heldin_test"]
        write_jsonl(tests_dir / f"heldin_{ds}_questions.jsonl", heldin)
        write_jsonl(tests_dir / f"unseen_{ds}_questions.jsonl", rows)
        test_counts[f"heldin_{ds}"] = len(heldin)
        test_counts[f"unseen_{ds}"] = len(rows)

    # --- pass 3: per-fold train/dev, with a disjointness proof ---
    stats: Dict[str, Any] = {"per_dataset": {k: dict(v) for k, v in counts.items()},
                             "test_sets": dict(test_counts), "folds": {}}
    for k in DATASETS:
        train_ds = [d for d in DATASETS if d != k]
        train_rows, dev_rows = [], []
        for d in train_ds:
            for ex in examples[d]:
                (dev_rows if dev_of(ex["metadata"]["qid"], args.dev_fraction) else train_rows).append(ex)
        fold_dir = out / f"fold_{k}"
        write_jsonl(fold_dir / "train.jsonl", train_rows)
        write_jsonl(fold_dir / "dev.jsonl", dev_rows)

        train_qids = {f'{e["metadata"]["dataset"]}/{e["metadata"]["qid"]}'
                      for e in train_rows + dev_rows}
        # Disjointness: no training qid may appear in ANY of this fold's test sets.
        overlaps = {}
        for d in train_ds:
            heldin_qids = {f"{d}/{r['id']}" for r in read_jsonl(tests_dir / f"heldin_{d}_questions.jsonl")}
            overlaps[f"heldin_{d}"] = len(train_qids & heldin_qids)
        unseen_qids = {f"{k}/{r['id']}" for r in read_jsonl(tests_dir / f"unseen_{k}_questions.jsonl")}
        overlaps[f"unseen_{k}"] = len(train_qids & unseen_qids)

        manifest = {
            "fold": k, "held_out_dataset": k, "train_datasets": train_ds,
            "train_examples": len(train_rows), "dev_examples": len(dev_rows),
            "train_questions": len(train_qids),
            "test_sets": [f"heldin_{d}" for d in train_ds] + [f"unseen_{k}"],
            "train_test_overlap": overlaps,
            "per_dataset_examples": {d: len(examples[d]) for d in train_ds},
        }
        (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        stats["folds"][k] = manifest
        bad = {kk: v for kk, v in overlaps.items() if v}
        flag = "OK" if not bad else f"OVERLAP {bad}"
        print(f"  fold {k:<18} train {len(train_rows):>6} ex / dev {len(dev_rows):>5} ex "
              f"from {len(train_qids):>5} questions  [{flag}]")

    (out / "splits.json").write_text(json.dumps(pools, indent=0), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nper-dataset:")
    for ds in DATASETS:
        c = counts[ds]
        print(f"  {ds:<18} trainable {c['trainable']:>5} | heldin {c['heldin_test']:>4} | "
              f"correct {c['correct']:>5} | train episodes {c['train_episodes']:>5} "
              f"-> {c['train_examples']:>6} examples")
    print("\ntest sets:", dict(test_counts))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
