#!/usr/bin/env python
"""Build the training/test splits from the consolidated episodes and question files.

Two groups are produced from ONE question-level pool assignment (``tgd/splits.py``):

Group B -- ``uniform`` (train on everything, test on the held-out 10%)
    uniform/train.jsonl, dev.jsonl     SFT examples from the trainable (90%) pool of
                                       every dataset, correct episodes only
    test/heldout_<ds>_questions.jsonl  the held-out 10% of each dataset (never trained on
                                       by ANY split built here)

Group A -- ``lodo`` (leave-one-dataset-out: train on three, test on the fourth)
    lodo/fold_<ds>/train.jsonl, dev.jsonl   SFT examples from the trainable pools of the
                                            three OTHER datasets
    test/full_<ds>_questions.jsonl          every question of dataset <ds> -- the unseen
                                            dataset for fold_<ds>
    ...plus the same heldout_<other>_questions.jsonl files for the three training datasets

Also written: ``pools.json`` (qid -> pool), ``stats.json``, and a ``manifest.json`` per
split with counts and the train/test qid-overlap proof (always 0). Run
``scripts/check_leakage.py`` afterwards for the independent check.

Only episodes whose final answer was correct become training examples. Test files use
ALL questions of a dataset regardless of collection-time correctness. Two hygiene
filters run on the training side: a question whose text duplicates a held-out question
(same text, different id -- datasets contain such duplicates) is excluded from training,
and any example still containing the ``[answer hidden]`` placeholder is dropped.

Usage::

    python scripts/build_splits.py --episodes data/episodes/episodes.jsonl.gz \
        --questions data/questions --out data/splits
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401
from tgd import DATASETS
from tgd.episode_lib import build_episode_examples
from tgd.io import load_jsonl, question_file, read_jsonl, write_jsonl
from tgd.splits import DEFAULT_DEV_SALT, DEFAULT_SALT, is_dev, pool_of

BOOLEAN_GOLD = {"yes", "no", "true", "false"}
PLACEHOLDER = "[answer hidden]"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
REQUIRED_KEYS = {"action": {"thought", "action"}, "plan": {"plan_summary", "steps"}}


def _text(part: Any) -> str:
    if isinstance(part, list):
        return " ".join(m.get("content", "") for m in part)
    if isinstance(part, dict):
        return part.get("content", "")
    return str(part or "")


def asserts_ungrounded_gold(ex: Dict[str, Any], gold: str) -> bool:
    """Target states the gold answer although the prompt never showed it: the episode was
    correct but not grounded (a lucky or forced finish). Training on it teaches the model
    to assert answers it cannot derive, so it is dropped. Boolean golds are exempt."""
    if not gold or gold.lower() in BOOLEAN_GOLD:
        return False
    pat = re.compile(r"\b" + re.escape(gold) + r"\b", re.I)
    return bool(pat.search(_text(ex.get("completion")))) and not pat.search(_text(ex.get("prompt")))


def target_wellformed(ex: Dict[str, Any]) -> bool:
    """The completion must parse as JSON and carry the keys its kind requires
    (a truncated generation would otherwise teach an incomplete shape)."""
    try:
        obj = json.loads(_text(ex.get("completion")))
    except Exception:
        return False
    kind = (ex.get("metadata") or {}).get("kind", "action")
    return REQUIRED_KEYS.get(kind, REQUIRED_KEYS["action"]) <= set(obj)


def write_split(name: str, out: Path, train_ds: List[str], examples, dev_fraction, dev_salt,
                test_qids: Dict[str, set], unseen: str | None) -> Dict[str, Any]:
    train_rows, dev_rows = [], []
    for d in train_ds:
        for ex in examples[d]:
            (dev_rows if is_dev(ex["metadata"]["qid"], dev_fraction, dev_salt) else train_rows).append(ex)
    d_out = out / name
    write_jsonl(d_out / "train.jsonl", train_rows)
    write_jsonl(d_out / "dev.jsonl", dev_rows)
    train_qids = {f'{e["metadata"]["dataset"]}/{e["metadata"]["qid"]}' for e in train_rows + dev_rows}
    tests = {f"heldout_{d}": test_qids[f"heldout_{d}"] for d in train_ds}
    if unseen:
        tests[f"full_{unseen}"] = test_qids[f"full_{unseen}"]
    overlaps = {t: len(train_qids & q) for t, q in tests.items()}
    manifest = {
        "split": name, "train_datasets": train_ds, "unseen_dataset": unseen,
        "train_examples": len(train_rows), "dev_examples": len(dev_rows),
        "train_questions": len(train_qids),
        "per_dataset_examples": {d: len(examples[d]) for d in train_ds},
        "test_sets": {t: len(q) for t, q in tests.items()},
        "train_test_qid_overlap": overlaps,
    }
    (d_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    flag = "OK" if not any(overlaps.values()) else f"OVERLAP {overlaps}"
    print(f"  {name:<22} train {len(train_rows):>6} / dev {len(dev_rows):>5} examples "
          f"from {len(train_qids):>5} questions   [{flag}]")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="data/episodes/episodes.jsonl.gz")
    ap.add_argument("--questions", default="data/questions")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--heldout-fraction", type=float, default=0.10)
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--salt", default=DEFAULT_SALT, help="hash salt of the held-out assignment")
    ap.add_argument("--dev-salt", default=DEFAULT_DEV_SALT)
    ap.add_argument("--limit", type=int, default=None, help="cap episodes read (smoke tests)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = list(args.datasets)

    # ---- pass 0: question files -> the held-out question texts (duplicate guard)
    questions: Dict[str, List[Dict[str, Any]]] = {ds: load_jsonl(question_file(args.questions, ds)) for ds in datasets}
    heldout_text = {norm(r["query"]) for ds in datasets for r in questions[ds]
                    if pool_of(str(r["id"]), args.heldout_fraction, args.salt) == "heldout_test"}

    # ---- pass 1: episodes -> pools + SFT examples (trainable & correct only)
    pools: Dict[str, str] = {}
    examples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    for ep in read_jsonl(args.episodes, args.limit):
        ds, qid = ep.get("dataset"), str(ep.get("qid"))
        if ds not in datasets:
            continue
        pool = pool_of(qid, args.heldout_fraction, args.salt)
        pools[f"{ds}/{qid}"] = pool
        counts[ds][pool] += 1
        correct = bool((ep.get("final_metrics") or {}).get("answer_correct"))
        counts[ds]["correct"] += int(correct)
        if not correct or pool != "trainable":
            continue
        if norm(ep.get("query", "")) in heldout_text:
            counts[ds]["dropped_duplicate_of_heldout"] += 1
            continue
        gold = (ep.get("gold_answer") or "").strip()
        built = []
        for ex in build_episode_examples(ep, run="episodes"):
            if asserts_ungrounded_gold(ex, gold):
                counts[ds]["dropped_ungrounded_gold"] += 1
                continue
            if not target_wellformed(ex):
                counts[ds]["dropped_malformed_target"] += 1
                continue
            if PLACEHOLDER in _text(ex.get("prompt")) or PLACEHOLDER in _text(ex.get("completion")):
                counts[ds]["dropped_placeholder"] += 1
                continue
            ex.setdefault("metadata", {})
            ex["metadata"]["dataset"] = ds
            ex["metadata"]["qid"] = qid
            ex["metadata"]["query"] = ep.get("query", "")
            built.append(ex)
        if not built:
            counts[ds]["episodes_without_examples"] += 1
            continue
        examples[ds].extend(built)
        counts[ds]["train_examples"] += len(built)
        counts[ds]["train_episodes"] += 1

    # ---- pass 2: test question files
    test_dir = out / "test"
    test_qids: Dict[str, set] = {}
    for ds in datasets:
        rows = questions[ds]
        if args.limit:  # smoke: keep the questions that have episodes
            rows = [r for r in rows if f"{ds}/{r['id']}" in pools]
        heldout = [r for r in rows if pool_of(str(r["id"]), args.heldout_fraction, args.salt) == "heldout_test"]
        write_jsonl(test_dir / f"heldout_{ds}_questions.jsonl", heldout)
        write_jsonl(test_dir / f"full_{ds}_questions.jsonl", rows)
        test_qids[f"heldout_{ds}"] = {f"{ds}/{r['id']}" for r in heldout}
        test_qids[f"full_{ds}"] = {f"{ds}/{r['id']}" for r in rows}

    # ---- pass 3: splits
    stats: Dict[str, Any] = {"per_dataset": {k: dict(v) for k, v in counts.items()},
                             "test_sets": {k: len(v) for k, v in test_qids.items()}, "splits": {}}
    print("group B (uniform):")
    stats["splits"]["uniform"] = write_split("uniform", out, datasets, examples, args.dev_fraction,
                                             args.dev_salt, test_qids, None)
    print("group A (leave-one-dataset-out):")
    for k in datasets:
        stats["splits"][f"lodo/fold_{k}"] = write_split(
            f"lodo/fold_{k}", out, [d for d in datasets if d != k], examples,
            args.dev_fraction, args.dev_salt, test_qids, k)
    (out / "pools.json").write_text(json.dumps(pools, indent=0), encoding="utf-8")
    stats["config"] = {k: getattr(args, k) for k in ("heldout_fraction", "dev_fraction", "salt", "dev_salt")}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nper dataset:")
    for ds in datasets:
        c = counts[ds]
        print(f"  {ds:<18} episodes {c['trainable'] + c['heldout_test']:>5} | heldout {c['heldout_test']:>4} | "
              f"correct {c['correct']:>5} | train episodes {c['train_episodes']:>5} -> {c['train_examples']:>6} examples")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
