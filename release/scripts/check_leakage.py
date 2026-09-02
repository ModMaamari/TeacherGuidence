#!/usr/bin/env python
"""Independent leakage audit of the splits in ``data/splits``.

Re-derives everything from the files on disk (it does not trust the manifests) and
checks, for every split and every test set it is evaluated on:

1. **qid disjointness** -- no train/dev example comes from a test question
   (``metadata.dataset/qid`` vs the test question ids).
2. **question-text disjointness** -- no test question's text appears anywhere in a
   training prompt or completion (catches duplicated questions under different ids).
3. **placeholder hygiene** -- no ``[answer hidden]`` token survives in any target.
4. **pool consistency** -- every train/dev qid hashes to the trainable pool and every
   held-out question to the held-out pool with the recorded salt (proves the split is
   the deterministic hash, not a hand edit).
5. **schema** -- every example is a prompt/completion message list, every completion
   is a JSON object with the keys its kind requires.
6. **within-dataset near-duplicates** -- test questions whose normalised text equals a
   training question's text (reported; these are dataset artefacts, not a split error,
   but a reader should know the count).

Writes ``<splits>/leakage_report.json`` and exits non-zero on any hard failure (1-5).

Usage::

    python scripts/check_leakage.py --splits data/splits --questions data/questions
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401
from tgd.io import load_jsonl, read_json, read_jsonl
from tgd.splits import pool_of

PLACEHOLDER = "[answer hidden]"
REQUIRED_KEYS = {"action": {"thought", "action"}, "plan": {"plan_summary", "steps"}}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _content(part: Any) -> str:
    if isinstance(part, list):
        return "\n".join(str(m.get("content", "")) for m in part)
    if isinstance(part, dict):
        return str(part.get("content", ""))
    return str(part or "")


def audit_split(split_dir: Path, test_sets: Dict[str, List[Dict[str, Any]]], salt: str,
                heldout_fraction: float) -> Dict[str, Any]:
    rows = list(read_jsonl(split_dir / "train.jsonl")) + list(read_jsonl(split_dir / "dev.jsonl"))
    problems: List[str] = []
    train_qids: Set[str] = set()
    train_text_by_ds: Dict[str, Set[str]] = collections.defaultdict(set)
    train_blob_parts: List[str] = []
    schema_bad = placeholder_bad = pool_bad = 0
    for ex in rows:
        md = ex.get("metadata") or {}
        ds, qid = md.get("dataset"), str(md.get("qid"))
        train_qids.add(f"{ds}/{qid}")
        if pool_of(qid, heldout_fraction, salt) != "trainable":
            pool_bad += 1
        p, c = _content(ex.get("prompt")), _content(ex.get("completion"))
        if PLACEHOLDER in c or PLACEHOLDER in p:
            placeholder_bad += 1
        try:
            obj = json.loads(c)
            kind = md.get("kind", "action")
            if not REQUIRED_KEYS.get(kind, REQUIRED_KEYS["action"]) <= set(obj):
                schema_bad += 1
        except Exception:
            schema_bad += 1
        if not (isinstance(ex.get("prompt"), list) and isinstance(ex.get("completion"), list)):
            schema_bad += 1
        q = md.get("query") or ""
        if q:
            train_text_by_ds[ds].add(norm(q))
        train_blob_parts.append(norm(p) + " " + norm(c))
    train_blob = "\n".join(train_blob_parts)

    per_test = {}
    for name, qrows in test_sets.items():
        ds = name.split("_", 1)[1]
        test_qids = {f"{ds}/{r['id']}" for r in qrows}
        qid_overlap = sorted(train_qids & test_qids)
        # text containment: a test question appearing verbatim inside any training text
        text_hits = [r["id"] for r in qrows if len(norm(r["query"])) >= 20 and norm(r["query"]) in train_blob]
        near_dups = [r["id"] for r in qrows if norm(r["query"]) in train_text_by_ds.get(ds, set())]
        heldout_pool_bad = 0
        if name.startswith("heldout_"):
            heldout_pool_bad = sum(1 for r in qrows if pool_of(str(r["id"]), heldout_fraction, salt) != "heldout_test")
        per_test[name] = {"n_test": len(qrows), "qid_overlap": len(qid_overlap),
                          "question_text_in_train": len(text_hits),
                          "near_duplicate_questions": len(near_dups),
                          "heldout_pool_mismatch": heldout_pool_bad}
        if qid_overlap:
            problems.append(f"{name}: {len(qid_overlap)} test qids in train (e.g. {qid_overlap[:3]})")
        if text_hits:
            problems.append(f"{name}: {len(text_hits)} test question texts found in training text (e.g. {text_hits[:3]})")
        if heldout_pool_bad:
            problems.append(f"{name}: {heldout_pool_bad} held-out questions do not hash to the held-out pool")
    if pool_bad:
        problems.append(f"{pool_bad} train/dev examples whose qid hashes to the held-out pool")
    if placeholder_bad:
        problems.append(f"{placeholder_bad} examples still contain '{PLACEHOLDER}'")
    if schema_bad:
        problems.append(f"{schema_bad} malformed examples (not prompt/completion messages or bad target JSON)")
    return {"train_dev_examples": len(rows), "train_questions": len(train_qids),
            "pool_mismatch": pool_bad, "placeholder": placeholder_bad, "schema_bad": schema_bad,
            "tests": per_test, "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--questions", default="data/questions")
    args = ap.parse_args()
    root = Path(args.splits)
    stats = read_json(root / "stats.json")
    salt = stats["config"]["salt"]; hf = stats["config"]["heldout_fraction"]

    tests_dir = root / "test"
    test_files = {p.stem.replace("_questions", ""): load_jsonl(p) for p in sorted(tests_dir.glob("*_questions.jsonl"))}
    # the held-out sets must be disjoint from the full sets of OTHER datasets trivially,
    # and heldout_<ds> must be a subset of full_<ds>
    subset_ok = all({r["id"] for r in test_files[f"heldout_{d}"]} <= {r["id"] for r in test_files[f"full_{d}"]}
                    for d in {k.split("_", 1)[1] for k in test_files if k.startswith("heldout_")})

    report: Dict[str, Any] = {"salt": salt, "heldout_fraction": hf, "heldout_subset_of_full": subset_ok,
                              "splits": {}}
    hard_fail = not subset_ok
    for split_dir in [root / "uniform"] + sorted((root / "lodo").glob("fold_*")):
        if not (split_dir / "train.jsonl").exists():
            continue
        man = read_json(split_dir / "manifest.json")
        wanted = {f"heldout_{d}": test_files[f"heldout_{d}"] for d in man["train_datasets"]}
        if man.get("unseen_dataset"):
            wanted[f"full_{man['unseen_dataset']}"] = test_files[f"full_{man['unseen_dataset']}"]
        res = audit_split(split_dir, wanted, salt, hf)
        name = str(split_dir.relative_to(root))
        report["splits"][name] = res
        status = "OK" if not res["problems"] else "FAIL"
        hard_fail |= bool(res["problems"])
        print(f"{name:<22} {status}  train/dev {res['train_dev_examples']:>6} ex, {res['train_questions']:>5} q | " +
              ", ".join(f"{t}: qid∩={v['qid_overlap']} text∩={v['question_text_in_train']} dup≈{v['near_duplicate_questions']}"
                        for t, v in res["tests"].items()))
        for pr in res["problems"]:
            print(f"    !! {pr}")
    report["status"] = "FAIL" if hard_fail else "OK"
    (root / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'LEAKAGE CHECK FAILED' if hard_fail else 'no leakage found'} -> {root / 'leakage_report.json'}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
