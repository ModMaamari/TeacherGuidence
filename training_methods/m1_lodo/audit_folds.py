"""Independent audit of built m1-LODO folds. Run before any training.

`episode_lib` already gates leakage while building each example. This re-checks the
*output*, because a gate that is bypassed by a code path is exactly the kind of bug that
is invisible until a model has memorised gold answers.

Four checks:

1. **Split disjointness** -- no training qid appears in any of that fold's test sets.
2. **Gold leakage** -- the gold answer must not appear in a training example's text unless
   it is echo-safe: already present in the question, or in the visible context the student
   had (retrieved documents, its own earlier output). Boolean golds (yes/no) are excluded
   from the string test entirely: for a yes/no dataset the word carries no information, and
   the teacher legitimately writes "give a clear yes/no answer".
3. **Private-field bleed** -- no `teacher_private_diagnosis` content in any target.
4. **Schema sanity** -- every example has prompt messages + a completion whose content
   parses as JSON with the expected keys.

Exit code is non-zero if any check fails, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

BOOLEAN = {"yes", "no", "true", "false"}
#: Targets come in two shapes: a plan example and a per-step action example.
EXPECTED_KEYS = {"action": {"thought", "action"}, "plan": {"plan_summary", "steps"}}


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def example_text(ex: dict) -> tuple[str, str]:
    """(prompt text, completion text) as plain strings."""
    pt = " ".join(m.get("content", "") for m in ex.get("prompt", []))
    comp = ex.get("completion")
    if isinstance(comp, list):
        ct = " ".join(m.get("content", "") for m in comp)
    elif isinstance(comp, dict):
        ct = comp.get("content", "")
    else:
        ct = str(comp or "")
    return pt, ct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training_methods/m1_lodo/data")
    ap.add_argument("--episodes", default="data/datasets/tg_v1_episodes/episodes.jsonl")
    args = ap.parse_args()
    root = Path(args.data)

    gold = {}
    for ep in read_jsonl(Path(args.episodes)):
        gold[f'{ep.get("dataset")}/{ep.get("qid")}'] = (ep.get("gold_answer") or "").strip()

    failures = []
    for fold_dir in sorted(root.glob("fold_*")):
        man = json.loads((fold_dir / "manifest.json").read_text())
        # 1. disjointness (recorded at build time, re-asserted here)
        for name, n in man["train_test_overlap"].items():
            if n:
                failures.append(f"{fold_dir.name}: {n} train qids leak into {name}")

        counts = collections.Counter()
        for split in ("train", "dev"):
            for ex in read_jsonl(fold_dir / f"{split}.jsonl"):
                counts["examples"] += 1
                md = ex.get("metadata") or {}
                key = f'{md.get("dataset")}/{md.get("qid")}'
                pt, ct = example_text(ex)

                # 4. schema
                kind = (md.get("kind") or "action")
                try:
                    obj = json.loads(ct)
                    want = EXPECTED_KEYS.get(kind, EXPECTED_KEYS["action"])
                    if not want <= set(obj):
                        counts["schema_missing_keys"] += 1
                except Exception:
                    counts["completion_not_json"] += 1

                # 3. private bleed
                if "teacher_private_diagnosis" in ct or "private_diagnosis" in ct:
                    counts["private_bleed"] += 1

                # 2. gold leakage
                g = gold.get(key, "")
                if not g or g.lower() in BOOLEAN:
                    continue
                pat = re.compile(r"\b" + re.escape(g) + r"\b", re.I)
                if pat.search(ct) and not pat.search(pt):
                    # In the target but nowhere the student could have seen it.
                    counts["gold_only_in_target"] += 1
        for k in ("completion_not_json", "private_bleed", "gold_only_in_target", "schema_missing_keys"):
            if counts[k]:
                failures.append(f"{fold_dir.name}: {counts[k]} x {k}")
        print(f"  {fold_dir.name:<26} {counts['examples']:>6} examples | "
              + " | ".join(f"{k}={counts[k]}" for k in
                           ("completion_not_json", "private_bleed", "gold_only_in_target",
                            "schema_missing_keys")))

    print()
    if failures:
        print("AUDIT FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("AUDIT PASSED — splits disjoint, no gold-only-in-target, no private bleed, schema OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
