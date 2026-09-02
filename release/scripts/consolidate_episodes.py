#!/usr/bin/env python
"""Consolidate teacher-guidance collection runs into one publishable episode file.

Input: one or more run directories written by ``scripts/collect_episodes.py`` (any
depth; every ``teacher_guidance_episodes.jsonl`` underneath is read). Output::

    <out>/episodes.jsonl[.gz]   one line per (dataset, qid): the best episode for that
                                question, in the publishable view (see tgd/episodes.py)
    <out>/index.jsonl           one small row per episode (dataset, qid, correct, steps, ...)
    <out>/stats.json            per-dataset totals

"Best" = a successful episode always beats one that ended in ``error``; among equals the
later file wins, so a retry run placed after the main run supersedes it.

Usage::

    python scripts/consolidate_episodes.py --runs runs/collect/hotpotqa runs/collect/musique \
        --out data/episodes --gzip
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401  (sys.path)
from tgd.episodes import is_error, iter_run_episodes, normalize_models, publishable, summary_row
from tgd.io import open_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gzip", action="store_true", help="write episodes.jsonl.gz instead of episodes.jsonl")
    ap.add_argument("--strip-privileged", action="store_true",
                    help="also drop the gold-bearing teacher prompt/raw fields (smaller, train-safe)")
    ap.add_argument("--rename-model", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite a model id everywhere (e.g. strip a provider prefix: vllm/student=org/model)")
    args = ap.parse_args()

    renames = dict(r.split("=", 1) for r in args.rename_model)
    best = {}
    n_read = 0
    for f, ep in iter_run_episodes(args.runs):
        n_read += 1
        key = (ep.get("dataset"), str(ep.get("qid")))
        prev = best.get(key)
        if prev is None or (is_error(prev) and not is_error(ep)) or (is_error(prev) == is_error(ep)):
            best[key] = ep
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ep_path = out / ("episodes.jsonl.gz" if args.gzip else "episodes.jsonl")
    stats = collections.defaultdict(collections.Counter)
    with open_text(ep_path, "wt") as fe, open(out / "index.jsonl", "w", encoding="utf-8") as fi:
        for key in sorted(best, key=lambda k: (str(k[0]), str(k[1]))):
            ep = normalize_models(publishable(best[key], strip_privileged=args.strip_privileged), renames)
            fe.write(json.dumps(ep, ensure_ascii=False) + "\n")
            row = summary_row(ep)
            fi.write(json.dumps(row, ensure_ascii=False) + "\n")
            s = stats[row["dataset"]]
            s["episodes"] += 1
            s["correct"] += int(row["correct"])
            s["grounded"] += int(row["grounded"])
            s["errors"] += int(is_error(ep))
            s["steps"] += int(row["steps"] or 0)
    (out / "stats.json").write_text(json.dumps({k: dict(v) for k, v in stats.items()}, indent=2))
    print(f"read {n_read} episodes -> {len(best)} unique (dataset, qid) -> {ep_path}")
    for ds, s in sorted(stats.items()):
        print(f"  {ds:<18} episodes {s['episodes']:>5}  correct {s['correct']:>5}  errors {s['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
