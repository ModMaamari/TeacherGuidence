"""Consolidate a tg_v1 collection into one deduplicated episode file + an index.

Downstream steps (dataset building, fold splitting, analysis) each need every episode,
and re-walking ~8,000 files across 3.8 GB takes minutes every time. This materialises the
corpus once:

    episodes.jsonl   one line per (dataset, qid) -- the best episode for that question
    index.jsonl      one small row per episode (dataset, qid, correctness, steps, teacher)
    stats.json       per-dataset totals

"Best" means a successful episode always beats one that ended in ``error``: a retry pass
re-collects questions the main run lost, so the same qid legitimately appears twice.

Usage::

    python scripts/consolidate_tgv1_episodes.py \
        --run-root data/simulation_output/tgv1 data/simulation_output/tgv1_retry \
        --out data/datasets/tg_v1_episodes
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    best: dict[tuple, dict] = {}
    files = 0
    for root in args.run_root:
        for p in Path(root).rglob("teacher_guidance_episodes.jsonl"):
            files += 1
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    ep = json.loads(line)
                    key = (ep.get("dataset"), ep.get("qid"))
                    cur = best.get(key)
                    if cur is None or (cur.get("stop_reason") == "error"
                                       and ep.get("stop_reason") != "error"):
                        best[key] = ep

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.defaultdict(collections.Counter)
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as ef, \
         (out / "index.jsonl").open("w", encoding="utf-8") as xf:
        for (ds, qid), ep in sorted(best.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            ef.write(json.dumps(ep, ensure_ascii=False) + "\n")
            m = ep.get("final_metrics") or {}
            row = {
                "dataset": ds, "qid": qid,
                "answer_correct": bool(m.get("answer_correct")),
                "teacher_answer_score": m.get("teacher_answer_score"),
                "exact_match": bool(m.get("exact_match")),
                "f1": m.get("f1"),
                "grounded": bool(m.get("answer_grounded")),
                "steps": len(ep.get("steps") or []),
                "stop_reason": ep.get("stop_reason"),
                "teacher": (ep.get("teacher_models_used") or [None])[0],
                "num_hops": ep.get("num_hops"),
            }
            xf.write(json.dumps(row) + "\n")
            s = stats[ds]
            s["episodes"] += 1
            s["correct"] += row["answer_correct"]
            s["grounded"] += row["grounded"]
            s["steps"] += row["steps"]
            s["errors"] += ep.get("stop_reason") == "error"

    summary = {ds: dict(c) for ds, c in stats.items()}
    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"read {files} files -> {len(best)} unique episodes")
    for ds, c in sorted(summary.items()):
        print(f"  {ds:<18} {c['episodes']:>5} episodes | {c['correct']:>5} correct "
              f"({100*c['correct']/max(c['episodes'],1):.1f}%) | {c['steps']} steps | {c['errors']} errors")
    print(f"wrote {out}/episodes.jsonl, index.jsonl, stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
