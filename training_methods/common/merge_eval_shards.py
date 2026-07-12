"""Merge sharded eval runs (eval_agent.py / teacher_eval_agent.py --shard i/n)
into one directory with a combined episodes.jsonl + metrics.json, so downstream
tooling (compare_evals.py, analysis) sees a single run.

Usage:
    python training_methods/common/merge_eval_shards.py \
        --out <merged-dir> <shard-run-dir> [<shard-run-dir> ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+", help="shard run dirs containing episodes.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = []
    shard_meta = []
    for d in args.shards:
        d = Path(d)
        with open(d / "episodes.jsonl") as fh:
            episodes += [json.loads(line) for line in fh if line.strip()]
        mfile = d / "metrics.json"
        if mfile.exists():
            shard_meta.append({"dir": str(d), **json.loads(mfile.read_text())})

    # de-dup by qid (last write wins) in case of overlapping reruns
    by_qid = {e["qid"]: e for e in episodes}
    episodes = list(by_qid.values())
    n = len(episodes)
    if not n:
        sys.exit("no episodes found in the given shard dirs")

    with open(out / "episodes.jsonl", "w") as w:
        for e in episodes:
            w.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")

    doc_recalls = [e["final_metrics"]["doc_recall"] for e in episodes
                   if e["final_metrics"].get("doc_recall") is not None]
    tj = [e for e in episodes if e.get("teacher_final_judgment")]
    agg = {
        "merged_from": [str(s) for s in args.shards],
        "n": n,
        "em": round(sum(e["final_metrics"]["exact_match"] for e in episodes) / n, 4),
        "f1": round(sum(e["final_metrics"]["f1"] for e in episodes) / n, 4),
        "cover_match": round(sum(e["final_metrics"]["cover_match"] for e in episodes) / n, 4),
        "doc_recall": round(statistics.mean(doc_recalls), 4) if doc_recalls else None,
        "mean_steps": round(statistics.mean(e["used_steps"] for e in episodes), 2),
        "stop_reasons": {r: sum(1 for e in episodes if e["stop_reason"] == r)
                         for r in {e["stop_reason"] for e in episodes}},
        "total_steps": sum(e["used_steps"] for e in episodes),
    }
    if tj:
        agg["teacher_final_judged"] = len(tj)
        agg["teacher_correct_final"] = sum(
            1 for e in tj if (e["teacher_final_judgment"] or {}).get("correct") == 1)
    # carry model/adapter/budget from the first shard's metrics if present
    if shard_meta:
        for k in ("arm", "model", "adapter", "budget", "teacher_router"):
            if k in shard_meta[0]:
                agg.setdefault(k, shard_meta[0][k])
    write_json(out / "metrics.json", agg)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
