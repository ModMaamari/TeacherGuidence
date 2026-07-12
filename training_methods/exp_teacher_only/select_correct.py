"""exp_teacher_only PART B: select the correct expert episodes to train on.

A wrong expert trajectory is a bad demonstration, so we keep only episodes the 120b got
right. The primary signal is the SEMANTIC judge (``judge_final_answers.py`` -> one
gpt-oss-120b verdict per final answer), because it is the exact metric the four-arm eval
uses, so the training filter and the eval metric agree. We keep ``verdict.correct == 1``.
The free lexical ``cover_match`` yield is reported alongside for reference (and used as a
fallback for any episode the judge failed to score).

Reads the collector's ``traces.jsonl`` and the judge's ``verdicts.jsonl`` (matched by qid),
writes ``selected_correct.jsonl`` (full trace lines of kept episodes) + ``selection_stats.json``.

Usage:
    # 1) judge every final answer
    .venv_train/bin/python training_methods/common/judge_final_answers.py \
        --out training_methods/exp_teacher_only/data/judge \
        data/simulation_output/traces_oss120b_teacheronly_3000/traces.jsonl
    # 2) select the correct ones
    .venv_train/bin/python training_methods/exp_teacher_only/select_correct.py \
        --traces   data/simulation_output/traces_oss120b_teacheronly_3000/traces.jsonl \
        --verdicts training_methods/exp_teacher_only/data/judge/verdicts.jsonl \
        --out      training_methods/exp_teacher_only/data/selected_correct.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--verdicts", required=True, help="judge_final_answers verdicts.jsonl")
    ap.add_argument("--out", required=True, help="selected_correct.jsonl output path")
    args = ap.parse_args()

    verdict_by_qid = {}
    for line in open(args.verdicts):
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("verdict")
        if v is not None:
            verdict_by_qid[r["qid"]] = v

    stats = collections.Counter()
    kept = []
    for line in open(args.traces):
        if not line.strip():
            continue
        ep = json.loads(line)
        stats["episodes"] += 1
        cover = bool((ep.get("final_metrics") or {}).get("cover_match"))
        em = bool((ep.get("final_metrics") or {}).get("exact_match"))
        stats["cover_correct"] += int(cover)
        stats["em_correct"] += int(em)
        v = verdict_by_qid.get(ep["qid"])
        if v is not None:
            stats["judged"] += 1
            judge_ok = v.get("correct") == 1
            stats["judge_correct"] += int(judge_ok)
            keep = judge_ok
        else:
            # judge failed to score this one: fall back to the lexical cover signal.
            stats["judge_missing"] += 1
            keep = cover
        if keep:
            kept.append(ep)

    with open(args.out, "w") as w:
        for ep in kept:
            w.write(json.dumps(ep, ensure_ascii=False) + "\n")

    n = max(stats["episodes"], 1)
    summary = {
        "episodes": int(stats["episodes"]),
        "judged": int(stats["judged"]),
        "judge_missing": int(stats["judge_missing"]),
        "kept": len(kept),
        "kept_rate": round(len(kept) / n, 4),
        "judge_correct": int(stats["judge_correct"]),
        "judge_correct_rate": round(stats["judge_correct"] / max(stats["judged"], 1), 4),
        "cover_correct": int(stats["cover_correct"]),
        "cover_correct_rate": round(stats["cover_correct"] / n, 4),
        "em_correct": int(stats["em_correct"]),
        "em_correct_rate": round(stats["em_correct"] / n, 4),
    }
    write_json(Path(args.out).parent / "selection_stats.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
