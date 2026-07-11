"""m2_rft filter: keep verifiably-correct rollouts and build the next SFT round.

Acceptance rule (STaR-style rejection sampling, gold answers are available for the
train split): cover_match AND f1 >= --min-f1 (default 0.5). Among accepted samples
of the same question, only the highest-reward episode is kept (dedup). Accepted
episodes become SFT examples in the m1 format -- plan turn + one example per step,
using the model's OWN raw outputs as targets (they are already teacher-free; there
is no teacher_guidance block in round >= 1 data, which is fine: the m1 mixture
already taught that behaviour, rounds reinforce successful search strategies).

Optionally merges with an existing training file (--merge-with, e.g. the m1 train
set) to produce the round's full training mixture.

Output: <out>/{rft_round.jsonl, merged_train.jsonl?, filter_stats.json}
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402
from agentsim.teacher_guidance.sft_export import DEFAULT_SYSTEM  # noqa: E402
from agentsim.teacher_guidance.json_utils import parse_student_action, parse_student_plan  # noqa: E402


def episode_to_examples(ep):
    """SFT rows from a rollout episode; only structurally valid turns are kept."""
    rows = []
    plan = ep.get("plan")
    if plan and plan.get("valid") and plan.get("raw"):
        obj, info = parse_student_plan(plan["raw"])
        if info.get("json_valid") and isinstance(obj, dict) and obj.get("steps"):
            rows.append({
                "prompt": [{"role": "system", "content": DEFAULT_SYSTEM},
                           {"role": "user", "content": plan["prompt"]}],
                "completion": [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}],
                "metadata": {"qid": ep["qid"], "kind": "plan", "step": 0, "source": "rft"},
            })
    for s in ep["steps"]:
        if not s.get("action_valid"):
            continue
        action, info = parse_student_action(s["student_raw"])
        if not info.get("action_valid"):
            continue
        rows.append({
            "prompt": [{"role": "system", "content": DEFAULT_SYSTEM},
                       {"role": "user", "content": s["student_prompt"]}],
            "completion": [{"role": "assistant",
                            "content": json.dumps(action.to_dict(), ensure_ascii=False)}],
            "metadata": {"qid": ep["qid"], "kind": "action", "step": s["t"],
                         "tool": action.action.tool, "source": "rft"},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", nargs="+", required=True,
                    help="one or more rollouts.jsonl files (shards are concatenated)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "round1"))
    ap.add_argument("--min-f1", type=float, default=0.5)
    ap.add_argument("--merge-with", default=None,
                    help="existing train.jsonl to merge (e.g. training_methods/m1_sft/data/train.jsonl)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("m2_filter", out / "filter.log")

    best = {}
    stats = collections.Counter()
    for path in args.rollouts:
        for line in open(path):
            if not line.strip():
                continue
            ep = json.loads(line)
            stats["episodes"] += 1
            m = ep["final_metrics"]
            if not (m["cover_match"] and m["f1"] >= args.min_f1):
                stats["rejected"] += 1
                continue
            stats["accepted"] += 1
            cur = best.get(ep["qid"])
            if cur is None or ep.get("reward", 0) > cur.get("reward", 0):
                best[ep["qid"]] = ep

    rows = []
    for ep in best.values():
        rows.extend(episode_to_examples(ep))
    with open(out / "rft_round.jsonl", "w") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"episodes={stats['episodes']} accepted={stats['accepted']} "
             f"unique-questions kept={len(best)} -> {len(rows)} SFT examples")

    merged_n = None
    if args.merge_with:
        merged = [l for l in open(args.merge_with) if l.strip()]
        with open(out / "merged_train.jsonl", "w") as w:
            for l in merged:
                w.write(l)
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        merged_n = len(merged) + len(rows)
        log.info(f"merged with {args.merge_with}: {merged_n} total examples")

    write_json(out / "filter_stats.json", {
        **{k: int(v) for k, v in stats.items()},
        "unique_questions_kept": len(best),
        "sft_examples": len(rows),
        "min_f1": args.min_f1,
        "merged_total": merged_n,
    })


if __name__ == "__main__":
    main()
