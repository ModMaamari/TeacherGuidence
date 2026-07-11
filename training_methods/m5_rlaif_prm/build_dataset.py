"""m5_rlaif_prm dataset builder: distill the teacher's step judgments into a PRM.

Every teacher-scored step across all six runs becomes a judge example:

    prompt     = judge instruction + the STUDENT-VISIBLE state (guidance stripped)
                 + the action the student took
    completion = a single digit 0-9  (round(teacher_score * 9))

The PRM sees ONLY what the student saw (never gold/hidden data), so it remains
usable at inference time as a dense reward / self-stop signal. granite has no
sequence-classification head, so the PRM is generative: at scoring time we read
the probability distribution over the ten digit tokens at the first generated
position and take the expected value -> a continuous score in [0, 1].

Outputs: prm_train.jsonl / prm_dev.jsonl / stats.json  (in --out)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402
from training_methods.common.episode_lib import (  # noqa: E402
    RUN_ROOTS,
    iter_run_episodes,
    qid_split,
)
from agentsim.teacher_guidance.sft_internalize import strip_teacher_guidance_block  # noqa: E402

JUDGE_SYSTEM = (
    "You are a process reward judge for a retrieval agent. Given the agent's current "
    "state and the action it just proposed, rate how helpful that action is for "
    "answering the question correctly and efficiently. Reply with ONE digit from 0 "
    "(harmful/wasted step) to 9 (excellent step)."
)


def judge_prompt(state_prompt: str, action: dict) -> list:
    user = (state_prompt.strip()
            + "\n\n=== PROPOSED ACTION ===\n"
            + json.dumps(action, ensure_ascii=False)
            + "\n\nRate this action 0-9. Reply with one digit only.")
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--max-examples", type=int, default=60000)
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("m5_build", out / "build.log")

    rows = []
    label_hist = collections.Counter()
    for run, root in RUN_ROOTS.items():
        n0 = len(rows)
        for _, ep in iter_run_episodes(REPO_ROOT / root):
            qid = ep.get("qid")
            for s in ep.get("steps") or []:
                g = s.get("student_visible_guidance") or {}
                score = g.get("score")
                action = s.get("student_action") or {}
                if not isinstance(score, (int, float)) or not (action.get("action") or {}).get("tool"):
                    continue
                digit = max(0, min(9, round(float(score) * 9)))
                label_hist[digit] += 1
                rows.append({
                    "prompt": judge_prompt(strip_teacher_guidance_block(s.get("student_prompt") or ""), action),
                    "completion": [{"role": "assistant", "content": str(digit)}],
                    "metadata": {"qid": qid, "run": run, "step": s.get("t"),
                                 "teacher_score": float(score), "digit": digit},
                })
        log.info(f"scanned {run}: total examples so far {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.max_examples]

    train, dev = [], []
    for r in rows:
        (dev if qid_split(r["metadata"]["qid"], args.dev_fraction) == "dev" else train).append(r)
    for name, data in (("prm_train", train), ("prm_dev", dev)):
        with open(out / f"{name}.jsonl", "w") as w:
            for r in data:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_json(out / "stats.json", {
        "train": len(train), "dev": len(dev),
        "label_histogram": {str(k): v for k, v in sorted(label_hist.items())},
        "max_examples": args.max_examples, "seed": args.seed,
    })
    log.info(f"DONE: train={len(train)} dev={len(dev)} labels={dict(sorted(label_hist.items()))}")


if __name__ == "__main__":
    main()
