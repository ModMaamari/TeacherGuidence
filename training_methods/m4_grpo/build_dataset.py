"""m4_grpo dataset builder: the RL question pool.

GRPO needs prompts (questions + their per-question retrieval scope), not
demonstrations. This script writes the pool = the 3000 train questions minus the
m1 dev questions (kept clean for evaluation), shuffled with a fixed seed.

Output: training_methods/m4_grpo/data/grpo_questions.jsonl + stats.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402
from training_methods.common.episode_lib import qid_split  # noqa: E402

QUESTIONS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--dev-fraction", type=float, default=0.03, help="must match m1's split")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("m4_build", out / "build.log")

    rows = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
    pool = [r for r in rows if qid_split(r["id"], args.dev_fraction) == "train"]
    random.Random(args.seed).shuffle(pool)
    with open(out / "grpo_questions.jsonl", "w") as w:
        for r in pool:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "stats.json", {
        "total_questions": len(rows), "pool": len(pool),
        "excluded_dev": len(rows) - len(pool), "seed": args.seed,
    })
    log.info(f"pool={len(pool)} (excluded {len(rows)-len(pool)} dev qids)")


if __name__ == "__main__":
    main()
