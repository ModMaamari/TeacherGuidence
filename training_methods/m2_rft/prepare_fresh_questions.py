"""m2_rft: prepare FRESH HotpotQA train questions for rejection-sampling rollouts.

Loads the distractor train split (90k, cached in the HF cache), removes every
question id already used by the 3000-question trace-collection set, shuffles with
a fixed seed and materializes the first --limit questions + their per-question
corpus in the same schema as the existing datasets.

Output: <out>/{fresh_questions.jsonl, fresh_corpus.jsonl, stats.json}
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
from agentsim.teacher_guidance.hotpot_converter import convert_examples  # noqa: E402

USED = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "fresh"))
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("m2_prepare", out / "prepare.log")

    used_ids = {json.loads(l)["id"] for l in open(USED) if l.strip()}
    log.info(f"{len(used_ids)} already-used question ids excluded")

    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", trust_remote_code=True)
    log.info(f"hotpot train split loaded: {len(ds)} examples")

    fresh = [ex for ex in ds if ex["id"] not in used_ids]
    random.Random(args.seed).shuffle(fresh)
    fresh = fresh[: args.limit]
    log.info(f"selected {len(fresh)} fresh questions (seed={args.seed})")

    q_rows, c_rows = convert_examples(fresh, split="train")
    with open(out / "fresh_questions.jsonl", "w") as w:
        for r in q_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "fresh_corpus.jsonl", "w") as w:
        for r in c_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "stats.json", {
        "n_questions": len(q_rows), "n_corpus_docs": len(c_rows),
        "excluded_used": len(used_ids), "seed": args.seed,
    })
    log.info(f"wrote {len(q_rows)} questions + {len(c_rows)} docs to {out}")


if __name__ == "__main__":
    main()
