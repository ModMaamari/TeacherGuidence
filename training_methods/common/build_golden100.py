"""Build the golden-100 hardest-question eval set.

The six budget runs (b1..b9) left 265 train questions with NO correct answer in
any run; the hard-question retry campaign (hard_q_xr) recovered 118 of them.
The remaining 147 were never answered by the teacher-guided student under any
setting -- the hardest questions we know of. This script samples 100 of them
(fixed seed) and materializes:

    <out>/golden100_questions.jsonl   question rows (same schema as train3000)
    <out>/golden100_corpus.jsonl      their per-question document scope
    <out>/summary.json                provenance + level/type breakdown

Usage:
    python training_methods/common/build_golden100.py \
        [--out training_methods/common/data/golden100] [--n 100] [--seed 13]
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

HARD_IDS = REPO_ROOT / "data/simulation_output/hard_q_xr/hard_question_ids.txt"
ANSWERED = REPO_ROOT / "data/datasets/best_answer_v1/ds4_hard_q_answers_118.jsonl"
QUESTIONS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl"
CORPUS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "golden100"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("golden100", out / "build.log")

    hard = [l.strip() for l in open(HARD_IDS) if l.strip()]
    answered = {json.loads(l)["qid"] for l in open(ANSWERED) if l.strip()}
    never = [q for q in hard if q not in answered]
    log.info(f"hard={len(hard)} answered={len(answered)} never-answered={len(never)}")
    assert len(never) == len(hard) - len(answered), "answered ids must be a subset of hard ids"

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(never, min(args.n, len(never))))
    chosen_set = set(chosen)
    log.info(f"sampled {len(chosen)} golden questions (seed={args.seed})")

    rows = []
    with open(QUESTIONS) as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in chosen_set:
                rows.append(r)
    assert len(rows) == len(chosen), f"found {len(rows)} of {len(chosen)} question rows"

    n_docs = 0
    with open(out / "golden100_corpus.jsonl", "w") as w, open(CORPUS) as fh:
        for line in fh:
            if json.loads(line)["qid"] in chosen_set:
                w.write(line)
                n_docs += 1
    with open(out / "golden100_questions.jsonl", "w") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    levels = collections.Counter(r.get("level") for r in rows)
    types = collections.Counter(r.get("type") for r in rows)
    write_json(out / "summary.json", {
        "n_questions": len(rows),
        "n_corpus_docs": n_docs,
        "seed": args.seed,
        "provenance": "265 never-correct across b1..b9 minus 118 recovered by hard_q_xr",
        "levels": dict(levels),
        "types": dict(types),
        "question_ids": chosen,
    })
    log.info(f"wrote {len(rows)} questions + {n_docs} docs to {out}")
    log.info(f"levels={dict(levels)} types={dict(types)}")


if __name__ == "__main__":
    main()
