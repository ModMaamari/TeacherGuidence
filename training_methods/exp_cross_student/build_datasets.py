"""exp_cross_student dataset builder: does a student learn better from ITS OWN traces?

From the two MiniMax-M3 teacher-guidance runs on the same 3000 HotpotQA train questions --
one with Qwen3.5-0.8B as the student, one with Granite-4.1-3B -- build:

  * per source run, a guidance-as-internal-thought SFT dataset (same recipe as m1_sft:
    ``episode_lib.build_episode_examples``) from the episodes whose final answer was
    CORRECT (deterministic ``final_metrics.answer_correct``);
  * per source run, an end-to-end TEST set of 200 questions: 100 answered correctly by
    that student + 100 answered wrongly (both sampled with a fixed seed).

Contamination rule: both runs cover the SAME 3000 questions, and the same trained model
is later evaluated on BOTH test sets. So the union of ALL test qids (both runs) is
excluded from BOTH training datasets -- otherwise a "cross-student" eval question could
have been seen in training via the other run and the comparison would be biased.

Outputs under training_methods/exp_cross_student/data/:
    <src>/train.jsonl, dev.jsonl            SFT examples (prompt/completion messages)
    <src>/test_questions.jsonl              200 question rows for end2end eval
    <src>/test_qids.json                    {"correct": [...], "wrong": [...]}
    build_stats.json                        counts and filter telemetry for both sources

Usage:
    .venv/bin/python training_methods/exp_cross_student/build_datasets.py [--seed 13]
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

from training_methods.common.episode_lib import (  # noqa: E402
    build_episode_examples,
    iter_run_episodes,
    qid_split,
)

SOURCES = {
    "qwen": "data/simulation_output/tg_m3_qwen08b_b3_eden",
    "granite": "data/simulation_output/tg_m3_granite_b3_eden",
}
QUESTIONS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl"
N_TEST_CORRECT = 100
N_TEST_WRONG = 100


def answer_correct(ep: dict) -> bool:
    return bool((ep.get("final_metrics") or {}).get("answer_correct"))


def load_run(root: Path) -> dict:
    """qid -> episode for a consolidated run (3000 unique qids)."""
    eps = {}
    for _, ep in iter_run_episodes(root):
        eps[str(ep["qid"])] = ep
    return eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {src: load_run(REPO_ROOT / root) for src, root in SOURCES.items()}
    for src, eps in runs.items():
        assert len(eps) == 3000, f"{src}: expected 3000 episodes, got {len(eps)}"

    # ---- per-source test sets: 100 correct + 100 wrong (fixed seed) -------------
    rng = random.Random(args.seed)
    test_qids: dict[str, dict[str, list[str]]] = {}
    for src, eps in runs.items():
        correct = sorted(q for q, e in eps.items() if answer_correct(e))
        wrong = sorted(q for q, e in eps.items() if not answer_correct(e))
        test_qids[src] = {
            "correct": rng.sample(correct, N_TEST_CORRECT),
            "wrong": rng.sample(wrong, N_TEST_WRONG),
        }

    # union of all test qids is banned from BOTH training sets (see module docstring)
    banned: set[str] = set()
    for sel in test_qids.values():
        banned |= set(sel["correct"]) | set(sel["wrong"])

    questions = {str(json.loads(l)["id"]): l for l in QUESTIONS.read_text().splitlines() if l.strip()}
    stats: dict = {"seed": args.seed, "banned_test_qids_total": len(banned), "sources": {}}

    for src, eps in runs.items():
        src_out = out / src
        src_out.mkdir(parents=True, exist_ok=True)
        sel = test_qids[src]

        # -- test artifacts --------------------------------------------------------
        with open(src_out / "test_qids.json", "w") as w:
            json.dump(sel, w, indent=2)
        with open(src_out / "test_questions.jsonl", "w") as w:
            for q in sel["correct"] + sel["wrong"]:
                w.write(questions[q] + "\n")

        # -- training examples from correct, non-banned episodes --------------------
        counters = collections.Counter()
        train, dev = [], []
        for qid, ep in sorted(eps.items()):
            if not answer_correct(ep):
                counters["skipped_wrong"] += 1
                continue
            if qid in banned:
                counters["skipped_banned_test"] += 1
                continue
            exs = build_episode_examples(ep, run=src)
            n_steps = len(ep.get("steps") or [])
            counters["episodes"] += 1
            counters["examples"] += len(exs)
            counters["examples_gated_out"] += (n_steps + 1) - len(exs)
            for e in exs:
                e["metadata"]["source"] = src
                (dev if qid_split(qid, args.dev_fraction) == "dev" else train).append(e)

        for name, rows in (("train.jsonl", train), ("dev.jsonl", dev)):
            with open(src_out / name, "w") as w:
                for e in rows:
                    w.write(json.dumps(e, ensure_ascii=False) + "\n")

        kinds = collections.Counter(e["metadata"]["kind"] for e in train + dev)
        tools = collections.Counter(e["metadata"].get("tool") for e in train + dev
                                    if e["metadata"]["kind"] == "action")
        stats["sources"][src] = {
            "run_root": SOURCES[src],
            "episodes_correct_total": sum(1 for e in eps.values() if answer_correct(e)),
            "episodes_used": counters["episodes"],
            "skipped_banned_test": counters["skipped_banned_test"],
            "train_examples": len(train),
            "dev_examples": len(dev),
            "examples_gated_out": counters["examples_gated_out"],
            "kinds": dict(kinds),
            "tools": dict(tools),
            "test_correct": len(sel["correct"]),
            "test_wrong": len(sel["wrong"]),
        }
        print(f"[{src}] episodes={counters['episodes']} train={len(train)} dev={len(dev)} "
              f"(banned-test skipped={counters['skipped_banned_test']}, "
              f"gated out={counters['examples_gated_out']})")

    with open(out / "build_stats.json", "w") as w:
        json.dump(stats, w, indent=2)
    print(f"stats -> {out / 'build_stats.json'}")


if __name__ == "__main__":
    main()
