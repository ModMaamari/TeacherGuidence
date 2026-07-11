"""m1_sft dataset builder: guidance-as-internal-thought SFT examples.

Sources (all teacher-correct, one trajectory per (qid, run)):
  1. ds1 horizontal best  -- per-question best episode across the six budget runs
  2. ds4 hard-q answers   -- the 118 hard questions recovered by the retry campaign
  3. teacher_accept episodes from b5h/b9h -- natural-stop behaviour (ds1 skews to
     cheap forced-finish runs; without these the model never learns to CHOOSE finish)

Each episode becomes (plan example) + (one example per step); the step's incoming
teacher feedback (previous step's, plan feedback for step 1) is kept VERBATIM in
second person as a ``teacher_guidance`` object generated FIRST in the target JSON.
``[answer hidden]`` placeholders are restored echo-safe or dropped; a leakage gate
rejects unsafe examples (see common/episode_lib.py).

Outputs (in --out, default training_methods/m1_sft/data):
    train.jsonl / dev.jsonl   {"prompt": [messages], "completion": [message], "metadata": ...}
    dev_questions.jsonl       question rows of dev qids (for agent-loop eval)
    stats.json                counts, filters, restoration/gate telemetry
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
from training_methods.common.episode_lib import (  # noqa: E402
    RUN_ROOTS,
    build_episode_examples,
    iter_run_episodes,
    load_jsonl,
    qid_split,
    teacher_correct,
)

DS1 = REPO_ROOT / "data/datasets/best_answer_v1/ds1_horizontal_best.jsonl"
DS4 = REPO_ROOT / "data/datasets/best_answer_v1/ds4_hard_q_answers_118.jsonl"
QUESTIONS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--accept-runs", default="b5h,b9h",
                    help="runs whose teacher_accept episodes are added for stop behaviour")
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--no-accepts", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("m1_build", out / "build.log")

    stats = collections.Counter()
    per_source = collections.Counter()
    seen: set = set()  # (qid, run) pairs already taken
    examples = []

    def add_episode(ep, run, source):
        key = (ep.get("qid"), run)
        if key in seen:
            stats["dup_qid_run_skipped"] += 1
            return
        seen.add(key)
        exs = build_episode_examples(ep, run)
        n_steps = len(ep.get("steps") or [])
        stats["episodes"] += 1
        stats["examples"] += len(exs)
        # plan + usable steps vs raw steps tells us how many were gated out
        stats["steps_total"] += n_steps
        stats["examples_gated_out"] += (n_steps + 1) - len(exs)
        for e in exs:
            e["metadata"]["source"] = source
            md = e["metadata"]
            stats["restored_placeholders"] += md.get("restored", 0) or 0
            stats["dropped_sentences"] += md.get("dropped_sentences", 0) or 0
            per_source[source] += 1
            examples.append(e)

    log.info("loading ds1 horizontal best ...")
    for rec in load_jsonl(DS1):
        add_episode(rec, rec.get("bas_run", "ds1"), "ds1")
    log.info(f"ds1 done: {stats['episodes']} episodes, {stats['examples']} examples")

    log.info("loading ds4 hard-question answers ...")
    for rec in load_jsonl(DS4):
        add_episode(rec, rec.get("bas_run", "ds4"), "ds4")
    log.info(f"+ds4 done: {stats['episodes']} episodes, {stats['examples']} examples")

    if not args.no_accepts:
        for run in [r for r in args.accept_runs.split(",") if r]:
            root = REPO_ROOT / RUN_ROOTS[run]
            n0 = stats["episodes"]
            for _, ep in iter_run_episodes(root):
                if ep.get("stop_reason") == "teacher_accept" and teacher_correct(ep):
                    add_episode(ep, run, f"accept_{run}")
            log.info(f"+accepts {run}: {stats['episodes'] - n0} episodes")

    # train/dev split by qid (never split a question across sides)
    train, dev = [], []
    for e in examples:
        (dev if qid_split(e["metadata"]["qid"], args.dev_fraction) == "dev" else train).append(e)

    with open(out / "train.jsonl", "w") as w:
        for e in train:
            w.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "dev.jsonl", "w") as w:
        for e in dev:
            w.write(json.dumps(e, ensure_ascii=False) + "\n")

    # dev question rows for agent-loop evaluation
    dev_qids = {e["metadata"]["qid"] for e in dev}
    n_devq = 0
    with open(out / "dev_questions.jsonl", "w") as w, open(QUESTIONS) as fh:
        for line in fh:
            if json.loads(line)["id"] in dev_qids:
                w.write(line)
                n_devq += 1

    tools = collections.Counter(e["metadata"].get("tool") for e in examples
                                if e["metadata"]["kind"] == "action")
    kinds = collections.Counter(e["metadata"]["kind"] for e in examples)
    guided = sum(1 for e in examples if e["metadata"].get("had_guidance"))
    summary = {
        "train_examples": len(train),
        "dev_examples": len(dev),
        "dev_questions": n_devq,
        "unique_qids": len({e["metadata"]["qid"] for e in examples}),
        "episodes": stats["episodes"],
        "examples_gated_out": stats["examples_gated_out"],
        "restored_placeholders": stats["restored_placeholders"],
        "dropped_placeholder_sentences": stats["dropped_sentences"],
        "dup_qid_run_skipped": stats["dup_qid_run_skipped"],
        "with_teacher_guidance_block": guided,
        "kinds": dict(kinds),
        "tools": dict(tools),
        "per_source": dict(per_source),
    }
    write_json(out / "stats.json", summary)
    log.info(f"DONE: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
