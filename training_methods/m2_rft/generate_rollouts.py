"""m2_rft rollout generator: sample teacherless episodes with the CURRENT policy.

For each fresh question, samples --n-per-question episodes at --temperature through
the shared HF agent loop. Episodes stream to a timestamped artifacts dir as they
finish (append-only jsonl, safe to inspect mid-run).

Shard across the 4 GPUs by running 4 processes with --shard 0/4 .. 3/4 and
CUDA_VISIBLE_DEVICES set per process (see GUIDE.md).

Output: <out>/<ts>_<tag>/rollouts.jsonl + rollout_stats.json + rollouts.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402
from training_methods.common.hf_agent_loop import PolicyModel, load_questions, run_episode  # noqa: E402
from training_methods.m4_grpo.rewards import episode_reward  # noqa: E402
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--questions", default=str(Path(__file__).parent / "data" / "fresh" / "fresh_questions.jsonl"))
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "data" / "fresh" / "fresh_corpus.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "runs" / "rollouts"))
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--n-per-question", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    shard_sfx = f"_s{args.shard.replace('/', 'of')}" if args.shard else ""
    run_dir = timestamped_dir(args.out, args.tag + shard_sfx)
    log = setup_logger("m2_rollouts", run_dir / "rollouts.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")

    questions = load_questions(args.questions, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        questions = [q for j, q in enumerate(questions) if j % n == i]
        log.info(f"shard {i}/{n}: {len(questions)} questions")
    retriever = HotpotLocalRetriever(args.corpus)
    policy = PolicyModel(args.model, args.adapter, device=args.device)
    log.info(f"policy: {args.model} adapter={args.adapter}")

    n_correct = 0
    total = 0
    with open(run_dir / "rollouts.jsonl", "w") as w:
        for k, q in enumerate(questions, 1):
            for s in range(args.n_per_question):
                ep = run_episode(policy, q, retriever, budget=args.budget,
                                 temperature=args.temperature)
                ep["reward"] = episode_reward(ep)["reward"]
                ep["sample_idx"] = s
                total += 1
                n_correct += bool(ep["final_metrics"]["cover_match"])
                w.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
                w.flush()
            if k % 10 == 0 or k == len(questions):
                log.info(f"[{k}/{len(questions)}] episodes={total} cover-correct={n_correct} "
                         f"({n_correct/max(total,1):.1%})")

    write_json(run_dir / "rollout_stats.json", {
        "questions": len(questions), "episodes": total,
        "cover_correct": n_correct, "cover_rate": round(n_correct / max(total, 1), 4),
        "n_per_question": args.n_per_question, "temperature": args.temperature,
        "budget": args.budget, "adapter": args.adapter,
    })
    log.info(f"DONE: {total} episodes, {n_correct} cover-correct")


if __name__ == "__main__":
    main()
