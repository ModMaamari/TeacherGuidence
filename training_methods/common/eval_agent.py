"""Teacherless agent evaluation for any trained checkpoint (base or base+LoRA).

Runs the HF agent loop over a question file and writes timestamped artifacts:

    <out>/<ts>_<tag>/episodes.jsonl    full episodes (prompts, actions, observations)
    <out>/<ts>_<tag>/metrics.json      aggregate EM / F1 / cover-match / doc-recall
    <out>/<ts>_<tag>/eval.log          timestamped log

Used for BOTH standard evals (e.g. a slice of the 3000 train questions held out as
dev) and the golden-100 hardest-questions test.

Examples:
    # base model on golden-100
    .venv_train/bin/python training_methods/common/eval_agent.py \
        --questions training_methods/common/data/golden100/golden100_questions.jsonl \
        --corpus    training_methods/common/data/golden100/golden100_corpus.jsonl \
        --out training_methods/m1_sft/runs/eval_golden100 --tag base --budget 4

    # trained adapter
    ... --adapter training_methods/m1_sft/runs/<ts>_train/adapter --tag m1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402
from training_methods.common.hf_agent_loop import (  # noqa: E402
    PolicyModel,
    load_questions,
    run_episode,
    run_episodes_batched,
)
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="base dir; a timestamped subdir is created")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--hidden-budget", action="store_true")
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default=None, help="i/n to run only shard i of n (0-based)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help=">1 runs episodes in lockstep with batched generation "
                         "(~batch-size-fold fewer forward passes; same semantics)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch/random/numpy (meaningful with --temperature > 0)")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="hf = in-process transformers; vllm = OpenAI-compatible "
                         "server (continuous batching; start via serve_vllm.sh)")
    ap.add_argument("--server-url", default="http://127.0.0.1:8300")
    ap.add_argument("--served-model", default="student",
                    help="served model name on the vLLM server (a LoRA module "
                         "name to evaluate an adapter)")
    args = ap.parse_args()

    run_dir = timestamped_dir(args.out, args.tag)
    log = setup_logger("eval_agent", run_dir / "eval.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")

    questions = load_questions(args.questions, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        questions = [q for j, q in enumerate(questions) if j % n == i]
        log.info(f"shard {i}/{n}: {len(questions)} questions")
    retriever = HotpotLocalRetriever(args.corpus)
    log.info(f"loaded {len(questions)} questions | corpus={args.corpus}")

    if args.seed is not None:
        import random as _random

        import numpy as _np
        import torch as _torch
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        _torch.manual_seed(args.seed)
        log.info(f"seeded everything with {args.seed}")

    if args.backend == "vllm":
        from training_methods.common.vllm_backend import VllmPolicy, wait_ready

        served = wait_ready(args.server_url, args.served_model, timeout_s=120)
        policy = VllmPolicy(args.server_url, args.served_model, seed=args.seed,
                            max_parallel=max(args.batch_size, 4))
        log.info(f"vllm policy: {args.server_url} model={args.served_model} (served: {served})")
    else:
        policy = PolicyModel(args.model, args.adapter, device=args.device)
        log.info(f"policy loaded: {args.model} adapter={args.adapter}")
    t_run0 = __import__("time").time()

    episodes = []
    with open(run_dir / "episodes.jsonl", "w") as w:

        def _record(ep):
            episodes.append(ep)
            w.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
            w.flush()
            m = ep["final_metrics"]
            log.info(
                f"[{len(episodes)}/{len(questions)}] qid={ep['qid']} em={m['exact_match']} "
                f"f1={m['f1']} cover={m['cover_match']} steps={ep['used_steps']} "
                f"stop={ep['stop_reason']} ans={ep['final_answer'][:60]!r}"
            )

        if args.batch_size > 1:
            run_episodes_batched(
                policy, questions, retriever,
                budget=args.budget,
                disclose_budget=not args.hidden_budget,
                with_plan=not args.no_plan,
                temperature=args.temperature,
                batch_size=args.batch_size,
                on_episode=_record,
                logger=log,
            )
        else:
            for q in questions:
                _record(run_episode(
                    policy, q, retriever,
                    budget=args.budget,
                    disclose_budget=not args.hidden_budget,
                    with_plan=not args.no_plan,
                    temperature=args.temperature,
                    logger=log,
                ))

    n = len(episodes)
    doc_recalls = [e["final_metrics"]["doc_recall"] for e in episodes
                   if e["final_metrics"]["doc_recall"] is not None]

    import torch as _torch
    gpu_peak_gb = round(_torch.cuda.max_memory_allocated() / 1e9, 3) if _torch.cuda.is_available() else None
    gstats = [s.get("gen_stats") for e in episodes for s in e["steps"]]
    gstats += [(e.get("plan") or {}).get("gen_stats") for e in episodes]
    gstats = [g for g in gstats if g]
    tok_prompt = sum(g["prompt_tokens"] for g in gstats)
    tok_out = sum(g["completion_tokens"] for g in gstats)
    gen_time = round(sum(g["gen_s"] for g in gstats), 1)
    agg = {
        "model": args.model,
        "adapter": args.adapter,
        "seed": args.seed,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "wall_time_s": round(__import__("time").time() - t_run0, 1),
        "gpu_peak_mem_gb": gpu_peak_gb,
        "student_prompt_tokens": tok_prompt,
        "student_completion_tokens": tok_out,
        "student_gen_time_s": gen_time,
        "tokens_per_episode": round((tok_prompt + tok_out) / max(n, 1), 1),
        "n": n,
        "budget": args.budget,
        "em": round(sum(e["final_metrics"]["exact_match"] for e in episodes) / n, 4),
        "f1": round(sum(e["final_metrics"]["f1"] for e in episodes) / n, 4),
        "cover_match": round(sum(e["final_metrics"]["cover_match"] for e in episodes) / n, 4),
        "doc_recall": round(statistics.mean(doc_recalls), 4) if doc_recalls else None,
        "mean_steps": round(statistics.mean(e["used_steps"] for e in episodes), 2),
        "stop_reasons": {r: sum(1 for e in episodes if e["stop_reason"] == r)
                         for r in {e["stop_reason"] for e in episodes}},
        "invalid_action_steps": sum(
            1 for e in episodes for s in e["steps"] if not s["action_valid"]),
        "total_steps": sum(e["used_steps"] for e in episodes),
    }
    write_json(run_dir / "metrics.json", agg)
    log.info(f"AGGREGATE: {json.dumps(agg)}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
