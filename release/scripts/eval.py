#!/usr/bin/env python
"""Evaluate one arm on one question file. Resumable, monitorable, one command per arm.

Arms (identical questions, corpus, tools, budget and metrics for all four)::

    student   the student acts alone (base model, or base + LoRA adapter)      [GPU]
    guided    the student acts, a teacher reviews the plan and every step       [GPU + API]
    teacher   the teacher model is the agent, nobody guides it                  [API only]

The student is served either by a vLLM server (``--student vllm``, recommended; start it
with scripts/serve_vllm.sh, evaluate an adapter by its served name) or in-process by
transformers (``--student hf --model <id> [--adapter <dir>]``). Teacher / agent models
are provider-prefixed ids (see docs/PROVIDERS.md), comma-separated for a fallback chain.

Output (``--out`` is a fixed directory)::

    <out>/episodes.jsonl   one line per finished question, appended as it lands
    <out>/status.json      progress: done / total / rate / eta, rewritten every episode
    <out>/metrics.json     aggregate (recomputed from episodes.jsonl at the end)
    <out>/eval.log         UTC-timestamped log
    <out>/.done            written when every question has an episode

Re-running the same command resumes: questions already in episodes.jsonl are skipped.
``--finalize`` only recomputes metrics.json from the episodes on disk.

Examples::

    # arm 1 -- base student on the held-out set of one dataset
    python scripts/eval.py --arm student --student vllm --served-model student \
        --questions data/splits/test/heldout_musique_questions.jsonl \
        --corpus data/questions/musique/musique_corpus.jsonl.gz --out runs/eval/base/musique

    # arm 4 -- trained student (adapter served as "uniform")
    python scripts/eval.py --arm student --student vllm --served-model uniform ...

    # arm 2 -- base student guided by a teacher API
    python scripts/eval.py --arm guided --student vllm --served-model student \
        --teacher oai-teacher/deepseek-v4-flash --concurrency 6 ...

    # arm 3 -- the teacher alone
    python scripts/eval.py --arm teacher --agent-model oai-teacher/deepseek-v4-flash --concurrency 4 ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401
from tgd.io import append_jsonl, load_jsonl, read_jsonl
from tgd.logging_utils import setup_logger, write_json
from tgd.metrics import aggregate


def finalize(out: Path, args, log, extra=None) -> dict:
    episodes = load_jsonl(out / "episodes.jsonl") if (out / "episodes.jsonl").exists() else []
    agg = aggregate(episodes)
    agg.update({"arm": args.arm, "questions": args.questions, "corpus": args.corpus,
                "budget": args.budget, "hidden_budget": args.hidden_budget,
                "student": {"backend": args.student, "model": args.model, "adapter": args.adapter,
                            "served_model": args.served_model, "temperature": args.student_temperature},
                "teacher_router": args.teacher.split(",") if args.teacher else None,
                "agent_model": args.agent_model, "seed": args.seed, **(extra or {})})
    write_json(out / "metrics.json", agg)
    log.info(f"AGGREGATE: {json.dumps({k: v for k, v in agg.items() if not isinstance(v, dict)})}")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["student", "guided", "teacher"], required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="fixed output directory (resumable)")
    ap.add_argument("--budget", type=int, default=3, help="tool steps before a forced finish")
    ap.add_argument("--hidden-budget", action="store_true", default=True,
                    help="do not tell the agent its budget (default)")
    ap.add_argument("--disclose-budget", dest="hidden_budget", action="store_false")
    ap.add_argument("--no-plan", action="store_true", help="skip the planning turn")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n: run only shard i of n (0-based) of the questions")
    ap.add_argument("--seed", type=int, default=13)
    # student
    ap.add_argument("--student", choices=["vllm", "hf", "mock"], default="vllm",
                    help="mock = offline canned policy (smoke tests only)")
    ap.add_argument("--server-url", default="http://127.0.0.1:8300")
    ap.add_argument("--served-model", default="student", help="vLLM served name (base: student; adapter: its name)")
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b", help="HF id/path (hf backend, and recorded)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (hf backend)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--student-temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=16, help="student arm: episodes decoded in lockstep")
    # teacher / agent
    ap.add_argument("--teacher", default=None, help="guided arm: teacher model id(s), comma = fallback chain")
    ap.add_argument("--agent-model", default=None, help="teacher arm: the model that acts")
    ap.add_argument("--teacher-temperature", type=float, default=0.1)
    ap.add_argument("--teacher-max-tokens", type=int, default=2500)
    ap.add_argument("--student-max-tokens", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=6, help="guided/teacher arms: episodes in flight")
    ap.add_argument("--finalize", action="store_true", help="only recompute metrics.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("eval", out / "eval.log")
    if args.finalize:
        print(json.dumps(finalize(out, args, log), indent=2))
        return 0
    if args.arm == "guided" and not args.teacher:
        ap.error("--arm guided needs --teacher")
    if args.arm == "teacher" and not args.agent_model:
        ap.error("--arm teacher needs --agent-model")
    log.info(f"args: {vars(args)}")

    qpath = Path(args.questions)
    if not qpath.exists() and Path(str(qpath) + ".gz").exists():
        qpath = Path(str(qpath) + ".gz")
    questions = load_jsonl(qpath, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        questions = [q for j, q in enumerate(questions) if j % n == i]
    done_qids = {e["qid"] for e in read_jsonl(out / "episodes.jsonl")} if (out / "episodes.jsonl").exists() else set()
    todo = [q for q in questions if q["id"] not in done_qids]
    total = len(questions)
    log.info(f"{total} questions | {len(done_qids)} already done | {len(todo)} to run")
    if not todo:
        finalize(out, args, log)
        (out / ".done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print("nothing to do (already complete)")
        return 0

    import random
    random.seed(args.seed)
    from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever
    retriever = HotpotLocalRetriever(args.corpus)

    t0 = time.time()
    n_done = [len(done_qids)]

    def record(ep):
        append_jsonl(out / "episodes.jsonl", ep)
        n_done[0] += 1
        m = ep["final_metrics"]
        rate = (n_done[0] - len(done_qids)) / max(time.time() - t0, 1e-9)
        write_json(out / "status.json", {"arm": args.arm, "done": n_done[0], "total": total,
                                         "rate_per_min": round(rate * 60, 2),
                                         "eta_min": round((total - n_done[0]) / max(rate, 1e-9) / 60, 1)})
        log.info(f"[{n_done[0]}/{total}] qid={ep['qid']} em={m['exact_match']} f1={m['f1']} "
                 f"cover={m['cover_match']} steps={ep['used_steps']} stop={ep['stop_reason']} "
                 f"ans={str(ep['final_answer'])[:60]!r}")

    # ---- policy for the student arms
    policy = None
    if args.arm in ("student", "guided"):
        if args.student == "vllm":
            from tgd.vllm_backend import VllmPolicy, wait_ready
            served = wait_ready(args.server_url, args.served_model, timeout_s=300)
            policy = VllmPolicy(args.server_url, args.served_model, seed=args.seed,
                                max_parallel=max(args.batch_size, args.concurrency, 4))
            log.info(f"vLLM student: {args.server_url} model={args.served_model} (served: {served})")
        elif args.student == "mock":
            from tgd.mock_policy import MockPolicy
            policy = MockPolicy()
            log.info("MOCK student (smoke test only)")
        else:
            import numpy as np, torch
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            from tgd.hf_agent_loop import PolicyModel
            policy = PolicyModel(args.model, args.adapter, device=args.device)
            log.info(f"HF student: {args.model} adapter={args.adapter}")

    if args.arm == "student":
        from tgd.hf_agent_loop import run_episodes_batched
        run_episodes_batched(policy, todo, retriever, budget=args.budget,
                             disclose_budget=not args.hidden_budget, with_plan=not args.no_plan,
                             temperature=args.student_temperature, batch_size=args.batch_size,
                             on_episode=record, logger=log)
    else:
        from agentsim.clients.llm_client import LLMClient
        from tgd.guided_loop import LOCAL_STUDENT, HybridLLMClient, run_guided_episode
        client = HybridLLMClient(policy, LLMClient(), serialize_gpu=(args.student == "hf"))
        if args.arm == "guided":
            student_model, router, skip = LOCAL_STUDENT, args.teacher.split(","), False
        else:
            student_model, router, skip = args.agent_model, [args.agent_model], True
        meta = dict(corpus_path=args.corpus, disclose_budget=not args.hidden_budget,
                    student_model=student_model, teacher_router=router, skip_teacher=skip,
                    student_temperature=args.student_temperature,
                    teacher_temperature=args.teacher_temperature, with_plan=not args.no_plan,
                    teacher_max_tokens=args.teacher_max_tokens, student_max_tokens=args.student_max_tokens)

        async def runner():
            sem = asyncio.Semaphore(args.concurrency)

            async def one(row):
                async with sem:
                    try:
                        ep = await run_guided_episode(client, row, budget=args.budget, log=log, **meta)
                    except Exception as exc:  # noqa: BLE001 -- keep the run alive; the question is retried on resume
                        log.error(f"qid={row['id']} failed: {type(exc).__name__}: {str(exc)[:200]}")
                        return
                    record(ep)
            await asyncio.gather(*(one(r) for r in todo))
        asyncio.run(runner())

    extra = {"wall_time_s": round(time.time() - t0, 1)}
    try:
        import torch
        if torch.cuda.is_available():
            extra["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:  # noqa: BLE001
        pass
    agg = finalize(out, args, log, extra)
    remaining = total - n_done[0]
    if remaining == 0:
        (out / ".done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log.info("complete")
    else:
        log.warning(f"{remaining} questions have no episode (failed); re-run the same command to retry them")
    print(json.dumps({k: v for k, v in agg.items() if not isinstance(v, dict)}, indent=2))
    return 0 if remaining == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
