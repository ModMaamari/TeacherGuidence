"""
Resumable teacher-guidance run with a vLLM-served student.

Same shape as run_batched_parallel.py (many concurrent episodes per GPU) but the student
is served by vLLM instead of Ollama, and the run is RESUMABLE:

  * vLLM per GPU: one OpenAI-compatible server per GPU, continuously batching every
    episode assigned to that GPU. A 3B student on an 80GB A100 leaves almost all memory to
    the KV cache, so --max-num-seqs can be large and student latency barely grows with
    concurrency. The model is served under its real HF id, so workers address it as
    `vllm/<hf-id>` via a per-worker VLLM_ENDPOINT and traces record the real student.
  * Resume: the episodes already on disk are the source of truth (see tg_run_status). Each
    launch re-derives the UNANSWERED qids and shards only those, so re-running the same
    command after any interruption continues where it stopped and never redoes work.

Note the real bottleneck is the remote reasoning teacher (~10-20s/call), not the GPU:
raising --workers-per-gpu raises teacher concurrency, which is what actually shortens wall
time (at the cost of more fallthrough to paid fallbacks when the primary throttles).

Usage:
    python scripts/run_tg_vllm.py --num-samples 3000 --num-gpus 3 --workers-per-gpu 12 \
        --budget 3 --planning-steps 3 --hidden-budget
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import yaml  # noqa: E402

from agentsim.config import config  # noqa: E402,F401  (imports .env)
from scripts.gen_fau_smoke_template import build_fau_smoke_template  # noqa: E402
from scripts.consolidate_run_shards import consolidate  # noqa: E402
from scripts.tg_run_status import compute_status  # noqa: E402

TEMPLATES_DIR = REPO_ROOT / "templates" / "simulations"
VLLM_READY_TIMEOUT_S = 900  # cold start: weight load + kernel compile


def shard_round_robin(items: List, num_workers: int) -> List[List]:
    """Deal ``items`` round-robin into ``num_workers`` shards (no empty shards)."""
    n = max(1, min(num_workers, len(items)))
    return [items[i::n] for i in range(n)]


def _free_gpus(n: int) -> List[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [(int(i), int(f)) for i, f in (l.split(",") for l in out.strip().splitlines() if l.strip())]
    rows.sort(key=lambda r: -r[1])  # most free first
    return [str(i) for i, _ in rows[:n]]


def _start_vllm(gpu_id: str, port: int, model: str, mem_util: float, max_num_seqs: int,
                max_model_len: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    # vLLM JIT-compiles kernels at startup and needs the venv's ninja on PATH.
    env["PATH"] = f"{REPO_ROOT}/.venv_vllm/bin:{env.get('PATH','')}"
    cmd = [
        str(REPO_ROOT / ".venv_vllm/bin/vllm"), "serve", model,
        # Serve under the REAL model id, not a placeholder like "student": the served name
        # is what episodes record as student_model (as `vllm/<served-name>`), and a trace
        # that says "vllm/student" has lost the one fact that matters -- which model it was.
        "--served-model-name", model,
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(mem_util),
        "--max-num-seqs", str(max_num_seqs),
        # Tens of thousands of student calls per run: keep the per-request access log out
        # of the server log. (vLLM 0.25 removed --disable-log-requests.)
        "--disable-uvicorn-access-log",
    ]
    log_f = open(log_path, "w", encoding="utf-8")
    # Popen directly (no shell pipe/tee): the pipeline form detaches the real vLLM PID and
    # leaves servers holding GPU memory after teardown.
    return subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)


def _wait_ready(ports: List[int], procs: List[subprocess.Popen]) -> None:
    deadline = time.time() + VLLM_READY_TIMEOUT_S
    pending = set(ports)
    while pending and time.time() < deadline:
        for p in list(pending):
            try:
                if httpx.get(f"http://127.0.0.1:{p}/v1/models", timeout=2).status_code == 200:
                    pending.discard(p)
                    print(f"  vLLM ready on :{p}", flush=True)
            except Exception:
                pass
        for proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"a vLLM server exited early (rc={proc.returncode}); check its log")
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"vLLM not ready in {VLLM_READY_TIMEOUT_S}s on ports {sorted(pending)}")


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--num-gpus", type=int, default=3)
    ap.add_argument("--gpu-ids", default=None, help="comma-separated GPU ids (overrides --num-gpus)")
    ap.add_argument("--workers-per-gpu", type=int, default=12)
    ap.add_argument("--student-model", default="ibm-granite/granite-4.1-3b",
                    help="HF id served by vLLM; also the served model name, so workers "
                         "address it as vllm/<hf-id> and traces record the real model")
    ap.add_argument("--budget", type=int, default=3, help="running-step budget")
    ap.add_argument("--planning-steps", type=int, default=3, help="plan-review rounds")
    ap.add_argument("--max-plan-steps", type=int, default=6)
    ap.add_argument("--hidden-budget", action="store_true",
                    help="do NOT disclose the remaining budget to the student")
    ap.add_argument("--teacher-model", default="edenai/lilac/minimaxai/minimax-m3")
    ap.add_argument("--teacher-router",
                    default="edenai/lilac/minimaxai/minimax-m3,"
                            "fau/MiniMaxAI/MiniMax-M3-MXFP8")
    ap.add_argument("--base-port", type=int, default=8400)
    ap.add_argument("--mem-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--out-root", default="data/simulation_output/tg_m3_granite_b3")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--tag", default="tgm3")
    ap.add_argument("--questions", default="./data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl")
    ap.add_argument("--corpus", default="./data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl")
    ap.add_argument("--no-consolidate", action="store_true")
    args = ap.parse_args()

    if Path(args.out_root).is_absolute():
        args.out_root = os.path.relpath(args.out_root, REPO_ROOT)
    work_dir = REPO_ROOT / args.out_root
    questions_path = Path(args.questions)

    # ---- resume: only run what is still unanswered -------------------------------
    status = compute_status(work_dir, questions_path, args.num_samples)
    print(f"=== resume check: {status['answered']}/{status['requested']} answered, "
          f"{status['unanswered']} remaining ===", flush=True)
    if status["unanswered"] == 0:
        print("nothing to do -- every requested question already has an episode.")
        return
    todo_qids = set(status["unanswered_qids"])
    q_lines = [
        line for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and str(json.loads(line)["id"]) in todo_qids
    ]
    print(f"running {len(q_lines)} questions this pass", flush=True)

    teacher_router = [r.strip() for r in args.teacher_router.split(",") if r.strip()]
    gpu_ids = ([g.strip() for g in args.gpu_ids.split(",") if g.strip()]
               if args.gpu_ids else _free_gpus(args.num_gpus))
    num_workers = min(len(gpu_ids) * args.workers_per_gpu, len(q_lines))
    shards = shard_round_robin(q_lines, num_workers)
    workflow = f"hotpot_teacher_guided_b{args.budget}_plan_review"

    print(f"GPUs {gpu_ids} | {len(shards)} workers ({args.workers_per_gpu}/GPU) | "
          f"workflow={workflow} | budget={args.budget} planning={args.planning_steps} "
          f"hidden_budget={args.hidden_budget}", flush=True)

    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    written_templates: List[Path] = []
    servers: List[subprocess.Popen] = []
    t_start = time.time()
    try:
        # ---- one vLLM server per GPU ---------------------------------------------
        ports = [args.base_port + int(g) for g in gpu_ids]
        for gpu_id, port in zip(gpu_ids, ports):
            print(f"  starting vLLM GPU {gpu_id} :{port} ({args.student_model})", flush=True)
            servers.append(_start_vllm(
                gpu_id, port, args.student_model, args.mem_util, args.max_num_seqs,
                args.max_model_len, logs_dir / f"vllm_gpu{gpu_id}.log",
            ))
        _wait_ready(ports, servers)

        # ---- per-worker shard + template ------------------------------------------
        procs = []
        for w, shard in enumerate(shards):
            gpu_slot = w % len(gpu_ids)
            port = ports[gpu_slot]
            qpath = work_dir / f"{args.tag}_w{w}_questions.jsonl"
            qpath.write_text("\n".join(shard) + "\n", encoding="utf-8")
            template_id = f"{args.tag}_w{w}"
            template = build_fau_smoke_template(
                template_id=template_id,
                student_model=f"vllm/{args.student_model}",
                teacher_model=args.teacher_model,
                teacher_router=teacher_router,
                num_samples=len(shard),
                questions_path=str(qpath),
                corpus_path=args.corpus,
                output_dir=f"./{args.out_root.rstrip('/')}/w{w}",
                budget=args.budget,
                workflow=workflow,
                planning_steps=args.planning_steps,
                max_plan_steps=args.max_plan_steps,
                student_use_response_schema=True,
                disclose_budget=not args.hidden_budget,
            )
            tpath = TEMPLATES_DIR / f"{template_id}.yaml"
            tpath.write_text(yaml.dump(template, sort_keys=False), encoding="utf-8")
            written_templates.append(tpath)

            env = os.environ.copy()
            env["VLLM_ENDPOINT"] = f"http://127.0.0.1:{port}"
            env["OLLAMA_ENABLED"] = "false"
            log_f = open(logs_dir / f"sim_w{w}.log", "w", encoding="utf-8")
            procs.append((w, subprocess.Popen(
                [sys.executable, "-m", "agentsim.cli", "simulate", template_id],
                cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            )))
        print(f"launched {len(procs)} simulate workers", flush=True)

        for w, p in procs:
            code = p.wait()
            print(f"  <- worker {w} exit={code} ({time.time() - t_start:.0f}s elapsed)", flush=True)
    finally:
        for s in servers:
            _stop(s)
        for tpath in written_templates:
            tpath.unlink(missing_ok=True)

    if not args.no_consolidate:
        moved, _ = consolidate(work_dir, args.run_name)
        print(f"consolidated {moved} episodes into '{work_dir.name}/{args.run_name}'", flush=True)

    final = compute_status(work_dir, questions_path, args.num_samples)
    print(f"\n==== pass done in {time.time() - t_start:.0f}s ====", flush=True)
    print(f"answered {final['answered']}/{final['requested']} ({final['pct_done']:.1f}%) | "
          f"unanswered {final['unanswered']}", flush=True)
    if final["answered"]:
        print(f"answer_correct: {final['answer_correct']}/{final['answered']} "
              f"({final['answer_correct_pct']:.1f}%) | teachers: {final['teacher_models_used']}", flush=True)
    print(f"output root: {work_dir}", flush=True)


if __name__ == "__main__":
    main()
