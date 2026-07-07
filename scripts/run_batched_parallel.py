"""
Batched multi-GPU teacher-guided run: several student episodes IN FLIGHT PER GPU.

Unlike run_fau_smoke_parallel.py (one sample per GPU at a time), this packs multiple
concurrent episodes onto each GPU -- worthwhile for small students (e.g. qwen3.5:0.8b) on
big cards (80 GB), where a single episode leaves the GPU almost idle and the real
bottleneck is the remote teacher latency, not local compute.

Design per GPU g (of the top --num-gpus free GPUs):
    ONE `ollama serve` bound to GPU g (CUDA_VISIBLE_DEVICES=g) on port base_port+g, started
    with OLLAMA_NUM_PARALLEL=--workers-per-gpu so it serves that many concurrent
    generations of the (single) student model.
    --workers-per-gpu `agentsim simulate` processes, each owning a round-robin shard of the
    samples, all pointing at GPU g's endpoint.

So concurrency = num_gpus * workers_per_gpu episodes at once. The teacher is the remote
FAU/OpenRouter router (no local GPU), so the only local pressure is the student model,
which is tiny and replicated cheaply by Ollama's parallel slots.

Usage:
    python scripts/run_batched_parallel.py --num-samples 100 --num-gpus 7 \
        --workers-per-gpu 4 --student ollama/qwen3.5:0.8b
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

import yaml  # noqa: E402

# Importing config loads .env (FAU/OpenRouter keys, Ollama endpoint) for this process.
from agentsim.config import config  # noqa: E402,F401
from scripts.gen_fau_smoke_template import build_fau_smoke_template, TEACHER_ROUTER  # noqa: E402
from scripts.consolidate_run_shards import consolidate  # noqa: E402

TEMPLATES_DIR = REPO_ROOT / "templates" / "simulations"
OLLAMA_READY_TIMEOUT_S = 180


# ---------------------------------------------------------------------------
# Pure planning (kept simple + deterministic for a quick self-check)
# ---------------------------------------------------------------------------
def plan_batched(
    num_samples: int, gpu_ids: List[str], workers_per_gpu: int, base_port: int = 11600,
    out_root: str = "data/simulation_output/batched_run", tag: str = "batched",
) -> List[Dict]:
    """One shared Ollama server per GPU + ``workers_per_gpu`` simulate workers per GPU.

    Returns a flat list of worker dicts; workers on the same GPU share ``gpu_id``/``port``/
    ``endpoint`` (one server) but own disjoint round-robin sample shards. Total workers is
    capped at ``num_samples`` so no worker is empty."""
    if not gpu_ids:
        raise ValueError("no gpu_ids provided")
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be >= 1")
    num_workers = min(len(gpu_ids) * workers_per_gpu, num_samples)
    num_gpus = len(gpu_ids)
    workers = []
    for w in range(num_workers):
        g = w % num_gpus  # spread workers across GPUs first, then stack
        workers.append({
            "index": w,
            "gpu_index": g,
            "gpu_id": gpu_ids[g],
            "port": base_port + g,
            "endpoint": f"http://127.0.0.1:{base_port + g}",
            "sample_indices": list(range(w, num_samples, num_workers)),
            "template_id": f"{tag}_w{w}",
            "output_dir": f"./{out_root.rstrip('/')}/w{w}",
        })
    return workers


def read_question_lines(path: Path, n: int) -> List[str]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:n]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _free_gpus(n: int) -> List[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, mem = [p.strip() for p in line.split(",")]
        rows.append((int(idx), int(mem)))
    rows.sort(key=lambda r: r[1])
    return [str(idx) for idx, _ in rows[:n]]


def _start_ollama(gpu_id: str, port: int, num_parallel: int, log_path: Path) -> subprocess.Popen:
    import httpx
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)   # concurrent generations per server
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"            # single student model
    env["OLLAMA_KEEP_ALIVE"] = "-1"                  # keep it resident for the whole run
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=log_f, stderr=subprocess.STDOUT)
    deadline = time.time() + OLLAMA_READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"ollama on port {port} (GPU {gpu_id}) not ready in {OLLAMA_READY_TIMEOUT_S}s")


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--num-gpus", type=int, default=7)
    ap.add_argument("--gpu-ids", default=None,
                    help="comma-separated GPU ids to pin (e.g. '1,2,3'); overrides "
                         "--num-gpus auto-selection so concurrent runs use disjoint GPUs")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--student", default="ollama/qwen3.5:0.8b")
    ap.add_argument("--base-port", type=int, default=11600)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--workflow", default="hotpot_teacher_guided_b5_plan_review")
    ap.add_argument("--planning-steps", type=int, default=1)
    ap.add_argument("--max-plan-steps", type=int, default=6)
    ap.add_argument("--student-no-schema", action="store_true",
                    help="disable grammar-constrained student decoding (default: on)")
    ap.add_argument("--hidden-budget", action="store_true")
    ap.add_argument("--wiki", action="store_true",
                    help="enable the per-episode agent wiki (wiki_read/wiki_write tools)")
    ap.add_argument("--teacher-model", default="fau/gpt-oss-120b",
                    help="teacher model id (first provider tried)")
    ap.add_argument("--teacher-router", default=None,
                    help="comma-separated teacher fallback chain overriding the default "
                         "FAU-first router (e.g. when FAU gpt-oss-120b is unresponsive)")
    ap.add_argument("--out-root", default="data/simulation_output/qwen0p8b_run100")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--tag", default="batched",
                    help="unique per-run prefix for the transient worker template ids "
                         "(set distinct tags for concurrent runs so they don't clash)")
    ap.add_argument("--no-consolidate", action="store_true")
    ap.add_argument("--questions", default="./data/datasets/hotpot_teacher_guidance_exp100/hotpot_distractor_validation_questions.jsonl")
    ap.add_argument("--corpus", default="./data/datasets/hotpot_teacher_guidance_exp100/hotpot_distractor_validation_corpus.jsonl")
    args = ap.parse_args()

    if not config.provider_available("fau/gpt-oss-120b") and not config.provider_available("custom/openai/gpt-oss-120b"):
        sys.exit("No teacher provider configured (set FAU_LLM_API_KEY or CUSTOM_LLM_API_KEY in .env).")

    teacher_router = ([r.strip() for r in args.teacher_router.split(",") if r.strip()]
                      if args.teacher_router else list(TEACHER_ROUTER))
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()] if args.gpu_ids else _free_gpus(args.num_gpus)
    workers = plan_batched(args.num_samples, gpu_ids, args.workers_per_gpu, args.base_port,
                           out_root=args.out_root, tag=args.tag)
    per_gpu = {}
    for w in workers:
        per_gpu.setdefault(w["gpu_id"], []).append(w["index"])
    print(f"GPUs {gpu_ids} | {len(workers)} workers "
          f"({args.workers_per_gpu}/GPU) | shards {[len(w['sample_indices']) for w in workers]}", flush=True)

    work_dir = REPO_ROOT / args.out_root
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    q_lines = read_question_lines(Path(args.questions), args.num_samples)
    student_tag = args.student.replace("ollama/", "", 1)

    written_templates: List[Path] = []
    servers: List[subprocess.Popen] = []
    t_start = time.time()
    try:
        for w in workers:
            shard_lines = [q_lines[i] for i in w["sample_indices"]]
            qpath = work_dir / f"w{w['index']}_questions.jsonl"
            qpath.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")
            template = build_fau_smoke_template(
                template_id=w["template_id"], student_model=args.student,
                teacher_model=args.teacher_model, teacher_router=teacher_router,
                num_samples=len(shard_lines), questions_path=str(qpath), corpus_path=args.corpus,
                output_dir=w["output_dir"], budget=args.budget, workflow=args.workflow,
                planning_steps=args.planning_steps, max_plan_steps=args.max_plan_steps,
                student_use_response_schema=not args.student_no_schema,
                disclose_budget=not args.hidden_budget,
                wiki_enabled=args.wiki,
            )
            tpath = TEMPLATES_DIR / f"{w['template_id']}.yaml"
            tpath.write_text(yaml.dump(template, sort_keys=False), encoding="utf-8")
            written_templates.append(tpath)

        # One Ollama server per unique GPU.
        first_by_gpu = {}
        for w in workers:
            if w["gpu_id"] not in first_by_gpu:
                first_by_gpu[w["gpu_id"]] = w
                servers.append(_start_ollama(
                    w["gpu_id"], w["port"], args.workers_per_gpu,
                    logs_dir / f"ollama_gpu{w['gpu_id']}.log",
                ))
        print(f"{len(servers)} Ollama servers ready; pulling {student_tag} ...", flush=True)
        pull_env = os.environ.copy()
        pull_env["OLLAMA_HOST"] = f"127.0.0.1:{workers[0]['port']}"
        subprocess.run(["ollama", "pull", student_tag], env=pull_env, check=True)
        # Pull on every other server too (each GPU's server has its own resident copy).
        for gpu_id, w in first_by_gpu.items():
            if w is workers[0]:
                continue
            e = os.environ.copy(); e["OLLAMA_HOST"] = f"127.0.0.1:{w['port']}"
            subprocess.run(["ollama", "pull", student_tag], env=e, check=True)

        procs = []
        for w in workers:
            env = os.environ.copy()
            env["OLLAMA_ENDPOINT"] = w["endpoint"]
            env["OLLAMA_ENABLED"] = "true"
            log_f = open(logs_dir / f"sim_w{w['index']}.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [sys.executable, "-m", "agentsim.cli", "simulate", w["template_id"]],
                cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            )
            procs.append((w, p, time.time()))
        print(f"launched {len(procs)} simulate workers", flush=True)

        results = []
        for w, p, t0 in procs:
            code = p.wait()
            results.append((w, code, time.time() - t0))
            print(f"  <- worker {w['index']} (GPU {w['gpu_id']}) exit={code} in {time.time() - t0:.0f}s", flush=True)
    finally:
        for s in servers:
            _stop(s)
        for tpath in written_templates:
            tpath.unlink(missing_ok=True)

    correct = total = unknown = 0
    for w, code, secs in results:
        for ep_file in (REPO_ROOT / w["output_dir"].lstrip("./")).rglob("teacher_guidance_episodes.jsonl"):
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ep = json.loads(line)
                total += 1
                if (ep.get("final_metrics", {}) or {}).get("answer_correct"):
                    correct += 1
                if str(ep.get("final_answer", "")).strip().lower() == "unknown":
                    unknown += 1
    print(f"\n==== batched run done in {time.time() - t_start:.0f}s ====", flush=True)
    if total:
        print(f"episodes: {total} | answer_correct: {correct} ({correct/total:.1%}) | unknown: {unknown}", flush=True)
    else:
        print("no episodes produced", flush=True)

    if not args.no_consolidate and total:
        moved, removed = consolidate(work_dir, args.run_name)
        print(f"consolidated {moved} episodes into '{work_dir.name}/{args.run_name}'", flush=True)
    print(f"output root: {work_dir}", flush=True)


if __name__ == "__main__":
    main()
