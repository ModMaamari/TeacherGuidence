"""
Run the FAU gpt-oss-120b teacher smoke with one episode per GPU, all in parallel.

Each of N samples is handled by its own worker: a dedicated `ollama serve` bound to a
single GPU (via CUDA_VISIBLE_DEVICES) on its own port, plus one `agentsim simulate`
process pointed at a one-question dataset and at that GPU's Ollama endpoint. The teacher
(fau/gpt-oss-120b) is a remote gateway, so it consumes no local GPU -- the GPUs are used
only for the local student model, exactly one episode each.

Layout per worker i:
    OLLAMA_HOST=127.0.0.1:(base_port+i), CUDA_VISIBLE_DEVICES=gpu_ids[i]
    dataset  -> <work_dir>/w{i}_questions.jsonl   (one line: sample i)
    template -> templates/simulations/fau_smoke_w{i}.yaml   (num_samples=1)
    output   -> data/simulation_output/fau_smoke/w{i}

Usage:
    FAU_LLM_API_KEY=sk-... python scripts/run_fau_smoke_parallel.py
    python scripts/run_fau_smoke_parallel.py --num-samples 8 --student ollama/qwen3.5:4b
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

from scripts.gen_fau_smoke_template import build_fau_smoke_template  # noqa: E402
from scripts.consolidate_run_shards import consolidate  # noqa: E402

TEMPLATES_DIR = REPO_ROOT / "templates" / "simulations"
OLLAMA_READY_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Pure planning (unit-tested)
# ---------------------------------------------------------------------------
def plan_workers(num_samples: int, gpu_ids: List[str], base_port: int = 11500) -> List[Dict]:
    """One worker per sample. Each worker gets a distinct GPU when num_samples <=
    len(gpu_ids) (the intended "one GPU per episode"); if there are more samples than
    GPUs it round-robins so nothing is dropped."""
    if not gpu_ids:
        raise ValueError("no gpu_ids provided")
    workers = []
    for i in range(num_samples):
        workers.append({
            "index": i,
            "sample_index": i,
            "gpu_id": gpu_ids[i % len(gpu_ids)],
            "port": base_port + i,
            "endpoint": f"http://127.0.0.1:{base_port + i}",
            "template_id": f"fau_smoke_w{i}",
            "output_dir": f"./data/simulation_output/fau_smoke/w{i}",
        })
    return workers


def plan_shards(
    num_samples: int, gpu_ids: List[str], base_port: int = 11500,
    out_root: str = "data/simulation_output/fau_run",
) -> List[Dict]:
    """A pool of one worker per GPU, each owning a round-robin shard of the samples.

    With more samples than GPUs, each GPU's worker processes its shard sequentially, so at
    any instant there is exactly one sample in flight per GPU (the requested "one sample
    per GPU, in parallel across GPUs"). Round-robin keeps shard sizes within 1 of each
    other."""
    if not gpu_ids:
        raise ValueError("no gpu_ids provided")
    num_workers = min(len(gpu_ids), num_samples)
    workers = []
    for w in range(num_workers):
        workers.append({
            "index": w,
            "sample_indices": list(range(w, num_samples, num_workers)),
            "gpu_id": gpu_ids[w],
            "port": base_port + w,
            "endpoint": f"http://127.0.0.1:{base_port + w}",
            "template_id": f"fau_run_w{w}",
            "output_dir": f"./{out_root.rstrip('/')}/w{w}",
        })
    return workers


def read_question_lines(path: Path, n: int) -> List[str]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:n]


# ---------------------------------------------------------------------------
# Side-effecting orchestration
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


def _start_ollama(gpu_id: str, port: int, log_path: Path) -> subprocess.Popen:
    import httpx
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=log_f, stderr=subprocess.STDOUT)
    deadline = time.time() + OLLAMA_READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"ollama on port {port} not ready in {OLLAMA_READY_TIMEOUT_S}s")


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--student", default="ollama/qwen3.5:2b")
    ap.add_argument("--base-port", type=int, default=11500)
    ap.add_argument("--budget", type=int, default=12, help="running (tool-use) step budget")
    ap.add_argument("--workflow", default="hotpot_teacher_guided_b12_plan_review")
    ap.add_argument("--planning-steps", type=int, default=3, help="plan review/revise rounds")
    ap.add_argument("--max-plan-steps", type=int, default=12)
    ap.add_argument("--student-no-schema", action="store_true",
                    help="disable grammar-constrained decoding for student calls (models "
                         "that collapse under llama.cpp grammar constraints, e.g. MiniCPM5)")
    ap.add_argument("--hidden-budget", action="store_true",
                    help="do not disclose the step budget to the student (efficient-plan / "
                         "answer-ASAP mode; budget revealed only on the forced last step)")
    ap.add_argument("--out-root", default="data/simulation_output/fau_run")
    ap.add_argument("--run-name", default="run", help="consolidated run dir name under out-root")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="keep per-worker shard dirs instead of merging them into one run")
    ap.add_argument("--questions", default="./data/datasets/hotpot_teacher_guidance_exp100/hotpot_distractor_validation_questions.jsonl")
    ap.add_argument("--corpus", default="./data/datasets/hotpot_teacher_guidance_exp100/hotpot_distractor_validation_corpus.jsonl")
    args = ap.parse_args()

    if not (os.getenv("FAU_LLM_API_KEY") or os.getenv("LLMAPI_KEY")):
        sys.exit("Set FAU_LLM_API_KEY (or LLMAPI_KEY) in the environment first.")

    gpu_ids = _free_gpus(args.num_gpus)
    workers = plan_shards(args.num_samples, gpu_ids, args.base_port, out_root=args.out_root)
    print(f"Planned {len(workers)} GPU workers on {gpu_ids} for {args.num_samples} samples "
          f"(shards: {[len(w['sample_indices']) for w in workers]})", flush=True)

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
        # Materialize per-worker dataset shard + template.
        for w in workers:
            shard_lines = [q_lines[i] for i in w["sample_indices"]]
            qpath = work_dir / f"w{w['index']}_questions.jsonl"
            qpath.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")
            template = build_fau_smoke_template(
                template_id=w["template_id"], student_model=args.student,
                num_samples=len(shard_lines), questions_path=str(qpath), corpus_path=args.corpus,
                output_dir=w["output_dir"], budget=args.budget, workflow=args.workflow,
                planning_steps=args.planning_steps, max_plan_steps=args.max_plan_steps,
                student_use_response_schema=not args.student_no_schema,
                disclose_budget=not args.hidden_budget,
            )
            tpath = TEMPLATES_DIR / f"{w['template_id']}.yaml"
            tpath.write_text(yaml.dump(template, sort_keys=False), encoding="utf-8")
            written_templates.append(tpath)

        # Start one Ollama server per GPU, then pull the student model once (shared
        # on-disk model store) via the first server.
        for w in workers:
            servers.append(_start_ollama(w["gpu_id"], w["port"], logs_dir / f"ollama_w{w['index']}.log"))
        print("All Ollama servers ready; pulling student model ...", flush=True)
        pull_env = os.environ.copy()
        pull_env["OLLAMA_HOST"] = f"127.0.0.1:{workers[0]['port']}"
        subprocess.run(["ollama", "pull", student_tag], env=pull_env, check=True)

        # Launch all worker simulate processes in parallel (one per GPU).
        procs = []
        for w in workers:
            env = os.environ.copy()
            env["OLLAMA_ENDPOINT"] = w["endpoint"]
            env["OLLAMA_ENABLED"] = "true"
            if os.getenv("LLMAPI_KEY") and not os.getenv("FAU_LLM_API_KEY"):
                env["FAU_LLM_API_KEY"] = os.environ["LLMAPI_KEY"]
            log_f = open(logs_dir / f"sim_w{w['index']}.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [sys.executable, "-m", "agentsim.cli", "simulate", w["template_id"]],
                cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            )
            procs.append((w, p, time.time()))
            print(f"  -> worker {w['index']} (GPU {w['gpu_id']}, {len(w['sample_indices'])} samples) started", flush=True)

        results = []
        for w, p, t0 in procs:
            code = p.wait()
            results.append((w, code, time.time() - t0))
            print(f"  <- worker {w['index']} exit={code} in {time.time() - t0:.0f}s", flush=True)
    finally:
        for s in servers:
            _stop(s)
        for tpath in written_templates:
            tpath.unlink(missing_ok=True)

    # Summarize correctness across all workers.
    correct = total = 0
    for w, code, secs in results:
        for ep_file in (REPO_ROOT / w["output_dir"].lstrip("./")).rglob("teacher_guidance_episodes.jsonl"):
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                total += 1
                if (json.loads(line).get("final_metrics", {}) or {}).get("answer_correct"):
                    correct += 1
    print(f"\n==== FAU run done in {time.time() - t_start:.0f}s ====", flush=True)
    print(f"episodes: {total} | answer_correct: {correct} ({correct / total:.1%})" if total else "no episodes",
          flush=True)

    # Merge the per-GPU shard dirs into a single run so the explorer shows one run.
    if not args.no_consolidate and total:
        moved, removed = consolidate(work_dir, args.run_name)
        print(f"consolidated {moved} episodes into one run '{work_dir.name}/{args.run_name}' "
              f"(removed shard dirs: {removed})", flush=True)
    print(f"output root: {work_dir}", flush=True)


if __name__ == "__main__":
    main()
