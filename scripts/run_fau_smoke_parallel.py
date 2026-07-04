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
    ap.add_argument("--student", default="ollama/qwen3.5:4b")
    ap.add_argument("--base-port", type=int, default=11500)
    ap.add_argument("--questions", default="./data/datasets/hotpot_teacher_guidance_exp30/hotpot_distractor_validation_questions.jsonl")
    ap.add_argument("--corpus", default="./data/datasets/hotpot_teacher_guidance_exp30/hotpot_distractor_validation_corpus.jsonl")
    args = ap.parse_args()

    if not (os.getenv("FAU_LLM_API_KEY") or os.getenv("LLMAPI_KEY")):
        sys.exit("Set FAU_LLM_API_KEY (or LLMAPI_KEY) in the environment first.")

    gpu_ids = _free_gpus(args.num_samples)
    workers = plan_workers(args.num_samples, gpu_ids, args.base_port)
    print(f"Planned {len(workers)} workers on GPUs {gpu_ids}")

    work_dir = REPO_ROOT / "data" / "simulation_output" / "fau_smoke"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    q_lines = read_question_lines(Path(args.questions), args.num_samples)
    student_tag = args.student.replace("ollama/", "", 1)

    written_templates: List[Path] = []
    servers: List[subprocess.Popen] = []
    try:
        # Materialize per-worker dataset + template.
        for w in workers:
            qpath = work_dir / f"w{w['index']}_questions.jsonl"
            qpath.write_text(q_lines[w["sample_index"]] + "\n", encoding="utf-8")
            template = build_fau_smoke_template(
                template_id=w["template_id"], student_model=args.student, num_samples=1,
                questions_path=str(qpath), corpus_path=args.corpus, output_dir=w["output_dir"],
            )
            tpath = TEMPLATES_DIR / f"{w['template_id']}.yaml"
            tpath.write_text(yaml.dump(template, sort_keys=False), encoding="utf-8")
            written_templates.append(tpath)

        # Start one Ollama server per GPU, then pull the student model once (shared
        # on-disk model store) via the first server.
        for w in workers:
            servers.append(_start_ollama(w["gpu_id"], w["port"], logs_dir / f"ollama_w{w['index']}.log"))
        print("All Ollama servers ready; pulling student model ...")
        pull_env = os.environ.copy()
        pull_env["OLLAMA_HOST"] = f"127.0.0.1:{workers[0]['port']}"
        subprocess.run(["ollama", "pull", student_tag], env=pull_env, check=True)

        # Launch all simulate processes in parallel.
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
            print(f"  -> worker {w['index']} (GPU {w['gpu_id']}, port {w['port']}) started")

        results = []
        for w, p, t0 in procs:
            code = p.wait()
            results.append((w, code, time.time() - t0))
            print(f"  <- worker {w['index']} exit={code} in {time.time() - t0:.0f}s")
    finally:
        for s in servers:
            _stop(s)
        for tpath in written_templates:
            tpath.unlink(missing_ok=True)

    # Summarize final answers.
    print("\n==== FAU smoke results ====")
    for w, code, secs in results:
        ep_files = list((REPO_ROOT / w["output_dir"].lstrip("./")).rglob("teacher_guidance_episodes.jsonl"))
        for ep_file in ep_files:
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ep = json.loads(line)
                fm = ep.get("final_metrics", {})
                print(f"w{w['index']} exit={code} correct={fm.get('answer_correct')} "
                      f"gold={ep.get('gold_answer')!r} final={str(ep.get('final_answer'))[:60]!r}")


if __name__ == "__main__":
    main()
