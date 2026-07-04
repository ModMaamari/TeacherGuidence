"""
Sequentially run the 4-model x 5-setting experiment matrix (see
gen_experiment_matrix_templates.py), one template at a time, grouped by student model
so each Ollama model is pulled/served once instead of once per setting.

For each model group:
    1. Pick the N currently-freest GPUs (nvidia-smi memory.used, ascending).
    2. Start `ollama serve` bound to those GPUs via CUDA_VISIBLE_DEVICES.
    3. `ollama pull` the model.
    4. Start a background nvidia-smi poll (memory/utilization every --gpu-poll-interval
       seconds) logged to a per-model file.
    5. Run each of the model's 5 settings (`agentsim simulate <template_id>`) in turn,
       logging stdout/stderr and wall-clock time; append one manifest row per finished
       template as it completes (crash-safe: earlier rows survive a later failure).
    6. Stop the GPU poll and the Ollama server before moving to the next model.

Resuming after a crash: by default, any (model, setting) that already has a successful
(exit_code == 0) row in an existing manifest at --out-dir is skipped -- this matters
because `agentsim simulate` itself only resumes a run whose checkpoint status is still
"running"; a completed run's checkpoint is marked "completed", so re-invoking it starts
a brand-new run from scratch (re-doing real, costed OpenRouter teacher calls) rather
than doing nothing. Pass --force-rerun-all to ignore the existing manifest and run
every config again regardless.

Usage:
    python scripts/run_experiment_matrix.py --out-dir reports/experiment_matrix_2026-07-03
    python scripts/run_experiment_matrix.py --dry-run   # print the plan, run nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gen_experiment_matrix_templates import MODELS, SETTINGS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_READY_TIMEOUT_S = 120


def plan_runs(models: Dict[str, str] = MODELS, settings: Dict[str, dict] = SETTINGS) -> List[Tuple[str, str, str]]:
    """Return [(model_slug, setting_key, template_id), ...] grouped by model, in a
    stable order, so each model's Ollama pull/serve happens exactly once."""
    runs = []
    for model_slug in models:
        for setting_key in settings:
            runs.append((model_slug, setting_key, f"exp_matrix_{model_slug}_{setting_key}"))
    return runs


def load_completed_configs(manifest_path: Path) -> set:
    """Return the {(model_slug, setting), ...} pairs with a successful (exit_code == 0)
    row in an existing manifest.jsonl, so a restarted run can skip them -- re-invoking
    `agentsim simulate` on an already-completed config starts an entirely new (costed)
    run rather than a no-op, so these must be skipped explicitly, not just left to the
    simulator's own checkpoint resume (which only resumes a still-"running" checkpoint).
    """
    completed = set()
    if not manifest_path.exists():
        return completed
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("exit_code") == 0:
                completed.add((row["model_slug"], row["setting"]))
    return completed


def select_free_gpus(nvidia_smi_csv: str, n: int = 4) -> List[str]:
    """Parse `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`
    output and return the n GPU indices with the least memory used, ascending."""
    rows = []
    for line in nvidia_smi_csv.strip().splitlines():
        if not line.strip():
            continue
        idx_str, mem_str = [p.strip() for p in line.split(",")]
        rows.append((int(idx_str), int(mem_str)))
    rows.sort(key=lambda r: r[1])
    return [str(idx) for idx, _ in rows[:n]]


def _nvidia_smi_free_gpus(n: int) -> List[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return select_free_gpus(out, n)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _start_ollama_server(gpu_ids: List[str], log_path: Path) -> subprocess.Popen:
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=log_f, stderr=subprocess.STDOUT)

    import httpx
    deadline = time.time() + OLLAMA_READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            httpx.get("http://127.0.0.1:11434/", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"ollama serve did not become ready within {OLLAMA_READY_TIMEOUT_S}s")


def _stop_process(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def _pull_model(model_id: str, log_path: Path) -> None:
    ollama_tag = model_id.replace("ollama/", "", 1)
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(["ollama", "pull", ollama_tag], stdout=f, stderr=subprocess.STDOUT, check=True)


def _start_gpu_monitor(log_path: Path, interval_s: int) -> subprocess.Popen:
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
            "--format=csv",
            "-l", str(interval_s),
        ],
        stdout=log_f, stderr=subprocess.STDOUT,
    )


def _run_template(template_id: str, log_path: Path) -> Tuple[int, float]:
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, "-m", "agentsim.cli", "simulate", template_id],
            cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT,
        )
    return result.returncode, time.time() - t0


def run_matrix(
    out_dir: Path, gpu_count: int = 4, gpu_poll_interval: int = 10, dry_run: bool = False,
    force_rerun_all: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    completed = set() if force_rerun_all else load_completed_configs(manifest_path)
    if completed:
        print(f"Skipping {len(completed)} already-completed config(s) found in {manifest_path}: "
              f"{sorted(completed)}", flush=True)

    runs = [r for r in plan_runs() if (r[0], r[1]) not in completed]
    by_model: Dict[str, List[Tuple[str, str, str]]] = {}
    for model_slug, setting_key, template_id in runs:
        by_model.setdefault(model_slug, []).append((model_slug, setting_key, template_id))

    if dry_run:
        for model_slug, group in by_model.items():
            print(f"[dry-run] model={model_slug} ({MODELS[model_slug]})")
            for _, setting_key, template_id in group:
                print(f"[dry-run]   setting={setting_key} -> {template_id}")
        return

    if not by_model:
        print("Nothing to do -- every config already has a successful manifest entry.")
        return

    with open(manifest_path, "a", encoding="utf-8") as manifest_f:
        for model_slug, group in by_model.items():
            print(f"=== model group: {model_slug} ({MODELS[model_slug]}) ===", flush=True)
            gpu_ids = _nvidia_smi_free_gpus(gpu_count)
            print(f"  GPUs: {gpu_ids}", flush=True)

            server_proc = _start_ollama_server(gpu_ids, logs_dir / f"{model_slug}_ollama_serve.log")
            try:
                _pull_model(MODELS[model_slug], logs_dir / f"{model_slug}_pull.log")

                gpu_log_path = logs_dir / f"{model_slug}_gpu.csv"
                gpu_proc = _start_gpu_monitor(gpu_log_path, gpu_poll_interval)
                try:
                    for _, setting_key, template_id in group:
                        print(f"  -> running {template_id}", flush=True)
                        started_at = _utcnow_iso()
                        run_log = logs_dir / f"{template_id}.log"
                        exit_code, wall_seconds = _run_template(template_id, run_log)
                        row = {
                            "model_slug": model_slug,
                            "model_id": MODELS[model_slug],
                            "setting": setting_key,
                            "template_id": template_id,
                            "gpu_ids": gpu_ids,
                            "started_at": started_at,
                            "ended_at": _utcnow_iso(),
                            "wall_seconds": wall_seconds,
                            "exit_code": exit_code,
                            "run_log": str(run_log.relative_to(out_dir)),
                            "gpu_log": str(gpu_log_path.relative_to(out_dir)),
                        }
                        manifest_f.write(json.dumps(row) + "\n")
                        manifest_f.flush()
                        status = "ok" if exit_code == 0 else f"FAILED (exit {exit_code})"
                        print(f"     {status} in {wall_seconds:.0f}s", flush=True)
                finally:
                    _stop_process(gpu_proc)
            finally:
                _stop_process(server_proc)

    print(f"Done. Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/experiment_matrix_2026-07-03")
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-poll-interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan and exit")
    parser.add_argument(
        "--force-rerun-all", action="store_true",
        help="Ignore any existing manifest and run every config again, even ones already "
             "marked successful (by default those are skipped -- see module docstring)",
    )
    args = parser.parse_args()

    run_matrix(
        Path(args.out_dir), gpu_count=args.gpu_count,
        gpu_poll_interval=args.gpu_poll_interval, dry_run=args.dry_run,
        force_rerun_all=args.force_rerun_all,
    )


if __name__ == "__main__":
    main()
