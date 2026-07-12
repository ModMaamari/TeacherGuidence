"""exp_unseen100: repeated three-arm evaluation on 100 unseen HotpotQA questions.

Arms (budget 4, student temperature 0.2, sampled decoding):
    base_teacher  granite-4.1-3b base + live gpt-oss-120b teacher (plan review +
                  per-step guidance) -- what external guidance alone buys
    m1            m1-trained student, no teacher -- internalized guidance alone
    m1_teacher    m1-trained student + live teacher -- internalized + external

5 repetitions x different seeds per arm = 15 runs, scheduled across the given GPUs
by a worker pool (one worker per GPU, longest jobs queued first). A sampler thread
records nvidia-smi memory/utilization for the whole experiment. After all runs, the
post-hoc judge scores every final answer of every run with the teacher model.

Artifacts under runs/<ts>_exp/:
    <arm>_s<seed>/          episodes.jsonl + metrics.json per run (agent-created)
    gpu_samples.csv         15s nvidia-smi samples (ts, gpu, mem MiB, util %)
    jobs.jsonl              one line per finished job (rc, wall time, gpu)
    judge/                  verdicts.jsonl + summary.json across all 15 runs
    experiment.json         config + timing of the whole experiment

Usage:
    .venv_train/bin/python training_methods/exp_unseen100/run_experiment.py \
        --gpus 1,2,3,5,6,7 [--seeds 11,23,37,53,71] [--limit N] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402

PY = str(REPO_ROOT / ".venv_train" / "bin" / "python")
EXP = Path(__file__).parent
QUESTIONS = EXP / "data" / "unseen100" / "fresh_questions.jsonl"
CORPUS = EXP / "data" / "unseen100" / "fresh_corpus.jsonl"
ADAPTER_DEFAULT = REPO_ROOT / "training_methods/m1_sft/runs/20260712T022817Z_train4gpu/adapter"

ARMS = ("base_teacher", "m1", "m1_teacher")


def build_job(arm: str, seed: int, run_dir: Path, args) -> dict:
    """One eval run = one subprocess command; the agent creates <run_dir>/<ts>_<tag>."""
    tag = f"{arm}_s{seed}"
    common = [
        "--questions", str(QUESTIONS), "--corpus", str(CORPUS),
        "--out", str(run_dir), "--tag", tag, "--budget", str(args.budget),
        "--seed", str(seed),
    ]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if arm == "m1":
        cmd = [PY, str(REPO_ROOT / "training_methods/common/eval_agent.py"),
               "--adapter", str(args.adapter),
               "--temperature", str(args.student_temperature),
               "--batch-size", str(args.student_batch), *common]
        est = 1  # relative cost rank: teacherless is the fast one
    else:
        cmd = [PY, str(REPO_ROOT / "training_methods/common/teacher_eval_agent.py"),
               "--student-temperature", str(args.student_temperature),
               "--concurrency", str(args.teacher_concurrency), *common]
        if arm == "m1_teacher":
            cmd += ["--adapter", str(args.adapter)]
        est = 3  # teacher arms are API-latency bound: schedule first
    return {"arm": arm, "seed": seed, "tag": tag, "cmd": cmd, "cost_rank": est}


def gpu_sampler(path: Path, stop: threading.Event, interval_s: float = 15.0) -> None:
    with open(path, "w") as w:
        w.write("ts_utc,gpu,mem_used_mib,util_pct\n")
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                for line in out.strip().splitlines():
                    idx, mem, util = (x.strip() for x in line.split(","))
                    w.write(f"{ts},{idx},{mem},{util}\n")
                w.flush()
            except Exception:  # noqa: BLE001 -- sampling must never kill the run
                pass
            stop.wait(interval_s)


def worker(gpu: str, jobs: "queue.Queue[dict]", results: list, run_dir: Path, log, lock) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        log_file = run_dir / f"job_{job['tag']}.log"
        log.info(f"[gpu {gpu}] START {job['tag']}")
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                job["cmd"], stdout=lf, stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu},
                cwd=str(REPO_ROOT),
            )
        rec = {**{k: job[k] for k in ("arm", "seed", "tag")},
               "gpu": gpu, "rc": proc.returncode, "wall_s": round(time.time() - t0, 1)}
        with lock:
            results.append(rec)
            with open(run_dir / "jobs.jsonl", "a") as w:
                w.write(json.dumps(rec) + "\n")
        lvl = log.info if proc.returncode == 0 else log.error
        lvl(f"[gpu {gpu}] DONE {job['tag']} rc={proc.returncode} in {rec['wall_s']}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="1,2,3,5,6,7")
    ap.add_argument("--seeds", default="11,23,37,53,71")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--student-temperature", type=float, default=0.2)
    ap.add_argument("--student-batch", type=int, default=16)
    ap.add_argument("--teacher-concurrency", type=int, default=5)
    ap.add_argument("--judge-concurrency", type=int, default=10)
    ap.add_argument("--adapter", default=str(ADAPTER_DEFAULT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-base", default=str(EXP / "runs"))
    ap.add_argument("--smoke", action="store_true",
                    help="2 questions, 2 seeds, budget 2 — validates the whole flow")
    args = ap.parse_args()
    if args.smoke:
        args.limit, args.seeds, args.budget = 2, "11,23", 2
        args.teacher_concurrency = 2

    run_dir = timestamped_dir(args.out_base, "exp" + ("_smoke" if args.smoke else ""))
    log = setup_logger("exp_unseen100", run_dir / "experiment.log")
    seeds = [int(s) for s in args.seeds.split(",")]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log.info(f"artifacts: {run_dir}")
    log.info(f"gpus={gpus} seeds={seeds} limit={args.limit} budget={args.budget}")

    jobs = [build_job(arm, seed, run_dir, args) for arm in ARMS for seed in seeds]
    jobs.sort(key=lambda j: -j["cost_rank"])  # teacher jobs first for better packing
    q: "queue.Queue[dict]" = queue.Queue()
    for j in jobs:
        q.put(j)
    write_json(run_dir / "experiment.json", {
        "config": vars(args), "gpus": gpus, "seeds": seeds,
        "n_jobs": len(jobs), "jobs": [{k: j[k] for k in ("arm", "seed", "tag")} for j in jobs],
        "questions": str(QUESTIONS), "adapter": args.adapter,
    })

    stop = threading.Event()
    sampler = threading.Thread(
        target=gpu_sampler, args=(run_dir / "gpu_samples.csv", stop), daemon=True)
    sampler.start()

    t0 = time.time()
    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, q, results, run_dir, log, lock))
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    failed = [r for r in results if r["rc"] != 0]
    log.info(f"all {len(results)} jobs done in {(time.time()-t0)/60:.1f} min; failed: {len(failed)}")

    # post-hoc teacher verdict on every final answer of every run
    episode_files = sorted(str(p) for p in run_dir.glob("*/episodes.jsonl"))
    log.info(f"judging {len(episode_files)} runs")
    judge = subprocess.run(
        [PY, str(REPO_ROOT / "training_methods/common/judge_final_answers.py"),
         "--out", str(run_dir / "judge"), "--concurrency", str(args.judge_concurrency),
         *episode_files],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    (run_dir / "judge_console.log").write_text(judge.stdout + "\n" + judge.stderr)
    log.info(f"judge rc={judge.returncode}")

    write_json(run_dir / "experiment.json", {
        "config": vars(args), "gpus": gpus, "seeds": seeds,
        "n_jobs": len(results), "failed_jobs": failed,
        "total_wall_min": round((time.time() - t0) / 60, 1),
        "judge_rc": judge.returncode,
    })
    status = "COMPLETE" if not failed and judge.returncode == 0 else "COMPLETE WITH FAILURES"
    log.info(f"=== exp_unseen100 {status} — artifacts: {run_dir} ===")
    print(f"=== exp_unseen100 {status} — artifacts: {run_dir} ===")


if __name__ == "__main__":
    main()
