"""m1-LODO orchestrator: train 4 leave-one-dataset-out folds, then evaluate 24 runs.

Design notes that matter:

**Why a job queue rather than a script per fold.** Training and evaluation are both
embarrassingly parallel across (fold, test set), but a 3B LoRA uses ~20 GB of a 141 GB
H200 -- one job per GPU would waste 85% of the card. Jobs are therefore placed on GPU
*slots* (several per GPU) by a worker pool, and the pool is the only thing that knows how
many fit.

**Why vLLM for eval and HF for training.** A single vLLM server per GPU loads the base
model once and serves every fold adapter by name (`--enable-lora`), continuously batching
all concurrent episodes. Training needs the HF stack, so the two live in separate venvs.

**Resumability.** Every job's completion is a file on disk (`adapter/` for train,
`metrics.json` for eval). A re-run skips whatever already exists, so an interrupted run
continues rather than restarting -- the failure mode that cost us hours during collection.

Usage::

    python training_methods/m1_lodo/run_lodo.py --stage all --gpus 0,1,2
    python training_methods/m1_lodo/run_lodo.py --stage eval --gpus 0,1,2 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
CORPUS = {
    "hotpotqa": "data/datasets/tg_v1/hotpotqa/hotpotqa_train_corpus.jsonl",
    "2wikimultihopqa": "data/datasets/tg_v1/2wikimultihopqa/2wikimultihopqa_train_corpus.jsonl",
    "musique": "data/datasets/tg_v1/musique/musique_train_corpus.jsonl",
    "strategyqa": "data/datasets/tg_v1/strategyqa/strategyqa_train_corpus.jsonl",
}
VENV_TRAIN = REPO / ".venv_train/bin/python"
VENV_VLLM = REPO / ".venv_vllm/bin/python"
HF_HOME = str(REPO / ".hf_cache")
#: Frigga compute nodes cannot write to /home, and vLLM's engine treats a failed cache
#: mkdir as fatal (not just the model-info cache -- torch.compile and NCCL dirs too).
#: Every cache root is therefore pinned onto /shared.
CACHE_ROOT = str(REPO / ".cache")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, path: Optional[Path] = None) -> None:
    line = f"{utc()} | {msg}"
    print(line, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def env_for(gpu: str) -> Dict[str, str]:
    e = dict(os.environ)
    e.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HOME": HF_HOME, "HF_HUB_CACHE": HF_HOME + "/hub",
        "TOKENIZERS_PARALLELISM": "false",
        "HOME": CACHE_ROOT,                 # anything that still resolves ~ lands here
        "XDG_CACHE_HOME": CACHE_ROOT,
        "VLLM_CACHE_ROOT": CACHE_ROOT + "/vllm",
        "TRITON_CACHE_DIR": CACHE_ROOT + "/triton",
        "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT + "/inductor",
    })
    for d in (CACHE_ROOT, CACHE_ROOT + "/vllm", CACHE_ROOT + "/triton",
              CACHE_ROOT + "/inductor"):
        Path(d).mkdir(parents=True, exist_ok=True)
    return e


# --------------------------------------------------------------------------- train
def train_jobs(out: Path, args) -> List[Dict[str, Any]]:
    jobs = []
    for ds in DATASETS:
        run_dir = out / "train" / f"fold_{ds}"
        jobs.append({
            "kind": "train", "name": f"train:{ds}", "fold": ds,
            "done_marker": run_dir / "adapter" / "adapter_config.json",
            "cmd": [str(VENV_TRAIN), "training_methods/m1_sft/train.py",
                    "--model", args.model,
                    "--train-file", str(DATA / f"fold_{ds}" / "train.jsonl"),
                    "--dev-file", str(DATA / f"fold_{ds}" / "dev.jsonl"),
                    "--run-dir", str(run_dir),
                    "--epochs", str(args.epochs), "--lr", str(args.lr),
                    "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
                    "--lora-r", str(args.lora_r), "--lora-alpha", str(args.lora_alpha),
                    "--seed", str(args.seed)] + (["--smoke"] if args.smoke else []),
            "log": out / "logs" / f"train_{ds}.log",
        })
    return jobs


# ---------------------------------------------------------------------------- eval
def eval_matrix(args) -> List[Dict[str, str]]:
    """(model, test_set) pairs: each fold on its own 4 sets, plus base on all 8."""
    rows = []
    for fold in DATASETS:
        for other in [d for d in DATASETS if d != fold]:
            rows.append({"model": f"fold_{fold}", "test": f"heldin_{other}", "dataset": other})
        rows.append({"model": f"fold_{fold}", "test": f"unseen_{fold}", "dataset": fold})
    for ds in DATASETS:
        rows.append({"model": "base", "test": f"heldin_{ds}", "dataset": ds})
        rows.append({"model": "base", "test": f"unseen_{ds}", "dataset": ds})
    return rows


def eval_jobs(out: Path, args, servers: Dict[str, str]) -> List[Dict[str, Any]]:
    jobs = []
    for row in eval_matrix(args):
        model, test, ds = row["model"], row["test"], row["dataset"]
        qfile = DATA / "tests" / f"{test}_questions.jsonl"
        n_shards = args.unseen_shards if test.startswith("unseen") else 1
        for shard in range(n_shards):
            tag = f"{model}__{test}" + (f"__s{shard}" if n_shards > 1 else "")
            run_dir = out / "eval" / tag
            served = "student" if model == "base" else model
            cmd = [str(VENV_TRAIN), "training_methods/common/eval_agent.py",
                   "--model", args.model,
                   "--questions", str(qfile), "--corpus", str(REPO / CORPUS[ds]),
                   "--out", str(out / "eval"), "--tag", tag,
                   "--budget", str(args.budget), "--hidden-budget",
                   "--temperature", "0", "--seed", str(args.seed),
                   "--backend", "vllm", "--served-model", served,
                   "--batch-size", str(args.eval_batch)]
            if n_shards > 1:
                cmd += ["--shard", f"{shard}/{n_shards}"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            jobs.append({"kind": "eval", "name": tag, "model": model, "test": test,
                         "dataset": ds, "shard": shard, "n_shards": n_shards,
                         "cmd": cmd, "run_dir": run_dir,
                         "done_marker": out / "eval" / f"{tag}.done",
                         "log": out / "logs" / f"eval_{tag}.log"})
    return jobs


# ------------------------------------------------------------------------- servers
def start_servers(out: Path, gpus: List[str], args) -> List[Dict[str, Any]]:
    """One vLLM server per GPU, each serving the base model plus every fold adapter."""
    servers = []
    for i, gpu in enumerate(gpus):
        port = args.vllm_base_port + i
        adapters = []
        for ds in DATASETS:
            a = out / "train" / f"fold_{ds}" / "adapter"
            if a.exists():
                adapters += [f"fold_{ds}={a}"]
        cmd = [str(VENV_VLLM.parent / "vllm"), "serve", args.model,
               "--served-model-name", "student", "--port", str(port),
               "--gpu-memory-utilization", str(args.vllm_mem_util),
               "--max-model-len", str(args.max_model_len), "--dtype", "bfloat16"]
        if adapters:
            cmd += ["--enable-lora", "--max-lora-rank", str(args.lora_r),
                    "--max-loras", str(len(adapters)), "--lora-modules"] + adapters
        lg = (out / "logs" / f"vllm_gpu{gpu}_{port}.log").open("w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env_for(gpu), stdout=lg, stderr=subprocess.STDOUT)
        servers.append({"gpu": gpu, "port": port, "proc": p, "log": lg,
                        "url": f"http://127.0.0.1:{port}"})
    return servers


def wait_servers(servers: List[Dict[str, Any]], out: Path,
                 timeout: int = 5400) -> List[Dict[str, Any]]:
    """Poll every server round-robin until all are serving.

    Returns the servers that became ready. A GPU allocation is scarce and was already lost
    once to this function raising: if two of three servers are serving, the right move is to
    run on two and log the third as degraded, not to throw away the whole allocation. Only a
    complete failure (nothing serving) is fatal.

    Servers are polled together rather than one after another: they start simultaneously
    and contend for CPU and the shared filesystem while loading weights and capturing CUDA
    graphs, so a sequential wait spends the whole budget on whichever happens to be slowest
    and reports nothing about the others. Startup was measured at 745s for two concurrent
    servers and longer for three, so the budget is generous by design -- being slow to
    start is normal, and killing a healthy run for it is not.
    """
    import urllib.request
    t0 = time.time()
    pending = {s["port"]: s for s in servers}
    while pending:
        for port, s in list(pending.items()):
            if s["proc"].poll() is not None:
                log(f"DEGRADED: vLLM on gpu {s['gpu']} exited rc={s['proc'].returncode}; "
                    f"see {out/'logs'}/vllm_gpu{s['gpu']}_{s['port']}.log", out / "run.log")
                del pending[port]
                continue
            try:
                urllib.request.urlopen(s["url"] + "/v1/models", timeout=5)
            except Exception:
                continue
            log(f"vllm ready gpu={s['gpu']} port={port} ({time.time()-t0:.0f}s)",
                out / "run.log")
            del pending[port]
        if not pending:
            break
        waited = time.time() - t0
        if waited > timeout:
            log("DEGRADED: giving up on gpu(s) %s after %ds; continuing with %d server(s)"
                % ([x["gpu"] for x in pending.values()], timeout, len(servers) - len(pending)),
                out / "run.log")
            for x in pending.values():
                try:
                    x["proc"].terminate()
                except Exception:
                    pass
            break
        if int(waited) % 120 < 6:
            log(f"waiting for {len(pending)} server(s) ({waited:.0f}s elapsed)",
                out / "run.log")
        time.sleep(6)
    ready = [x for x in servers if x["port"] not in pending]
    if not ready:
        raise SystemExit("no vLLM server became ready")
    return ready


# ------------------------------------------------------------------------ scheduler
def run_pool(jobs: List[Dict[str, Any]], slots: List[Dict[str, str]], out: Path,
             status: Dict[str, Any]) -> int:
    """Run jobs over a fixed set of slots (a slot pins CUDA device / server url)."""
    q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    todo = []
    for j in jobs:
        if j["done_marker"].exists():
            log(f"skip {j['name']} (already done)", out / "run.log")
            status["skipped"].append(j["name"])
            continue
        todo.append(j)
        q.put(j)
    log(f"{len(todo)} job(s) to run over {len(slots)} slot(s)", out / "run.log")
    failures: List[str] = []
    lock = threading.Lock()

    def worker(slot: Dict[str, str]) -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            j["log"].parent.mkdir(parents=True, exist_ok=True)
            cmd = list(j["cmd"])
            if j["kind"] == "eval":
                cmd += ["--server-url", slot["url"]]
            t0 = time.time()
            log(f"START {j['name']} on gpu {slot['gpu']}", out / "run.log")
            with j["log"].open("w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd, cwd=str(REPO), env=env_for(slot["gpu"]),
                                     stdout=lf, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            ok = rc == 0
            if ok and j["kind"] == "eval":
                j["done_marker"].write_text(utc(), encoding="utf-8")
            with lock:
                status["jobs"].append({"name": j["name"], "rc": rc, "seconds": round(dt, 1),
                                       "gpu": slot["gpu"], "finished": utc()})
                (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                if not ok:
                    failures.append(j["name"])
            log(f"{'DONE ' if ok else 'FAIL '}{j['name']} rc={rc} in {dt/60:.1f} min "
                f"(see {j['log']})", out / "run.log")

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        log(f"FAILURES: {failures}", out / "run.log")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--out", default=str(HERE / "runs" / "lodo"))
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--jobs-per-gpu", type=int, default=2, help="training jobs co-located per GPU")
    ap.add_argument("--clients-per-server", type=int, default=4, help="eval clients per vLLM server")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--unseen-shards", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vllm-base-port", type=int, default=8400)
    ap.add_argument("--vllm-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--eval-attempts", type=int, default=3,
                    help="in-allocation retries of the eval stage before giving up")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    status = {"started": utc(), "args": vars(args), "jobs": [], "skipped": []}
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    log(f"m1-LODO stage={args.stage} gpus={gpus} out={out}", out / "run.log")

    rc = 0
    if args.stage in ("train", "all"):
        slots = [{"gpu": g, "url": ""} for g in gpus for _ in range(args.jobs_per_gpu)]
        log(f"== TRAIN == {len(slots)} slots", out / "run.log")
        rc |= run_pool(train_jobs(out, args), slots, out, status)
        if rc:
            log("training had failures; not starting eval", out / "run.log")
            return rc

    if args.stage in ("eval", "all"):
        log("== EVAL == starting vLLM servers", out / "run.log")
        # Retry inside the allocation: a crashed attempt must not hand the GPUs back,
        # and every finished shard is skipped on the next pass, so a retry is cheap.
        for attempt in range(1, args.eval_attempts + 1):
            servers = start_servers(out, gpus, args)
            try:
                ready = wait_servers(servers, out)
                slots = [{"gpu": s["gpu"], "url": s["url"]}
                         for s in ready for _ in range(args.clients_per_server)]
                log(f"attempt {attempt}: {len(ready)} server(s), {len(slots)} client slots",
                    out / "run.log")
                rc = run_pool(eval_jobs(out, args, {}), slots, out, status)
            except SystemExit as exc:
                log(f"attempt {attempt} failed: {exc}", out / "run.log")
                rc = 1
            finally:
                for s in servers:
                    try:
                        s["proc"].terminate()
                    except Exception:
                        pass
                    s["log"].close()
                log("vLLM servers stopped", out / "run.log")
            if rc == 0:
                break
            if attempt < args.eval_attempts:
                log(f"retrying eval stage ({attempt+1}/{args.eval_attempts}) in 60s",
                    out / "run.log")
                time.sleep(60)

    status["finished"] = utc()
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    log(f"m1-LODO finished rc={rc}", out / "run.log")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
