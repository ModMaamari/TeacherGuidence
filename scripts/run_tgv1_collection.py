"""Run the tg_v1 teacher-guidance collection: N datasets x M shards, resumable.

`agentsim simulate` walks one question file serially, so a 2,000-episode dataset is a
20-hour queue behind a gateway that comfortably serves 64 concurrent calls. This splits
each dataset into round-robin shards, each with its own question file and template, and
runs them concurrently -- the same pattern as run_batched_parallel.py, minus the GPU
plumbing, because both models here are API-served.

Round-robin (not contiguous) sharding matters: questions arrive in dataset order, so
contiguous blocks would give one worker all the easy early questions and skew any
per-shard statistic someone later computes.

Resumable: a shard whose episode file already holds all its questions is skipped, so
re-running after an interruption continues rather than restarts.

Usage::

    python scripts/run_tgv1_collection.py --datasets hotpotqa musique --shards 6
    python scripts/run_tgv1_collection.py --all --shards 6 --num-samples 2000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_tgv1_templates import DATASETS, build  # noqa: E402

SHARD_ROOT = REPO_ROOT / "data" / "datasets" / "tg_v1_shards"
OUT_ROOT = "./data/simulation_output/tgv1"
TEMPLATE_DIR = REPO_ROOT / "templates" / "simulations"


def question_lines(dataset: str, limit: int) -> List[str]:
    d, stem = DATASETS[dataset]
    path = REPO_ROOT / "data" / "datasets" / "tg_v1" / d / f"{stem}_questions.jsonl"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:limit]


def episodes_done(out_dir: Path) -> int:
    """Episodes already exported under a shard's output dir."""
    n = 0
    for p in out_dir.rglob("teacher_guidance_episodes.jsonl"):
        n += sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    return n


def prepare_shard(dataset: str, shard: int, n_shards: int, lines: List[str]) -> Dict:
    """Write one shard's question file and template; return its plan entry."""
    d, stem = DATASETS[dataset]
    shard_lines = lines[shard::n_shards]          # round-robin
    shard_dir = SHARD_ROOT / dataset
    shard_dir.mkdir(parents=True, exist_ok=True)
    qpath = shard_dir / f"{stem}_s{shard}.jsonl"
    qpath.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")

    tid = f"tgv1_{dataset}_s{shard}"
    tpl = build(dataset, len(shard_lines), tid, OUT_ROOT)
    tpl["datasets"][0]["path"] = "./" + str(qpath.relative_to(REPO_ROOT))
    (TEMPLATE_DIR / f"{tid}.yaml").write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return {"dataset": dataset, "shard": shard, "template": tid,
            "questions": len(shard_lines),
            "out_dir": REPO_ROOT / "data" / "simulation_output" / "tgv1" / tid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, choices=sorted(DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--shards", type=int, default=6, help="concurrent workers per dataset")
    ap.add_argument("--plan-only", action="store_true", help="write shards/templates, run nothing")
    ap.add_argument("--fau-timeout", default="600")
    args = ap.parse_args()

    datasets = sorted(DATASETS) if args.all or not args.datasets else args.datasets
    plans: List[Dict] = []
    for ds in datasets:
        lines = question_lines(ds, args.num_samples)
        for s in range(args.shards):
            plans.append(prepare_shard(ds, s, args.shards, lines))

    total_q = sum(p["questions"] for p in plans)
    print(f"planned {len(plans)} shards over {len(datasets)} dataset(s), {total_q} episodes")
    for p in plans:
        print(f"  {p['template']:<34} {p['questions']:>5} questions")
    if args.plan_only:
        return 0

    procs = []
    log_dir = REPO_ROOT / "data" / "simulation_output" / "tgv1" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env_extra = {"FAU_TIMEOUT": args.fau_timeout}
    for p in plans:
        done = episodes_done(p["out_dir"]) if p["out_dir"].exists() else 0
        if done >= p["questions"]:
            print(f"  skip {p['template']} (already {done}/{p['questions']})")
            continue
        log = (log_dir / f"{p['template']}.log").open("w", encoding="utf-8")
        import os
        env = {**os.environ, **env_extra}
        procs.append((p, subprocess.Popen(
            [sys.executable, "-m", "agentsim.cli", "simulate", p["template"]],
            cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print(f"launched {len(procs)} shard workers")

    t0 = time.time()
    failed = []
    for p, proc, log in procs:
        rc = proc.wait()
        log.close()
        got = episodes_done(p["out_dir"])
        status = "ok" if rc == 0 and got >= p["questions"] else "FAILED"
        if status != "ok":
            failed.append(p["template"])
        print(f"  [{status}] {p['template']} rc={rc} episodes={got}/{p['questions']}")
    print(f"all shards finished in {(time.time()-t0)/60:.1f} min; failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
