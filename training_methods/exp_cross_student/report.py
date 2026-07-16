"""exp_cross_student report: does a student learn better from ITS OWN traces?

Aggregates everything the experiment produced into one JSON + printed summary:

  * end2end accuracy per (model x test set): deterministic cover-match (the metric of
    record), EM / F1 / doc-recall, and the post-hoc teacher-judge verdict; mean +/- sd
    over seeds, plus correct-half / wrong-half breakdown of each test set;
  * efficiency: steps used, tokens per episode, episode wall time;
  * training: examples, wall time, final train/eval loss, tokens seen;
  * hardware: GPU memory / utilization during training and evals (from the samplers).

Usage:
    .venv/bin/python training_methods/exp_cross_student/report.py \
        [--evals-dir <runs/<ts>_evals>] [--manifest <runs/<ts>_manifest.env>]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EXP = Path(__file__).parent
MODELS = ("base", "m1_self", "m1_cross")
TESTSETS = ("qwen", "granite")


def _latest(pattern: str) -> Path:
    hits = sorted(EXP.glob(pattern))
    assert hits, f"nothing matches {pattern}"
    return hits[-1]


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def load_test_halves() -> dict:
    """testset -> qid -> 'correct'|'wrong' (which half of the test set it came from)."""
    halves = {}
    for ts in TESTSETS:
        sel = json.loads((EXP / "data" / ts / "test_qids.json").read_text())
        halves[ts] = {q: "correct" for q in sel["correct"]} | {q: "wrong" for q in sel["wrong"]}
    return halves


def collect_eval_runs(evals_dir: Path, judge: dict, halves: dict) -> dict:
    """model -> testset -> list of per-seed run summaries."""
    out: dict = {m: {t: [] for t in TESTSETS} for m in MODELS}
    for m in MODELS:
        for run_dir in sorted((evals_dir / m).glob("*_*")):
            epf = run_dir / "episodes.jsonl"
            mf = run_dir / "metrics.json"
            if not epf.exists() or not mf.exists():
                continue
            tag = run_dir.name.split("_", 1)[1]          # <model>_<testset>test_s<seed>
            testset = "qwen" if "_qwentest_" in f"_{tag}_" else "granite"
            seed = tag.rsplit("_s", 1)[-1]
            metrics = json.loads(mf.read_text())
            eps = [json.loads(l) for l in epf.read_text().splitlines() if l.strip()]

            by_half = {"correct": [], "wrong": []}
            ep_wall = []
            judged_ok = judged_n = 0
            for ep in eps:
                fm = ep.get("final_metrics") or {}
                half = halves[testset].get(str(ep.get("qid")))
                if half:
                    by_half[half].append(1 if fm.get("cover_match") else 0)
                ep_wall.append(ep.get("elapsed_s"))
                v = judge.get((str(ep.get("qid")), str(epf.resolve())))
                if v is not None:
                    judged_n += 1
                    judged_ok += int(v)
            row = {
                "seed": seed,
                "n": len(eps),
                "cover_match": metrics.get("cover_match"),
                "exact_match": metrics.get("em"),
                "f1": metrics.get("f1"),
                "doc_recall": metrics.get("doc_recall"),
                "cover_on_correct_half": (sum(by_half["correct"]) / len(by_half["correct"])) if by_half["correct"] else None,
                "cover_on_wrong_half": (sum(by_half["wrong"]) / len(by_half["wrong"])) if by_half["wrong"] else None,
                "judge_correct_rate": (judged_ok / judged_n) if judged_n else None,
                "avg_steps": metrics.get("mean_steps"),
                "avg_completion_tokens": metrics.get("tokens_per_episode"),
                "mean_episode_s": statistics.mean([w for w in ep_wall if w is not None]) if any(w is not None for w in ep_wall) else None,
                "wall_s_total": metrics.get("wall_time_s"),
                "stop_reasons": metrics.get("stop_reasons"),
                "invalid_action_steps": metrics.get("invalid_action_steps"),
            }
            out[m][testset].append(row)
    return out


def load_judge(evals_dir: Path) -> dict:
    """(qid, episodes-file path) -> correct 0/1 from the post-hoc judge verdicts."""
    verdicts = {}
    vf = evals_dir / "judge" / "verdicts.jsonl"
    if not vf.exists():
        return verdicts
    for l in vf.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        v = r.get("verdict") or {}
        if v.get("correct") is not None:
            verdicts[(str(r.get("qid")), str(r.get("source", "")))] = int(v["correct"])
    return verdicts


def load_training(manifest: Path) -> dict:
    kv = dict(l.split("=", 1) for l in manifest.read_text().splitlines() if "=" in l)
    out = {}
    for name, key in (("m1_self", "SELF_RUN"), ("m1_cross", "CROSS_RUN")):
        rd = REPO_ROOT / kv[key] if not Path(kv[key]).is_absolute() else Path(kv[key])
        fm = json.loads((rd / "final_metrics.json").read_text()) if (rd / "final_metrics.json").exists() else {}
        cfg = json.loads((rd / "train_config.json").read_text()) if (rd / "train_config.json").exists() else {}
        tf = (cfg.get("args") or {}).get("train_file", "")
        n_train = sum(1 for l in open(REPO_ROOT / tf)) if tf and (REPO_ROOT / tf).exists() else None
        out[name] = {
            "run_dir": str(rd),
            "train_file": tf,
            "train_examples": n_train,
            "train_runtime_s": fm.get("train_runtime"),
            "train_loss": fm.get("train_loss"),
            "eval_loss": fm.get("eval_loss"),
            "train_samples_per_s": fm.get("train_samples_per_second"),
            "total_flos": fm.get("total_flos"),
        }
    return out


def gpu_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    by_gpu = collections.defaultdict(lambda: {"mem": [], "util": []})
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            by_gpu[row["gpu"]]["mem"].append(int(row["mem_used_mib"]))
            by_gpu[row["gpu"]]["util"].append(int(row["util_pct"]))
    return {
        g: {
            "samples": len(d["mem"]),
            "mem_mib_peak": max(d["mem"]),
            "mem_mib_mean": round(statistics.mean(d["mem"])),
            "util_pct_mean": round(statistics.mean(d["util"]), 1),
        }
        for g, d in sorted(by_gpu.items()) if max(d["mem"]) > 1000  # skip idle GPUs
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals-dir", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    evals_dir = Path(args.evals_dir) if args.evals_dir else _latest("runs/*_evals")
    manifest = Path(args.manifest) if args.manifest else _latest("runs/*_manifest.env")
    build_stats = json.loads((EXP / "data" / "build_stats.json").read_text())

    halves = load_test_halves()
    judge = load_judge(evals_dir)
    runs = collect_eval_runs(evals_dir, judge, halves)
    training = load_training(manifest)

    # aggregate over seeds
    agg: dict = {}
    for m in MODELS:
        agg[m] = {}
        for t in TESTSETS:
            rows = runs[m][t]
            if not rows:
                continue
            entry = {"seeds": len(rows), "n_per_seed": rows[0]["n"]}
            for k in ("cover_match", "exact_match", "f1", "doc_recall",
                      "cover_on_correct_half", "cover_on_wrong_half",
                      "judge_correct_rate", "avg_steps", "avg_completion_tokens"):
                mean, sd = _mean_sd([r[k] for r in rows])
                if mean is not None:
                    entry[k] = {"mean": round(mean, 4), "sd": round(sd, 4)}
            agg[m][t] = entry

    ts_gpu = sorted(EXP.glob("runs/*_gpu_samples.csv"))
    report = {
        "datasets": build_stats,
        "training": training,
        "eval_aggregate": agg,
        "eval_per_run": runs,
        "gpu_training": gpu_summary(ts_gpu[-1]) if ts_gpu else {},
        "gpu_evals": gpu_summary(evals_dir / "gpu_samples.csv"),
        "evals_dir": str(evals_dir),
        "manifest": str(manifest),
    }
    out = evals_dir / "report.json"
    out.write_text(json.dumps(report, indent=2))

    # -------- printed summary --------
    print("=" * 78)
    print("exp_cross_student: Qwen3.5-0.8B trained on SELF vs CROSS (granite) data")
    print("=" * 78)
    for m in MODELS:
        tr = training.get(m, {})
        extra = (f" | trained on {tr.get('train_examples')} ex in "
                 f"{(tr.get('train_runtime_s') or 0)/60:.0f} min, "
                 f"loss {tr.get('train_loss'):.3f}" if tr.get("train_loss") else "")
        print(f"\n--- {m}{extra} ---")
        for t in TESTSETS:
            e = agg.get(m, {}).get(t)
            if not e:
                print(f"  {t}-test: (no runs)")
                continue
            print(f"  {t}-test  cover={e['cover_match']['mean']:.3f}±{e['cover_match']['sd']:.3f}"
                  f"  f1={e['f1']['mean']:.3f}"
                  f"  correct-half={e['cover_on_correct_half']['mean']:.3f}"
                  f"  wrong-half={e['cover_on_wrong_half']['mean']:.3f}"
                  + (f"  judge={e['judge_correct_rate']['mean']:.3f}" if "judge_correct_rate" in e else "")
                  + f"  steps={e['avg_steps']['mean']:.2f}")
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
