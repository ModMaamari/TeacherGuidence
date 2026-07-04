"""
Build the final report + plots for the 4-model x 5-setting experiment matrix.

Reads the orchestrator's manifest.jsonl plus every run's teacher_guidance_episodes.jsonl
(field names match scripts/aggregate_teacher_guidance_run.py for consistency), computes
per-(model,setting) metrics, generates matplotlib plots, and writes REPORT.md with the
plots embedded inline.

Usage:
    python scripts/build_experiment_matrix_report.py \
        --manifest reports/experiment_matrix_2026-07-03/manifest.jsonl \
        --sim-output-root data/simulation_output/exp_matrix \
        --out-dir reports/experiment_matrix_2026-07-03
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gen_experiment_matrix_templates import MODELS, SETTINGS  # noqa: E402

SETTING_LABELS = {k: v["desc"] for k, v in SETTINGS.items()}


# ---------------------------------------------------------------------------
# Pure aggregation (unit-tested against synthetic episode dicts)
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _median(values: List[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _all_calls(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every call-log entry (student + teacher) across plan-review and all steps."""
    calls: List[Dict[str, Any]] = []
    pr = episode.get("plan_review") or {}
    calls.extend(pr.get("initial_plan_calls") or [])
    calls.extend(pr.get("plan_calls") or [])  # teacher-planner mode
    for round_rec in pr.get("rounds") or []:
        calls.extend(round_rec.get("review_calls") or [])
        calls.extend(round_rec.get("revision_calls") or [])
    for step in episode.get("steps") or []:
        calls.extend(step.get("student_calls") or [])
        calls.extend(step.get("teacher_calls") or [])
    return calls


def _teacher_call_count(episode: Dict[str, Any]) -> int:
    pr = episode.get("plan_review") or {}
    n = len(pr.get("plan_calls") or [])
    for round_rec in pr.get("rounds") or []:
        n += len(round_rec.get("review_calls") or [])
    for step in episode.get("steps") or []:
        n += len(step.get("teacher_calls") or [])
    return n


def aggregate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the full metric set for one (model, setting) group of episodes."""
    if not episodes:
        return {"num_episodes": 0}

    em = [1.0 if e.get("final_metrics", {}).get("exact_match") else 0.0 for e in episodes]
    correct = [
        1.0 if e.get("final_metrics", {}).get("answer_correct", e.get("final_metrics", {}).get("exact_match")) else 0.0
        for e in episodes
    ]
    f1 = [float(e.get("final_metrics", {}).get("f1", 0.0)) for e in episodes]
    doc_recall = [float(e.get("final_metrics", {}).get("supporting_doc_recall", 0.0)) for e in episodes]
    steps_taken = [len(e.get("steps", [])) for e in episodes]

    invalid_json_steps = 0
    total_steps = 0
    leak_episodes = 0
    total_teacher_calls = 0
    total_tokens = 0
    total_cost_usd = 0.0
    for e in episodes:
        leaked = False
        for s in e.get("steps", []):
            total_steps += 1
            if not s.get("metrics", {}).get("json_valid", True):
                invalid_json_steps += 1
            # Only a real gold-answer leak counts. Hidden title/doc/span mentions are
            # detected for telemetry but are fair game (question entities /
            # already-retrieved docs) and are no longer redacted, so they must not
            # inflate the leakage rate. See agentsim/teacher_guidance/leakage.py.
            if (s.get("leakage_check", {}) or {}).get("gold_answer_leaked"):
                leaked = True
        if leaked:
            leak_episodes += 1
        total_teacher_calls += _teacher_call_count(e)
        for call in _all_calls(e):
            usage = call.get("usage") or {}
            total_tokens += int(usage.get("total_tokens") or 0)
            total_cost_usd += float(usage.get("cost") or 0.0)

    return {
        "num_episodes": len(episodes),
        "exact_match": _mean(em),
        "answer_correct": _mean(correct),
        "f1": _mean(f1),
        "supporting_doc_recall": _mean(doc_recall),
        "avg_steps": _mean([float(s) for s in steps_taken]),
        "median_steps": _median([float(s) for s in steps_taken]),
        "min_steps": min(steps_taken) if steps_taken else 0,
        "max_steps": max(steps_taken) if steps_taken else 0,
        "invalid_json_rate": round(invalid_json_steps / total_steps, 4) if total_steps else 0.0,
        "leakage_rate": round(leak_episodes / len(episodes), 4),
        "total_teacher_calls": total_teacher_calls,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
    }


def step_count_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """min/max/mean/median steps-taken across a group of episodes (setting C stats)."""
    steps = [float(len(e.get("steps", []))) for e in episodes]
    if not steps:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(steps), "max": max(steps),
        "mean": _mean(steps), "median": _median(steps),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    rows = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_episodes(run_dir: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    if not run_dir.exists():
        return episodes
    for path in run_dir.rglob("teacher_guidance_episodes.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def collect_qids(sim_output_root: Path) -> List[str]:
    qids = set()
    for path in sim_output_root.rglob("teacher_guidance_episodes.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    qids.add(json.loads(line).get("qid"))
    return sorted(qids)


def build_results(sim_output_root: Path, manifest_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    wall_seconds_by_run = {
        (row["model_slug"], row["setting"]): row.get("wall_seconds") for row in manifest_rows
    }
    results: Dict[str, Dict[str, Any]] = {}
    for model_slug in MODELS:
        for setting_key in SETTINGS:
            run_dir = sim_output_root / f"{model_slug}_{setting_key}"
            episodes = load_episodes(run_dir)
            metrics = aggregate_episodes(episodes)
            metrics["wall_seconds"] = wall_seconds_by_run.get((model_slug, setting_key))
            results[f"{model_slug}_{setting_key}"] = metrics
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(results: Dict[str, Dict[str, Any]], sim_output_root: Path, plots_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    model_slugs = list(MODELS.keys())
    setting_keys = list(SETTINGS.keys())
    written = []

    def _grouped_bar(metric_key: str, title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.15
        x = range(len(setting_keys))
        for i, model_slug in enumerate(model_slugs):
            values = [results[f"{model_slug}_{s}"].get(metric_key) or 0 for s in setting_keys]
            positions = [xi + i * width for xi in x]
            ax.bar(positions, values, width=width, label=model_slug)
        ax.set_xticks([xi + width * (len(model_slugs) - 1) / 2 for xi in x])
        ax.set_xticklabels(setting_keys)
        ax.set_xlabel("Setting")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(filename)

    _grouped_bar("answer_correct", "Answer-correct rate by model x setting", "Answer correct (mean)", "answer_correct.png")
    _grouped_bar("f1", "F1 by model x setting", "F1 (mean)", "f1.png")
    _grouped_bar("avg_steps", "Average steps-to-finish by model x setting", "Avg steps", "avg_steps.png")
    _grouped_bar("total_cost_usd", "Total teacher (OpenRouter) cost by model x setting", "Cost (USD)", "cost.png")

    # Setting C step-count distribution across models.
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = []
    for model_slug in model_slugs:
        run_dir = sim_output_root / f"{model_slug}_C"
        episodes = load_episodes(run_dir)
        box_data.append([len(e.get("steps", [])) for e in episodes] or [0])
    ax.boxplot(box_data, tick_labels=model_slugs)
    ax.axhline(5, color="grey", linestyle="--", linewidth=1, label="displayed budget (5)")
    ax.set_ylabel("Steps taken")
    ax.set_title("Setting C: steps taken vs. hidden 20-step cap (displayed budget stayed at 5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "setting_c_steps.png", dpi=120)
    plt.close(fig)
    written.append("setting_c_steps.png")

    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_report(
    out_dir: Path,
    results: Dict[str, Dict[str, Any]],
    plot_files: List[str],
    qids: List[str],
    sim_output_root: Path,
) -> Path:
    model_slugs = list(MODELS.keys())
    setting_keys = list(SETTINGS.keys())

    lines: List[str] = []
    lines.append("# Teacher Guidance experiment matrix report")
    lines.append("")
    lines.append(
        "4 student models x 5 teacher-guidance settings x 30 fixed random HotpotQA questions "
        "(teacher: `custom/z-ai/glm-5.2`)."
    )
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    for key in setting_keys:
        lines.append(f"- **{key}** -- {SETTING_LABELS[key]}")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    for slug, model_id in MODELS.items():
        lines.append(f"- `{slug}` -- `{model_id}`")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    header = "| Model | Setting | Episodes | Answer correct | EM | F1 | Doc recall | Avg steps | Median steps | Teacher calls | Tokens | Cost (USD) | Wall (s) |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for model_slug in model_slugs:
        for setting_key in setting_keys:
            m = results[f"{model_slug}_{setting_key}"]
            lines.append(
                f"| {model_slug} | {setting_key} | {_fmt(m.get('num_episodes'))} | "
                f"{_fmt(m.get('answer_correct'))} | {_fmt(m.get('exact_match'))} | {_fmt(m.get('f1'))} | "
                f"{_fmt(m.get('supporting_doc_recall'))} | {_fmt(m.get('avg_steps'))} | "
                f"{_fmt(m.get('median_steps'))} | {_fmt(m.get('total_teacher_calls'))} | "
                f"{_fmt(m.get('total_tokens'))} | {_fmt(m.get('total_cost_usd'))} | {_fmt(m.get('wall_seconds'))} |"
            )
    lines.append("")

    lines.append("## Setting C: steps-taken distribution (hidden 20-step cap)")
    lines.append("")
    lines.append("| Model | Min | Max | Mean | Median |")
    lines.append("|---|---|---|---|---|")
    for model_slug in model_slugs:
        run_dir = sim_output_root / f"{model_slug}_C"
        dist = step_count_distribution(load_episodes(run_dir))
        lines.append(f"| {model_slug} | {_fmt(dist['min'])} | {_fmt(dist['max'])} | {_fmt(dist['mean'])} | {_fmt(dist['median'])} |")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    for filename in plot_files:
        lines.append(f"![{filename}](plots/{filename})")
        lines.append("")

    total_cost = sum((m.get("total_cost_usd") or 0.0) for m in results.values())
    total_tokens = sum((m.get("total_tokens") or 0) for m in results.values())
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Total OpenRouter (teacher) cost: **${total_cost:.4f}**")
    lines.append(f"- Total tokens (student + teacher, all calls): **{total_tokens:,}**")
    lines.append(f"- Question set: {len(qids)} fixed HotpotQA qids (seed 2026): `{', '.join(qids)}`")
    lines.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/experiment_matrix_2026-07-03/manifest.jsonl")
    parser.add_argument("--sim-output-root", default="data/simulation_output/exp_matrix")
    parser.add_argument("--out-dir", default="reports/experiment_matrix_2026-07-03")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sim_output_root = Path(args.sim_output_root)
    manifest_rows = load_manifest(Path(args.manifest))

    results = build_results(sim_output_root, manifest_rows)
    plot_files = make_plots(results, sim_output_root, out_dir / "plots")
    qids = collect_qids(sim_output_root)
    report_path = write_report(out_dir, results, plot_files, qids, sim_output_root)
    print(f"Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
