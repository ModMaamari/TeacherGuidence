"""
best_answer_score: rank the same question's answers across trace-collection runs.

An episode's answer is *eligible* only when the teacher judged it correct
(``teacher_answer_score >= TEACHER_CORRECT_THRESHOLD``); ineligible episodes score 0.
Eligible episodes get a weighted sum of six [0, 1] components:

    steps        fewer used steps is better, normalized against the global 9-step budget
    f1           token-level F1 of final vs gold answer
    doc_recall   supporting_doc_recall (how much of the gold evidence was retrieved)
    exact_match  1.0 when EM
    cover_match  1.0 when cover-match answer_correct AND the teacher score is a full 1.0
    length       min/max length ratio of final vs gold answer (1.0 = same length)

Weights sum to 1, so the score lives in [0, 1] and is comparable across runs.

CLI: compute the 3000 x 6 score matrix over the six granite trace collections, write
per-run stats and plots.

    python scripts/best_answer_score.py --out-dir reports/best_answer_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

TEACHER_CORRECT_THRESHOLD = 0.40
GLOBAL_MAX_BUDGET = 9  # largest running-step budget across the compared runs

WEIGHTS = {
    "steps": 0.25,
    "f1": 0.20,
    "doc_recall": 0.15,
    "exact_match": 0.15,
    "cover_match": 0.10,
    "length": 0.15,
}

# run label -> consolidated run dir (all six collections share the same 3000 train qids)
DEFAULT_RUNS = {
    "b1": "data/simulation_output/traces_g3b_b1_3000/run",
    "b2": "data/simulation_output/traces_g3b_b2_3000/run",
    "b3": "data/simulation_output/traces_g3b_b3_3000/run",
    "b4": "data/simulation_output/traces_g3b_b4_3000/run",
    "b5h": "data/simulation_output/traces_g3b_b5_3000/run",
    "b9h": "data/simulation_output/traces_g3b_3000/run",
}

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"


def _length_similarity(final_answer: str, gold_answer: str) -> float:
    a, b = len(final_answer.strip()), len(gold_answer.strip())
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def score_components(episode: Dict[str, Any]) -> Dict[str, float]:
    """The six [0, 1] components, regardless of eligibility."""
    m = episode.get("final_metrics", {}) or {}
    used = episode.get("used_steps") or len(episode.get("steps", []) or [])
    teacher = float(m.get("teacher_answer_score") or 0.0)
    return {
        "steps": max(0.0, min(1.0, (GLOBAL_MAX_BUDGET - used) / (GLOBAL_MAX_BUDGET - 1))),
        "f1": float(m.get("f1") or 0.0),
        "doc_recall": float(m.get("supporting_doc_recall") or 0.0),
        "exact_match": 1.0 if m.get("exact_match") else 0.0,
        "cover_match": 1.0 if (m.get("answer_correct") and teacher >= 0.999) else 0.0,
        "length": _length_similarity(episode.get("final_answer") or "", episode.get("gold_answer") or ""),
    }


def is_teacher_correct(episode: Dict[str, Any]) -> bool:
    m = episode.get("final_metrics", {}) or {}
    return float(m.get("teacher_answer_score") or 0.0) >= TEACHER_CORRECT_THRESHOLD


def best_answer_score(episode: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Return ``(score, components)``; score is 0.0 when the teacher judged the answer wrong."""
    comps = score_components(episode)
    if not is_teacher_correct(episode):
        return 0.0, comps
    return sum(WEIGHTS[k] * v for k, v in comps.items()), comps


def load_run_episodes(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """qid -> episode for one consolidated run dir."""
    episodes: Dict[str, Dict[str, Any]] = {}
    for ep_file in sorted(run_dir.rglob(EPISODE_FILENAME)):
        for line in ep_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ep = json.loads(line)
                p = ep_file.parent
                ep["_sample_dir"] = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
                episodes[ep["qid"]] = ep
    return episodes


def load_all_runs(runs: Dict[str, str] | None = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """run label -> (qid -> episode), validating that all runs cover the same qids."""
    runs = runs or DEFAULT_RUNS
    loaded = {label: load_run_episodes(REPO_ROOT / rel) for label, rel in runs.items()}
    qid_sets = {label: set(eps) for label, eps in loaded.items()}
    base_label = next(iter(qid_sets))
    base = qid_sets[base_label]
    for label, qids in qid_sets.items():
        if qids != base:
            raise ValueError(
                f"run '{label}' covers {len(qids)} qids but '{base_label}' covers {len(base)} "
                f"(symmetric diff {len(qids ^ base)}); runs must share the same question set"
            )
    return loaded


def build_score_table(
    loaded: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Return ``(qids, scores[qid][run], components[qid][run])`` with a stable qid order."""
    labels = list(loaded)
    qids = sorted(loaded[labels[0]])
    scores: Dict[str, Dict[str, float]] = {}
    comps: Dict[str, Dict[str, Dict[str, float]]] = {}
    for qid in qids:
        scores[qid] = {}
        comps[qid] = {}
        for label in labels:
            s, c = best_answer_score(loaded[label][qid])
            scores[qid][label] = s
            comps[qid][label] = c
    return qids, scores, comps


# ------------------------------------------------------------------------- plots

# Reference dataviz tokens (light surface). Single categorical hue, validated:
# #2a78d6 passes lightness/chroma/contrast on #fcfcfb.
_BLUE = "#2a78d6"
_BLUE_DARK = "#184f95"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar(ax, labels, values, fmt):
    bars = ax.bar(labels, values, color=_BLUE, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9, color=_INK_2)


def write_plots(out: Path, labels, scores, stats, winners) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fig(w=7.0, h=3.6):
        fig, ax = plt.subplots(figsize=(w, h), facecolor=_SURFACE)
        _style_axes(ax)
        return fig, ax

    def _save(fig, name, title):
        fig.axes[0].set_title(title, color=_INK, fontsize=11, loc="left", pad=12)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150, facecolor=_SURFACE)
        plt.close(fig)

    fig, ax = _fig()
    _bar(ax, labels, [stats[l]["mean_score_all"] for l in labels], "{:.3f}")
    ax.set_ylim(0, 0.55)
    _save(fig, "mean_score_per_run.png",
          "Mean best_answer_score per run (all 3000 questions, wrong answers score 0)")

    fig, ax = _fig()
    data = [[scores[q][l] for q in scores if scores[q][l] > 0] for l in labels]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops={"color": _BLUE_DARK, "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 3,
                                "markerfacecolor": _MUTED, "markeredgecolor": _MUTED})
    for box in bp["boxes"]:
        box.set(facecolor=_BLUE, alpha=0.35, edgecolor=_BLUE)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set(color=_BASELINE)
    _save(fig, "score_distribution_per_run.png",
          "best_answer_score distribution per run (teacher-correct episodes only)")

    fig, ax = _fig()
    _bar(ax, labels, [winners[l] for l in labels], "{:.0f}")
    _save(fig, "ds1_winner_counts.png",
          "ds1 horizontal-best: questions won per run (best score across the 6 runs)")

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), facecolor=_SURFACE,
                             sharex=True, sharey=True)
    for ax, label, vals in zip(axes.flat, labels, data):
        _style_axes(ax)
        ax.hist(vals, bins=24, range=(0, 1), color=_BLUE)
        ax.set_title(f"{label}  (n={len(vals)})", color=_INK_2, fontsize=10, loc="left")
    fig.suptitle("best_answer_score histograms, teacher-correct episodes per run",
                 color=_INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "score_histograms.png", dpi=150, facecolor=_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------- CLI


def _run_stats(label: str, episodes: Dict[str, Dict[str, Any]],
               scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    vals = [scores[qid][label] for qid in scores]
    pos = [v for v in vals if v > 0]
    comp_sums: Dict[str, float] = {k: 0.0 for k in WEIGHTS}
    for qid, ep in episodes.items():
        if scores[qid][label] > 0:
            for k, v in score_components(ep).items():
                comp_sums[k] += v
    n_pos = max(1, len(pos))
    return {
        "mean_score_all": sum(vals) / len(vals),
        "mean_score_correct": (sum(pos) / len(pos)) if pos else 0.0,
        "n_correct": len(pos),
        "correct_rate": len(pos) / len(vals),
        "mean_components_correct": {k: comp_sums[k] / n_pos for k in WEIGHTS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="reports/best_answer_v1")
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    loaded = load_all_runs()
    qids, scores, _ = build_score_table(loaded)
    labels = list(loaded)

    with open(out / "score_matrix.csv", "w", encoding="utf-8") as f:
        f.write("qid," + ",".join(labels) + "\n")
        for qid in qids:
            f.write(qid + "," + ",".join(f"{scores[qid][l]:.6f}" for l in labels) + "\n")

    stats = {label: _run_stats(label, loaded[label], scores) for label in labels}
    winners = {label: 0 for label in labels}
    none_correct = 0
    for qid in qids:
        row = scores[qid]
        best = max(row.values())
        if best <= 0:
            none_correct += 1
            continue
        winners[max(labels, key=lambda l: (row[l], -loaded[l][qid].get("used_steps", 99)))] += 1
    summary = {
        "n_questions": len(qids),
        "runs": stats,
        "ds1_winner_counts": winners,
        "questions_with_no_correct_answer": none_correct,
        "weights": WEIGHTS,
        "teacher_correct_threshold": TEACHER_CORRECT_THRESHOLD,
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_plots(out, labels, scores, stats, winners)

    print(f"wrote {out / 'score_matrix.csv'} ({len(qids)} rows x {len(labels)} runs)")
    print(f"{'run':>5} | {'mean(all)':>9} | {'mean(correct)':>13} | {'n_correct':>9}")
    for label in labels:
        s = stats[label]
        print(f"{label:>5} | {s['mean_score_all']:>9.4f} | {s['mean_score_correct']:>13.4f} | {s['n_correct']:>9}")
    print("ds1 winner counts:", winners, "| no correct answer:", none_correct)


if __name__ == "__main__":
    main()
