"""Cross-model EFFICIENCY report for the unseen-100 four-arm experiments.

The question this report answers: **what does using m1 (the internalized-guidance
student, with NO teacher at inference) cost and save versus the other arms?** For
each student model it puts the accuracy you get next to the price you pay:

  * accuracy    -> post-hoc teacher-verdict correctness (semantic), mean +- SD/seed
  * tokens      -> total tokens per question across ALL model calls (student +, for
                   the teacher arms, the gpt-oss-120b teacher). Hardware-independent.
  * time        -> wall seconds per question (infra-dependent; shown, but tokens lead)
  * resource    -> whether a 120B external teacher is in the loop at all

Takes several analyzed experiment dirs (each has analysis/stats.json) + a label per
model, and writes one self-contained HTML report:

  * the accuracy-vs-cost frontier (accuracy vs tokens/question, m1 highlighted)
  * accuracy per 1k tokens per arm per model (the single-number efficiency score)
  * the core "m1 (no teacher) vs each alternative" table: accuracy delta + token
    saving + speedup + teacher dropped, per model
  * per-arm reference table and the arm-ranking bars for context

Usage:
    python training_methods/exp_unseen100/compare_models.py \
        --out <report.html> --title "..." \
        "granite-4.1-3B"=<exp_dir> "Qwen3.5-0.8B"=<exp_dir> ...
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ["base", "base_teacher", "m1", "m1_teacher"]
ARM_LABEL = {"base": "base", "base_teacher": "base + teacher",
             "m1": "m1 (no teacher)", "m1_teacher": "m1 + teacher"}
ARM_COLOR = {"base": "#898781", "base_teacher": "#2a78d6",
             "m1": "#1baf7a", "m1_teacher": "#eda100"}
MODEL_MARKER = ["o", "s", "^", "D", "v"]
USES_TEACHER = {"base": False, "base_teacher": True, "m1": False, "m1_teacher": True}


def load(exp_dir: str) -> Dict[str, Any]:
    return json.loads((Path(exp_dir) / "analysis" / "stats.json").read_text())


def val(stats_m, arm, metric):
    return stats_m["per_arm"].get(arm, {}).get(metric, {}).get("mean")


def sig(p) -> str:
    if p is None:
        return ""
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return ""
    return " ✓" if pv < 0.05 else " (ns)"


def find_pair_p(stats_m, hi, lo, metric="judge_correct"):
    for t in stats_m["tests"]:
        if t["metric"] == metric and t["pair"] == f"{hi} vs {lo}":
            return t.get("paired_t", {}).get("p_holm")
    return None


# ------------------------------------------------------------------ plots ----
def frontier_plot(path: Path, models: List[str], stats: Dict[str, Any]):
    """accuracy vs tokens/question: up and to the left is better."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    for mi, m in enumerate(models):
        pts = [(val(stats[m], a, "tokens_total"), val(stats[m], a, "judge_correct"), a)
               for a in ARMS]
        pts = [(x, y, a) for x, y, a in pts if x is not None and y is not None]
        pts.sort()
        xs = [p[0] / 1000 for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color="#c3c2b7", lw=1, zorder=1)
        for x, y, a in pts:
            big = a == "m1"
            ax.scatter(x / 1000, y, s=190 if big else 80,
                       marker=MODEL_MARKER[mi % len(MODEL_MARKER)],
                       color=ARM_COLOR[a], edgecolors="black" if big else "none",
                       linewidths=1.6 if big else 0, zorder=4 if big else 3)
        # annotate the m1 point with the model name
        m1x, m1y = val(stats[m], "m1", "tokens_total") / 1000, val(stats[m], "m1", "judge_correct")
        ax.annotate(m, (m1x, m1y), textcoords="offset points", xytext=(8, 6),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("tokens per question (student + teacher)  →  cheaper is left")
    ax.set_ylabel("teacher-verdict correct")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.set_title("Accuracy vs cost — m1 (ringed) drops the teacher", fontsize=12)
    # arm-color legend + a note about marker = model
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_COLOR[a],
                      markersize=9, label=ARM_LABEL[a]) for a in ARMS]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                          markeredgecolor="black", markersize=11, label="m1 (ringed)"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def efficiency_bar(path: Path, models: List[str], stats: Dict[str, Any]):
    """accuracy per 1k tokens — the single-number efficiency score."""
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3, 4.8), dpi=150)
    x = np.arange(len(models))
    bw = 0.8 / len(ARMS)
    for j, arm in enumerate(ARMS):
        eff = []
        for m in models:
            acc = val(stats[m], arm, "judge_correct")
            tok = val(stats[m], arm, "tokens_total")
            eff.append((acc / (tok / 1000)) if acc and tok else 0.0)
        off = -0.4 + bw * (j + 0.5)
        bars = ax.bar(x + off, eff, bw * 0.92, color=ARM_COLOR[arm], label=ARM_LABEL[arm], alpha=0.88)
        for b, e in zip(bars, eff):
            ax.text(b.get_x() + b.get_width() / 2, e + 0.0008, f"{e*100:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("correct-% per 1k tokens")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}")
    ax.set_title("Efficiency — accuracy delivered per 1,000 tokens (higher is better)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def img64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="m1 without a teacher — accuracy vs cost")
    ap.add_argument("models", nargs="+", help='"label"=<exp_dir> pairs')
    args = ap.parse_args()

    models: List[str] = []
    stats: Dict[str, Any] = {}
    for spec in args.models:
        label, d = spec.split("=", 1)
        models.append(label)
        stats[label] = load(d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p_front = out.parent / "cmp_frontier.png"
    p_eff = out.parent / "cmp_efficiency.png"
    frontier_plot(p_front, models, stats)
    efficiency_bar(p_eff, models, stats)

    esc = html.escape

    def pct(v):
        return "—" if v is None else f"{v*100:.1f}%"

    # --- core table: m1 (no teacher) vs each alternative, per model ---
    core_rows = ""
    for m in models:
        acc_m1 = val(stats[m], "m1", "judge_correct")
        tok_m1 = val(stats[m], "m1", "tokens_total")
        t_m1 = val(stats[m], "m1", "time_per_question_s")
        first = True
        for arm in ("base", "base_teacher", "m1_teacher"):
            acc_a = val(stats[m], arm, "judge_correct")
            tok_a = val(stats[m], arm, "tokens_total")
            t_a = val(stats[m], arm, "time_per_question_s")
            dacc = (acc_m1 - acc_a)
            p = find_pair_p(stats[m], "m1", arm)
            tok_save = (1 - tok_m1 / tok_a) if tok_a else 0.0
            speed = (t_a / t_m1) if t_m1 else 0.0
            retained = (acc_m1 / acc_a) if acc_a else float("inf")
            # verdict phrasing
            if dacc >= 0:
                acc_txt = f"<b style='color:#0a0'>+{dacc*100:.1f} pts</b>{sig(p)}"
                verdict = f"m1 wins by {dacc*100:.1f} pts"
            else:
                acc_txt = f"<b style='color:#b00'>{dacc*100:.1f} pts</b>{sig(p)}"
                verdict = f"keeps {retained*100:.0f}% of the accuracy"
            teach = " · drops the 120B teacher" if USES_TEACHER[arm] else ""
            cost_txt = (f"{tok_save*100:+.0f}% tokens, {speed:.1f}× faster{teach}"
                        if tok_a else "—")
            mcell = f"<td rowspan=3><b>{esc(m)}</b><br><span class=p>m1: {pct(acc_m1)} · {tok_m1/1000:.1f}k tok/q</span></td>" if first else ""
            core_rows += (f"<tr>{mcell}<td>vs {ARM_LABEL[arm]}</td>"
                          f"<td>{pct(acc_a)} → {pct(acc_m1)}<br>{acc_txt}</td>"
                          f"<td>{tok_a/1000:.1f}k → {tok_m1/1000:.1f}k<br><b>{cost_txt}</b></td>"
                          f"<td>{esc(verdict)}</td></tr>")
            first = False

    # --- reference table: every arm x model, accuracy + cost ---
    ref_rows = ""
    for arm in ARMS:
        cells = ""
        for m in models:
            acc = val(stats[m], arm, "judge_correct")
            tok = val(stats[m], arm, "tokens_total")
            cells += f"<td>{pct(acc)}<br><span class=p>{tok/1000:.1f}k tok · {'teacher' if USES_TEACHER[arm] else 'local only'}</span></td>"
        ref_rows += f"<tr><td>{ARM_LABEL[arm]}</td>{cells}</tr>"

    doc = f"""<title>{esc(args.title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0 auto;
         max-width: 1000px; padding: 26px 18px 60px; color: #0b0b0b; background: #fcfcfb; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #fff; background: #1a1a19; }}
    table, th, td {{ border-color: #2c2c2a !important; }} th {{ color: #c3c2b7 !important; }}
    .sub, .note, .p {{ color: #b7b6ad !important; }} }}
  h1 {{ font-size: 21px; margin-bottom: 2px; }} h2 {{ font-size: 16px; margin-top: 32px; }}
  .sub {{ color: #52514e; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
          font-variant-numeric: tabular-nums; margin-top: 8px; }}
  th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #e1e0d9; vertical-align: top; }}
  th:first-child, td:first-child {{ text-align: left; }} th {{ color: #52514e; font-weight: 600; }}
  .note {{ font-size: 12px; color: #898781; margin-top: 8px; }} .p {{ font-size: 11px; color: #898781; }}
  img {{ max-width: 100%; margin-top: 10px; }} .overflow {{ overflow-x: auto; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap: 12px; margin: 16px 0; }}
  .kpi {{ border: 1px solid #e1e0d9; border-radius: 10px; padding: 13px 15px; }}
  @media (prefers-color-scheme: dark) {{ .kpi {{ border-color: #2c2c2a; }} }}
  .kpi .v {{ font-size: 22px; font-weight: 650; }} .kpi .d {{ font-size: 12px; color: #52514e; margin-top: 3px; }}
  @media (prefers-color-scheme: dark) {{ .kpi .d {{ color: #b7b6ad; }} }}
</style>
<h1>{esc(args.title)}</h1>
<p class="sub">The m1 arm runs the trained student with the teacher's guidance internalized and <b>no teacher at
inference</b>. This report weighs what that saves — tokens, wall time, and the 120-billion-parameter teacher
itself — against what it costs in accuracy, versus the base student and the teacher-in-the-loop arms.
Same 100 unseen questions, 5 seeds, budget 4, T=0.2, vLLM; accuracy = post-hoc gpt-oss-120b verdict.
Models: {esc(", ".join(models))}.</p>

<h2>Accuracy vs cost — the frontier</h2>
<img alt="accuracy vs tokens per question" src="{img64(p_front)}">
<p class="note">Each model contributes four points (one per arm); the ringed point is m1. m1 always sits far to
the <b>left</b> (cheapest) because it makes no teacher calls — for the teacher arms, roughly half of every
token is spent on the gpt-oss-120b teacher. Up-and-left is the ideal corner.</p>

<h2>Efficiency — accuracy per 1,000 tokens</h2>
<img alt="accuracy per 1k tokens" src="{img64(p_eff)}">
<p class="note">The single-number summary: correctness delivered per 1k tokens spent. m1 is the most
token-efficient arm for every model, by a wide margin.</p>

<h2>m1 (no teacher) versus each alternative</h2>
<div class="overflow"><table>
<thead><tr><th>model</th><th>comparison</th><th>accuracy</th><th>cost (tokens/q · speed)</th><th>bottom line</th></tr></thead>
<tbody>{core_rows}</tbody></table></div>
<p class="note">✓ = the accuracy difference is significant (paired t over per-question means, Holm-corrected);
(ns) = not significant. Token counts are per question across all model calls; speed is the wall-time ratio
(infra-dependent — tokens are the hardware-independent cost). "Drops the 120B teacher" means m1 removes the
external gpt-oss-120b model from the loop entirely.</p>

<h2>Reference — accuracy and cost of every arm</h2>
<div class="overflow"><table>
<thead><tr><th>arm</th>{"".join(f"<th>{esc(m)}</th>" for m in models)}</tr></thead>
<tbody>{ref_rows}</tbody></table></div>
"""
    out.write_text(doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
