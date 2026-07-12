"""Cross-model comparison report for the unseen-100 four-arm experiments.

Takes several analyzed experiment dirs (each already has analysis/stats.json from
analyze_experiment.py) plus a human label per model, and writes a single
self-contained HTML report that puts the models side by side:

  * grouped bars: teacher-verdict correctness per arm, one group per model
  * a table of every arm x model (judge-correct mean +- SD, plus EM/F1/cover)
  * the "internalization lift" (m1 - base) and "teacher-on-top lift"
    (m1_teacher - m1) per model, so the qualitative story per model is explicit
  * narrative highlighting where the ranking of arms changes between models

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
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ["base", "base_teacher", "m1", "m1_teacher"]
ARM_LABEL = {"base": "base", "base_teacher": "base + teacher",
             "m1": "m1 (internalized)", "m1_teacher": "m1 + teacher"}
ARM_COLOR = {"base": "#898781", "base_teacher": "#2a78d6",
             "m1": "#1baf7a", "m1_teacher": "#eda100"}


def load(exp_dir: str) -> Dict[str, Any]:
    return json.loads((Path(exp_dir) / "analysis" / "stats.json").read_text())


def grouped_bar(path: Path, models: List[str], stats: Dict[str, Any], metric: str,
                ylabel: str):
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3, 5), dpi=150)
    n_arms = len(ARMS)
    group_w = 0.8
    bw = group_w / n_arms
    x = np.arange(len(models))
    for j, arm in enumerate(ARMS):
        means, sds = [], []
        for m in models:
            s = stats[m]["per_arm"].get(arm, {}).get(metric)
            means.append(s["mean"] if s else 0.0)
            sds.append(s["sd"] if s else 0.0)
        off = -group_w / 2 + bw * (j + 0.5)
        bars = ax.bar(x + off, means, bw * 0.92, yerr=sds, capsize=3,
                      color=ARM_COLOR[arm], label=ARM_LABEL[arm], alpha=0.88,
                      error_kw=dict(lw=1, alpha=0.6))
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mv + 0.015, f"{mv*100:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.set_ylim(0, max(0.8, ax.get_ylim()[1]))
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.10))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def img64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def sig_label(p) -> str:
    if p is None:
        return "—"
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return str(p)
    if pv < 0.001:
        return f"{pv:.1e} ✓"
    return (f"{pv:.3f} ✓" if pv < 0.05 else f"{pv:.3f}")


def find_pair(stats_m, hi, lo, metric="judge_correct"):
    for t in stats_m["tests"]:
        if t["metric"] == metric and t["pair"] == f"{hi} vs {lo}":
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Cross-model comparison — unseen-100")
    ap.add_argument("models", nargs="+", help='"label"=<exp_dir> pairs')
    args = ap.parse_args()

    models: List[str] = []
    stats: Dict[str, Any] = {}
    for spec in args.models:
        label, d = spec.split("=", 1)
        models.append(label)
        stats[label] = load(d)

    out = Path(args.out)
    plot_dir = out.parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    p_judge = plot_dir / "cmp_judge.png"
    p_cover = plot_dir / "cmp_cover.png"
    grouped_bar(p_judge, models, stats, "judge_correct", "teacher-verdict correct")
    grouped_bar(p_cover, models, stats, "cover", "cover-match")

    esc = html.escape

    def cell(m, arm, metric, pct=True):
        s = stats[m]["per_arm"].get(arm, {}).get(metric)
        if not s:
            return "—"
        return f"{s['mean']*100:.1f}% ± {s['sd']*100:.1f}" if pct else f"{s['mean']:.3f}"

    # main table: judge-correct, one row per arm, one col per model
    main_rows = ""
    for arm in ARMS:
        cells = "".join(f"<td>{cell(m, arm, 'judge_correct')}</td>" for m in models)
        main_rows += f"<tr><td>{ARM_LABEL[arm]}</td>{cells}</tr>"

    # per-model story: internalization lift (m1 vs base) and teacher-on-top (m1_teacher vs m1)
    lift_rows = ""
    for m in models:
        pa = stats[m]["per_arm"]
        base = pa.get("base", {}).get("judge_correct", {}).get("mean")
        m1 = pa.get("m1", {}).get("judge_correct", {}).get("mean")
        bt = pa.get("base_teacher", {}).get("judge_correct", {}).get("mean")
        m1t = pa.get("m1_teacher", {}).get("judge_correct", {}).get("mean")
        t_int = find_pair(stats[m], "m1", "base")
        t_top = find_pair(stats[m], "m1_teacher", "m1")
        t_m1_vs_bt = find_pair(stats[m], "m1", "base_teacher")
        best_arm = max(ARMS, key=lambda a: pa.get(a, {}).get("judge_correct", {}).get("mean", -1))
        lift_rows += (
            f"<tr><td><b>{esc(m)}</b></td>"
            f"<td>{(m1-base)*100:+.1f} pts<br><span class=p>{sig_label((t_int or {}).get('paired_t',{}).get('p_holm'))}</span></td>"
            f"<td>{(m1-bt)*100:+.1f} pts<br><span class=p>{sig_label((t_m1_vs_bt or {}).get('paired_t',{}).get('p_holm'))}</span></td>"
            f"<td>{(m1t-m1)*100:+.1f} pts<br><span class=p>{sig_label((t_top or {}).get('paired_t',{}).get('p_holm'))}</span></td>"
            f"<td>{ARM_LABEL[best_arm]} ({pa[best_arm]['judge_correct']['mean']*100:.0f}%)</td></tr>")

    doc = f"""<title>{esc(args.title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0 auto;
         max-width: 1000px; padding: 26px 18px 60px; color: #0b0b0b; background: #fcfcfb; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #fff; background: #1a1a19; }}
    table, th, td {{ border-color: #2c2c2a !important; }} th {{ color: #c3c2b7 !important; }}
    .sub, .note, .p {{ color: #c3c2b7 !important; }} }}
  h1 {{ font-size: 21px; margin-bottom: 2px; }} h2 {{ font-size: 16px; margin-top: 32px; }}
  .sub {{ color: #52514e; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
          font-variant-numeric: tabular-nums; margin-top: 8px; }}
  th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #e1e0d9; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }} th {{ color: #52514e; font-weight: 600; }}
  .note {{ font-size: 12px; color: #898781; margin-top: 8px; }} .p {{ font-size: 11px; color: #898781; }}
  img {{ max-width: 100%; margin-top: 10px; }} .overflow {{ overflow-x: auto; }}
</style>
<h1>{esc(args.title)}</h1>
<p class="sub">Same fixed granite-4.1-3B trace dataset trains each student (m1); all evaluated on the
identical 100 unseen HotpotQA questions, 5 seeds, budget 4, temperature 0.2, vLLM backend, with a
post-hoc gpt-oss-120b verdict on every final answer. Models: {esc(", ".join(models))}.</p>

<h2>Teacher-verdict correctness — every arm, every model</h2>
<img alt="judge-correct grouped bars" src="{img64(p_judge)}">
<div class="overflow"><table>
<thead><tr><th>arm</th>{"".join(f"<th>{esc(m)}</th>" for m in models)}</tr></thead>
<tbody>{main_rows}</tbody></table></div>
<p class="note">Mean ± SD of per-seed run rates. Error bars on the chart are SD across the 5 seeds.</p>

<h2>The story per model — where does the value come from?</h2>
<div class="overflow"><table>
<thead><tr><th>model</th><th>internalization<br>(m1 − base)</th><th>m1 vs<br>base + teacher</th>
<th>teacher on top<br>(m1+teacher − m1)</th><th>best arm</th></tr></thead>
<tbody>{lift_rows}</tbody></table></div>
<p class="note">✓ = significant at p<sub>Holm</sub> &lt; 0.05 (paired t over per-question means). "m1 vs base+teacher"
is the key contrast: positive means the trained student <b>without</b> a teacher already beats the untrained
student <b>with</b> the live teacher.</p>

<h2>Cover-match (lexical) — for reference</h2>
<img alt="cover grouped bars" src="{img64(p_cover)}">
"""
    out.write_text(doc)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
