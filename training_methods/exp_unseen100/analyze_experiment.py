"""exp_unseen100 analysis: statistics, significance tests, box plots, HTML report.

Reads one experiment dir produced by run_experiment.py (15 runs + judge verdicts +
gpu samples + job records) and writes into <exp-dir>/analysis/:

    stats.json          per-rep aggregates, per-arm mean/SD/var/CI, test results
    *.png               box plots (all parts: quartiles, whiskers, outliers, means,
                        jittered raw points) and the GPU-memory timeline
    report.html         self-contained evaluation page (tables + embedded plots)

Metrics per rep-run: EM, F1, cover-match, doc-recall, judge-correct rate (post-hoc
teacher verdict), mean steps, voluntary-finish rate, invalid actions, student/teacher
tokens, wall time, per-question and per-step time, peak GPU memory.

Tests (for each metric and arm pair): paired t-test and Wilcoxon signed-rank over
per-question values (each question's mean across the 5 seeds), plus Welch's t-test
over the 5 per-rep aggregates. Holm correction across the three arm pairs.

Usage:
    .venv_train/bin/python training_methods/exp_unseen100/analyze_experiment.py \
        --exp-dir training_methods/exp_unseen100/runs/<ts>_exp
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

ARMS = ["base", "base_teacher", "m1", "m1_teacher"]
ARM_LABEL = {
    "base": "base (no guidance)",
    "base_teacher": "base + teacher",
    "m1": "m1 (internalized)",
    "m1_teacher": "m1 + teacher",
}
ARM_COLOR = {"base": "#898781", "base_teacher": "#2a78d6",
             "m1": "#1baf7a", "m1_teacher": "#eda100"}
PAIRS = [("m1", "base"), ("base_teacher", "base"),
         ("m1", "base_teacher"), ("m1_teacher", "m1"), ("m1_teacher", "base_teacher")]

EP_METRICS = ["em", "f1", "cover", "doc_recall", "judge_correct", "steps",
              "voluntary_finish", "elapsed_s", "tokens_total"]
REP_ONLY = ["wall_time_s", "gpu_peak_mem_gb", "student_tokens", "teacher_tokens",
            "time_per_question_s", "time_per_step_s", "invalid_steps"]


def load_experiment(exp_dir: Path, log):
    """Return runs[(arm, seed)] = {"episodes": [...], "metrics": {...}} + judge map."""
    verdicts: Dict[tuple, Dict] = {}
    jf = exp_dir / "judge" / "verdicts.jsonl"
    if jf.exists():
        for line in open(jf):
            r = json.loads(line)
            verdicts[(str(Path(r["source"]).resolve()), r["qid"])] = r.get("verdict")
    runs: Dict[tuple, Dict[str, Any]] = {}
    for mfile in sorted(exp_dir.glob("*/metrics.json")):
        d = mfile.parent
        tag = d.name.split("_", 1)[1]           # <ts>_<arm>_s<seed>
        arm, seed = tag.rsplit("_s", 1)
        eps = [json.loads(l) for l in open(d / "episodes.jsonl")]
        src = str((d / "episodes.jsonl").resolve())
        for e in eps:
            e["judge_verdict"] = verdicts.get((src, e["qid"]))
        runs[(arm, int(seed))] = {
            "dir": str(d), "episodes": eps, "metrics": json.loads(mfile.read_text()),
        }
    log.info(f"loaded {len(runs)} runs: {sorted(runs)}")
    return runs


def episode_row(e: Dict[str, Any]) -> Dict[str, float]:
    fm = e["final_metrics"]
    # student tokens: teacherless runs record gen_stats; teacher runs record usage
    tok = 0
    for s in e["steps"]:
        g = s.get("gen_stats")
        if g:
            tok += g["prompt_tokens"] + g["completion_tokens"]
        for c in (s.get("student_calls") or []):
            u = c.get("usage") or {}
            tok += (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        for c in (s.get("teacher_calls") or []):
            u = c.get("usage") or {}
            tok += (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
    g = (e.get("plan") or {}).get("gen_stats")
    if g:
        tok += g["prompt_tokens"] + g["completion_tokens"]
    jv = e.get("judge_verdict")
    return {
        "em": float(fm["exact_match"]),
        "f1": fm["f1"],
        "cover": float(fm["cover_match"]),
        "doc_recall": fm["doc_recall"] if fm["doc_recall"] is not None else np.nan,
        "judge_correct": float(jv["correct"]) if jv else np.nan,
        "steps": e["used_steps"],
        "voluntary_finish": float(e["stop_reason"] in ("finish", "teacher_accept")),
        "elapsed_s": e.get("elapsed_s", np.nan),
        "tokens_total": tok,
    }


def rep_aggregate(arm: str, seed: int, run: Dict[str, Any]) -> Dict[str, Any]:
    eps = run["episodes"]
    rows = [episode_row(e) for e in eps]
    m = run["metrics"]
    out: Dict[str, Any] = {"arm": arm, "seed": seed, "n": len(eps)}
    for k in EP_METRICS:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    stu_tok = (m.get("student_prompt_tokens") or 0) + (m.get("student_completion_tokens") or 0)
    tea_tok = (m.get("teacher_prompt_tokens") or 0) + (m.get("teacher_completion_tokens") or 0)
    total_steps = sum(e["used_steps"] for e in eps)
    out.update({
        "wall_time_s": m.get("wall_time_s"),
        "gpu_peak_mem_gb": m.get("gpu_peak_mem_gb"),
        "student_tokens": stu_tok,
        "teacher_tokens": tea_tok,
        "invalid_steps": m.get("invalid_action_steps"),
        "time_per_question_s": round(m["wall_time_s"] / max(len(eps), 1), 2) if m.get("wall_time_s") else None,
        "time_per_step_s": round(m["wall_time_s"] / max(total_steps, 1), 2) if m.get("wall_time_s") else None,
        "stop_reasons": m.get("stop_reasons"),
    })
    return out


def arm_summary(reps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_reps": len(reps)}
    for k in EP_METRICS + REP_ONLY:
        vals = [r[k] for r in reps if r.get(k) is not None]
        if not vals:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = sps.t.ppf(0.975, len(vals) - 1) * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[k] = {
            "mean": round(mean, 4), "sd": round(sd, 4), "var": round(sd ** 2, 6),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "ci95": round(ci, 4), "values": [round(v, 4) for v in vals],
        }
    return out


def per_question_matrix(runs, arm: str, metric: str) -> Dict[str, float]:
    """qid -> mean of metric across this arm's reps."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for (a, _seed), run in runs.items():
        if a != arm:
            continue
        for e in run["episodes"]:
            v = episode_row(e)[metric]
            if not (isinstance(v, float) and math.isnan(v)):
                acc[e["qid"]].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def paired_tests(runs, reps_by_arm) -> List[Dict[str, Any]]:
    tests = []
    for metric in ("f1", "cover", "em", "judge_correct", "doc_recall", "tokens_total"):
        for hi, lo in PAIRS:
            mhi = per_question_matrix(runs, hi, metric)
            mlo = per_question_matrix(runs, lo, metric)
            qids = sorted(set(mhi) & set(mlo))
            a = np.array([mhi[q] for q in qids])
            b = np.array([mlo[q] for q in qids])
            entry: Dict[str, Any] = {
                "metric": metric, "pair": f"{hi} vs {lo}", "n_questions": len(qids),
                "mean_diff": round(float(np.mean(a - b)), 4),
            }
            if len(qids) >= 5 and np.std(a - b) > 0:
                t, p_t = sps.ttest_rel(a, b)
                entry["paired_t"] = {"t": round(float(t), 3), "p": float(p_t)}
                diffs = a - b
                nz = diffs[diffs != 0]
                if len(nz) >= 5:
                    w, p_w = sps.wilcoxon(a, b)
                    entry["wilcoxon"] = {"W": round(float(w), 1), "p": float(p_w)}
                d = float(np.mean(diffs) / np.std(diffs, ddof=1))
                entry["cohens_d_paired"] = round(d, 3)
            # secondary: Welch over the 5 per-rep aggregates
            ra = [r[metric] for r in reps_by_arm[hi] if r.get(metric) is not None]
            rb = [r[metric] for r in reps_by_arm[lo] if r.get(metric) is not None]
            if len(ra) >= 2 and len(rb) >= 2:
                t2, p2 = sps.ttest_ind(ra, rb, equal_var=False)
                entry["welch_reps"] = {"t": round(float(t2), 3), "p": float(p2)}
            tests.append(entry)
    # Holm correction within each metric across the three pairs (primary = paired_t)
    for metric in {t["metric"] for t in tests}:
        sub = [t for t in tests if t["metric"] == metric and "paired_t" in t]
        ps = sorted(((t["paired_t"]["p"], t) for t in sub), key=lambda x: x[0])
        for rank, (p, t) in enumerate(ps):
            t["paired_t"]["p_holm"] = min(1.0, p * (len(ps) - rank))
    for t in tests:
        for k in ("paired_t", "wilcoxon", "welch_reps"):
            if k in t:
                t[k]["p"] = float(f"{t[k]['p']:.2e}") if t[k]["p"] < 0.001 else round(t[k]["p"], 4)
                if "p_holm" in t[k]:
                    t[k]["p_holm"] = float(f"{t[k]['p_holm']:.2e}") if t[k]["p_holm"] < 0.001 else round(t[k]["p_holm"], 4)
    return tests


# ---------------------------------------------------------------- plotting ----
def style_ax(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def boxplot(path: Path, title: str, ylabel: str, data: Dict[str, List[float]],
            percent: bool = False):
    """Box with all parts: box/median/whiskers/caps/fliers + mean marker + jittered points."""
    arms = [a for a in ARMS if a in data and data[a]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    vals = [data[a] for a in arms]
    bp = ax.boxplot(vals, patch_artist=True, showmeans=True, widths=0.5,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black", linewidth=1.4),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
    for patch, arm in zip(bp["boxes"], arms):
        patch.set_facecolor(ARM_COLOR[arm])
        patch.set_alpha(0.45)
        patch.set_edgecolor(ARM_COLOR[arm])
    rng = np.random.default_rng(7)
    for i, arm in enumerate(arms, 1):
        x = rng.normal(i, 0.06, size=len(data[arm]))
        ax.scatter(x, data[arm], s=14, color=ARM_COLOR[arm], alpha=0.65, zorder=3,
                   edgecolors="none")
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=10)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.set_title(title, fontsize=11)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gpu_plot(path: Path, exp_dir: Path):
    import csv
    series: Dict[str, List[tuple]] = defaultdict(list)
    with open(exp_dir / "gpu_samples.csv") as fh:
        for row in csv.DictReader(fh):
            series[row["gpu"]].append((row["ts_utc"], float(row["mem_used_mib"]) / 1024))
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    cmap = plt.get_cmap("tab10")
    for i, (gpu, pts) in enumerate(sorted(series.items())):
        ys = [p[1] for p in pts]
        ax.plot(range(len(ys)), ys, label=f"GPU {gpu}", linewidth=1.2, color=cmap(i))
    ax.set_xlabel("sample (15 s interval)")
    ax.set_ylabel("memory used (GB)")
    ax.set_title("GPU memory over the experiment", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ report ----
def img64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def fmt_stat(s: Dict[str, Any], percent=False) -> str:
    if not s:
        return "—"
    f = (lambda v: f"{v*100:.1f}%") if percent else (lambda v: f"{v:.3f}")
    return f"{f(s['mean'])} ± {f(s['sd'])}"


def build_report(exp_dir: Path, out: Path, summaries, reps, tests, plots, config, judge_summary):
    esc = html.escape
    pct_metrics = {"em", "cover", "judge_correct", "voluntary_finish", "doc_recall"}

    def arm_cells(metric):
        best = None
        vals = {a: summaries[a].get(metric, {}).get("mean") for a in ARMS if a in summaries}
        num = {a: v for a, v in vals.items() if v is not None}
        if num:
            best = max(num, key=num.get) if metric not in ("elapsed_s", "tokens_total", "wall_time_s") else min(num, key=num.get)
        cells = ""
        for a in ARMS:
            if a not in summaries:
                continue
            s = summaries[a].get(metric)
            klass = ' class="best"' if a == best and s else ""
            cells += f"<td{klass}>{fmt_stat(s, metric in pct_metrics)}</td>"
        return cells

    # executive summary: pairwise arm comparisons on the primary metric
    jt = {t["pair"]: t for t in tests if t["metric"] == "judge_correct" and "paired_t" in t}
    summary_rows = ""
    verdicts = []
    for hi, lo in PAIRS:
        key = f"{hi} vs {lo}"
        t = jt.get(key)
        if not t or hi not in summaries or lo not in summaries:
            continue
        m_hi = summaries[hi]["judge_correct"]["mean"]
        m_lo = summaries[lo]["judge_correct"]["mean"]
        p = t["paired_t"].get("p_holm", t["paired_t"]["p"])
        sig = p < 0.05
        direction = "improves on" if t["mean_diff"] > 0 else "trails"
        verdict = (f"<b>{'significant' if sig else 'NOT significant'}</b>"
                   f" (p<sub>Holm</sub> = {p}, d = {t.get('cohens_d_paired', '—')})")
        summary_rows += (
            f"<tr><td>{ARM_LABEL[lo]} → {ARM_LABEL[hi]}</td>"
            f"<td>{m_lo*100:.1f}% → {m_hi*100:.1f}%</td>"
            f"<td>{t['mean_diff']*100:+.1f} pts</td><td>{verdict}</td></tr>")
        verdicts.append((hi, lo, t["mean_diff"], sig))
    sig_up = [(h, l, d) for h, l, d, s in verdicts if s and d > 0]
    non_sig = [(h, l) for h, l, d, s in verdicts if not s]
    summary_prose = ""
    if sig_up:
        summary_prose += ("Statistically significant improvements (teacher-verdict correctness): "
                          + "; ".join(f"<b>{ARM_LABEL[l]} → {ARM_LABEL[h]}</b> ({d*100:+.1f} pts)"
                                      for h, l, d in sig_up) + ". ")
    if non_sig:
        summary_prose += ("No significant difference between "
                          + "; ".join(f"<b>{ARM_LABEL[h]}</b> and <b>{ARM_LABEL[l]}</b>"
                                      for h, l in non_sig)
                          + " — these arms are statistically indistinguishable at α = 0.05.")

    metric_rows = ""
    label = {
        "em": "exact match", "f1": "F1", "cover": "cover-match", "doc_recall": "doc recall",
        "judge_correct": "teacher verdict correct", "steps": "steps used",
        "voluntary_finish": "voluntary finish", "elapsed_s": "time / question (s)",
        "tokens_total": "tokens / question", "wall_time_s": "wall time / run (s)",
        "gpu_peak_mem_gb": "peak GPU memory (GB)", "student_tokens": "student tokens / run",
        "teacher_tokens": "teacher tokens / run", "time_per_step_s": "time / step (s)",
        "invalid_steps": "invalid actions / run",
    }
    for metric in ["judge_correct", "em", "f1", "cover", "doc_recall", "steps",
                   "voluntary_finish", "elapsed_s", "tokens_total", "wall_time_s",
                   "time_per_step_s", "student_tokens", "teacher_tokens",
                   "gpu_peak_mem_gb", "invalid_steps"]:
        metric_rows += f"<tr><td>{label.get(metric, metric)}</td>{arm_cells(metric)}</tr>"

    test_rows = ""
    for t in tests:
        pt = t.get("paired_t", {})
        wc = t.get("wilcoxon", {})
        we = t.get("welch_reps", {})
        sig = "✓" if pt.get("p_holm", 1) < 0.05 else ""
        test_rows += (
            f"<tr><td>{label.get(t['metric'], t['metric'])}</td><td>{esc(t['pair'])}</td>"
            f"<td>{t['mean_diff']:+.4f}</td>"
            f"<td>{pt.get('t','—')}</td><td>{pt.get('p','—')}</td><td>{pt.get('p_holm','—')} {sig}</td>"
            f"<td>{wc.get('p','—')}</td><td>{we.get('p','—')}</td>"
            f"<td>{t.get('cohens_d_paired','—')}</td></tr>")

    rep_rows = ""
    for r in sorted(reps, key=lambda r: (ARMS.index(r["arm"]), r["seed"])):
        rep_rows += (
            f"<tr><td>{ARM_LABEL[r['arm']]}</td><td>{r['seed']}</td><td>{r['n']}</td>"
            f"<td>{(r['judge_correct'] or 0)*100:.0f}%</td><td>{(r['em'] or 0)*100:.0f}%</td>"
            f"<td>{r['f1']:.3f}</td><td>{(r['cover'] or 0)*100:.0f}%</td>"
            f"<td>{r['steps']:.2f}</td><td>{r['wall_time_s']:.0f}</td>"
            f"<td>{r['tokens_total']:.0f}</td><td>{r['gpu_peak_mem_gb'] or '—'}</td></tr>")

    plots_html = "".join(
        f'<h3>{esc(title)}</h3><img alt="{esc(title)}" src="{img64(p)}" style="max-width:100%">'
        for title, p in plots)

    doc = f"""<title>exp_unseen100 — repeated three-arm evaluation</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0 auto;
         max-width: 1020px; padding: 26px 18px 60px; color: #0b0b0b; background: #fcfcfb; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #fff; background: #1a1a19; }}
    table, th, td {{ border-color: #2c2c2a !important; }} th {{ color: #c3c2b7 !important; }}
    .sub, .note {{ color: #c3c2b7 !important; }} }}
  h1 {{ font-size: 21px; margin-bottom: 2px; }} h2 {{ font-size: 16px; margin-top: 34px; }}
  h3 {{ font-size: 13.5px; margin: 22px 0 6px; }}
  .sub {{ color: #52514e; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
          font-variant-numeric: tabular-nums; margin-top: 8px; }}
  th, td {{ padding: 5px 9px; text-align: right; border-bottom: 1px solid #e1e0d9; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }} th {{ color: #52514e; font-weight: 600; }}
  td.best {{ font-weight: 700; }}
  .note {{ font-size: 12px; color: #898781; margin-top: 8px; }}
  .overflow {{ overflow-x: auto; }}
</style>
<h1>exp_unseen100 — repeated three-arm evaluation</h1>
<p class="sub">100 unseen HotpotQA train questions (seed-101 sample, zero overlap with training/dev/golden) ·
budget {config.get('budget', 4)} · student temperature {config.get('student_temperature', 0.2)} ·
{len({r['seed'] for r in reps})} seeds per arm · student granite-4.1-3b (HF bf16) ·
teacher/judge gpt-oss-120b via FAU→OpenRouter</p>

<h2>Summary — which arm beats which, and is it significant?</h2>
<div class="overflow"><table>
<thead><tr><th>comparison</th><th>teacher-verdict correct</th><th>Δ</th><th>significance (paired t, Holm-corrected)</th></tr></thead>
<tbody>{summary_rows}</tbody></table></div>
<p class="note">{summary_prose}</p>
<p class="note">Primary metric: post-hoc teacher-verdict correctness (semantic). Each comparison is a paired
t-test over per-question means across seeds, Holm-corrected within the metric; d is paired Cohen's d.
See the full test table below for EM/F1/cover/doc-recall and the non-parametric checks.</p>

<h2>Arm summary (mean ± SD across seeds)</h2>
<div class="overflow"><table>
<thead><tr><th>metric</th>{"".join(f"<th>{ARM_LABEL[a]}</th>" for a in ARMS if a in summaries)}</tr></thead>
<tbody>{metric_rows}</tbody></table></div>
<p class="note">Bold = best arm (lowest for time/token metrics). "Teacher verdict correct" is the post-hoc
gpt-oss-120b judgment of every final answer — the semantic correctness metric; EM/F1/cover are lexical.</p>

<h2>Significance tests</h2>
<div class="overflow"><table>
<thead><tr><th>metric</th><th>pair</th><th>Δ mean</th><th>t (paired)</th><th>p</th>
<th>p (Holm) </th><th>p (Wilcoxon)</th><th>p (Welch, reps)</th><th>Cohen's d</th></tr></thead>
<tbody>{test_rows}</tbody></table></div>
<p class="note">Primary test: paired t over per-question means (each question averaged over its
{len({r['seed'] for r in reps})} seeds, n = {config.get('n_questions', 100)} questions), Holm-corrected
across the three arm pairs per metric; Wilcoxon signed-rank as the non-parametric check;
Welch's t over the per-seed aggregates (n = {len({r['seed'] for r in reps})}) as a conservative secondary. ✓ = p<sub>Holm</sub> &lt; 0.05.</p>

<h2>Plots</h2>
{plots_html}

<h2>Per-run detail (15 runs)</h2>
<div class="overflow"><table>
<thead><tr><th>arm</th><th>seed</th><th>n</th><th>judge✓</th><th>EM</th><th>F1</th><th>cover</th>
<th>steps</th><th>wall s</th><th>tok/q</th><th>GPU GB</th></tr></thead>
<tbody>{rep_rows}</tbody></table></div>
"""
    out.write_text(doc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)
    out = exp_dir / "analysis"
    out.mkdir(exist_ok=True)
    log = setup_logger("exp_analysis", out / "analysis.log")

    runs = load_experiment(exp_dir, log)
    reps = [rep_aggregate(arm, seed, run) for (arm, seed), run in sorted(runs.items())]
    reps_by_arm: Dict[str, List[Dict]] = defaultdict(list)
    for r in reps:
        reps_by_arm[r["arm"]].append(r)
    summaries = {arm: arm_summary(rs) for arm, rs in reps_by_arm.items()}
    tests = paired_tests(runs, reps_by_arm)

    # plots
    plots: List[tuple] = []

    def add_box(fname, title, ylabel, data, percent=False):
        p = out / fname
        boxplot(p, title, ylabel, data, percent)
        plots.append((title, p))

    pq = lambda metric: {a: list(per_question_matrix(runs, a, metric).values()) for a in ARMS}
    rep_vals = lambda metric: {a: [r[metric] for r in reps_by_arm[a] if r.get(metric) is not None]
                               for a in ARMS}
    add_box("box_judge_q.png", "Teacher-verdict correctness per question (mean over seeds)",
            "correct rate", pq("judge_correct"), percent=True)
    add_box("box_f1_q.png", "F1 per question (mean over seeds)", "F1", pq("f1"))
    add_box("box_cover_rep.png", "Cover-match per rep-run (5 seeds)", "cover rate",
            rep_vals("cover"), percent=True)
    add_box("box_judge_rep.png", "Teacher-verdict correct per rep-run (5 seeds)", "correct rate",
            rep_vals("judge_correct"), percent=True)
    add_box("box_em_rep.png", "Exact match per rep-run (5 seeds)", "EM", rep_vals("em"), percent=True)
    add_box("box_tokens_q.png", "Tokens per question (all model calls)", "tokens",
            pq("tokens_total"))
    add_box("box_time_q.png", "Time per question (s)", "seconds", pq("elapsed_s"))
    add_box("box_steps_q.png", "Steps used per question (mean over seeds)", "steps", pq("steps"))
    gp = out / "gpu_memory.png"
    try:
        gpu_plot(gp, exp_dir)
        plots.append(("GPU memory over the experiment", gp))
    except Exception as exc:  # noqa: BLE001
        log.warning(f"gpu plot failed: {exc}")

    config = {}
    ej = exp_dir / "experiment.json"
    if ej.exists():
        config = (json.loads(ej.read_text()).get("config") or {})
    config["n_questions"] = len(per_question_matrix(runs, "m1", "f1")) or config.get("limit") or 100
    judge_summary = {}
    js = exp_dir / "judge" / "summary.json"
    if js.exists():
        judge_summary = json.loads(js.read_text())

    write_json(out / "stats.json", {
        "per_rep": reps, "per_arm": summaries, "tests": tests, "config": config,
    })
    build_report(exp_dir, out / "report.html", summaries, reps, tests, plots, config, judge_summary)
    log.info(f"analysis written to {out} (report.html + {len(plots)} plots)")
    print(json.dumps({a: {k: summaries[a][k]["mean"] for k in ("judge_correct", "em", "f1", "cover")
                          if k in summaries[a]} for a in summaries}, indent=2))


if __name__ == "__main__":
    main()
