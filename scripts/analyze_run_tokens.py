"""Token analysis + interactive HTML report for a teacher-guidance trace run.

Counts input/output tokens for EVERY LLM call in every episode using REAL tokenizers
(no approximation): the granite tokenizer for the student (ollama/granite4.1:3b) and the
o200k_base tokenizer for the gpt-oss-120b teacher. It walks the actual call records
(``*_calls`` lists), so repair attempts, plan-review rounds and wiki updates are all
included, and routes each call to the right tokenizer by its ``model`` field.

Two token views are reported because they differ meaningfully:
  * tokenized (from the stored prompt / response_text) -- what the *text* contains;
  * provider-reported ``usage`` -- the true generation. For the teacher this is much
    larger than the visible output because gpt-oss-120b's hidden chain-of-thought
    ("reasoning") tokens are billed but not saved in response_text. (Student usage is 0 --
    Ollama didn't report it -- which is exactly why real tokenization is required.)

Writes a self-contained interactive Plotly report (plotly.js inlined, opens offline).

Usage:
    .venv_train/bin/python scripts/analyze_run_tokens.py \
        --run data/simulation_output/traces_g3b_3000 \
        --out reports/token_analysis_g3b_3000/report.html
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from transformers import AutoTokenizer
import tiktoken

GRAN = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-3b")
O200K = tiktoken.get_encoding("o200k_base")


def _role(model: str) -> str:
    m = (model or "").lower()
    # Locally-served students: ollama/<model> or vllm/<served-name> (serve_vllm.sh serves
    # the student under the literal name "student", so the model id carries no family).
    if "granite" in m or "ollama" in m or m.startswith("vllm/"):
        return "student"
    if "gpt-oss" in m or "gpt_oss" in m:
        return "teacher"
    return "student" if "qwen" in m or "minicpm" in m else "teacher"


def _tok(texts_student: List[str], texts_teacher: List[str]):
    s = [len(x) for x in GRAN(texts_student, add_special_tokens=False)["input_ids"]] if texts_student else []
    t = [len(x) for x in O200K.encode_batch(texts_teacher)] if texts_teacher else []
    return s, t


def _iter_calls(ep: Dict[str, Any]):
    """Yield (step_index, role, call_dict) for every LLM call in the episode.
    step 0 == plan stage; 1..N == tool-use steps."""
    pr = ep.get("plan_review") or {}
    for key in ("initial_plan_calls", "plan_calls"):
        for c in (pr.get(key) or []):
            yield 0, _role(c.get("model")), c
    for rnd in (pr.get("rounds") or []):
        for key in ("review_calls", "revision_calls"):
            for c in (rnd.get(key) or []):
                yield 0, _role(c.get("model")), c
    for s in (ep.get("steps") or []):
        t = s.get("t", 0)
        for c in (s.get("student_calls") or []):
            yield t, _role(c.get("model")), c
        for c in (s.get("teacher_calls") or []):
            yield t, _role(c.get("model")), c
        w = s.get("wiki_update_call")
        if isinstance(w, dict) and w.get("prompt") is not None:
            yield t, _role(w.get("model")), w


def analyze(run: str) -> Dict[str, Any]:
    files = sorted(glob.glob(str(Path(run) / "run" / "hotpot_questions" / "sample_*" /
                                 "teacher_guidance_episodes.jsonl")))
    episodes: List[Dict[str, Any]] = []       # per-episode aggregates
    per_step = {}                             # (role, step) -> list of (in,out) tokens
    calls_flat = []                           # per-call: (role, step, in_tok, out_tok, n_attempts, stored_in, stored_out)
    student_model = teacher_model = None

    for f in files:
        line = open(f).readline()
        if not line.strip():
            continue
        ep = json.loads(line)
        student_model = student_model or ep.get("student_model")
        teacher_model = teacher_model or ep.get("teacher_model")

        # collect this episode's call texts, batch-tokenize by role
        s_in_txt, s_out_txt, s_meta = [], [], []
        t_in_txt, t_out_txt, t_meta = [], [], []
        for step, role, c in _iter_calls(ep):
            pin = c.get("prompt") or ""
            pout = c.get("response_text") or ""
            u = c.get("usage") or {}
            si, so = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
            if role == "student":
                s_in_txt.append(pin); s_out_txt.append(pout); s_meta.append((step, si, so))
            else:
                t_in_txt.append(pin); t_out_txt.append(pout); t_meta.append((step, si, so))
        s_in, _ = _tok(s_in_txt, [])
        s_out, _ = _tok(s_out_txt, [])
        _, t_in = _tok([], t_in_txt)
        _, t_out = _tok([], t_out_txt)

        agg = {"qid": ep.get("qid"), "used_steps": ep.get("used_steps") or len(ep.get("steps") or []),
               "budget": ep.get("budget"),
               "correct": bool((ep.get("final_metrics") or {}).get("cover_match")),
               "student_in": 0, "student_out": 0, "teacher_in": 0, "teacher_out": 0,
               "teacher_in_stored": 0, "teacher_out_stored": 0,
               "n_student_calls": len(s_meta), "n_teacher_calls": len(t_meta)}
        for (step, si, so), it, ot in zip(s_meta, s_in, s_out):
            agg["student_in"] += it; agg["student_out"] += ot
            per_step.setdefault(("student", step), []).append((it, ot))
            calls_flat.append(("student", step, it, ot, si, so))
        for (step, si, so), it, ot in zip(t_meta, t_in, t_out):
            agg["teacher_in"] += it; agg["teacher_out"] += ot
            agg["teacher_in_stored"] += si; agg["teacher_out_stored"] += so
            per_step.setdefault(("teacher", step), []).append((it, ot))
            calls_flat.append(("teacher", step, it, ot, si, so))
        agg["total_in"] = agg["student_in"] + agg["teacher_in"]
        agg["total_out"] = agg["student_out"] + agg["teacher_out"]
        episodes.append(agg)

    return {"episodes": episodes, "per_step": per_step, "calls": calls_flat,
            "student_model": student_model, "teacher_model": teacher_model,
            "n_files": len(files)}


def dist(vals: List[float]) -> Dict[str, float]:
    a = np.array(vals, dtype=float)
    if a.size == 0:
        return {k: 0 for k in ("mean", "median", "sd", "min", "p5", "p25", "p75", "p95", "max", "sum")}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "sd": float(a.std()),
            "min": float(a.min()), "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "sum": float(a.sum())}


def fig_html(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False})


IN_COLOR = "#3b82f6"   # blue = input
OUT_COLOR = "#f59e0b"  # amber = output
_FONT = dict(family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
             color="#334155", size=12)


def _style(fig, title, height=380):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#0f172a")),
                      template="plotly_white", height=height, font=_FONT,
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      margin=dict(t=58, b=44, l=56, r=24),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


def hist_in_out(inv, outv, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=inv, name="input tokens", marker_color=IN_COLOR, opacity=0.62, nbinsx=60))
    fig.add_trace(go.Histogram(x=outv, name="output tokens", marker_color=OUT_COLOR, opacity=0.62, nbinsx=60))
    _style(fig, title)
    fig.update_layout(barmode="overlay", xaxis_title="tokens per episode", yaxis_title="episodes")
    return fig


_REPORT_CSS = """
:root{
  --ground:#eef1f6;--surface:#ffffff;--ink:#131a2b;--muted:#5b6678;--hair:#dfe4ec;--chip:#eef3f9;
  --chart:#ffffff;
  --in:#3b82f6;--out:#f59e0b;--billed:#ef4444;--good:#16a34a;--student:#0d9488;--teacher:#7c3aed;
  --mono:ui-monospace,SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0b0f17;--surface:#141b26;--ink:#e7edf6;--muted:#8b96a9;--hair:#232c3a;--chip:#1a2330;
}}
:root[data-theme="light"]{--ground:#eef1f6;--surface:#fff;--ink:#131a2b;--muted:#5b6678;--hair:#dfe4ec;--chip:#eef3f9;}
:root[data-theme="dark"]{--ground:#0b0f17;--surface:#141b26;--ink:#e7edf6;--muted:#8b96a9;--hair:#232c3a;--chip:#1a2330;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:34px 22px 60px}
.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-size:27px;font-weight:700;letter-spacing:-.015em;margin:7px 0 8px;text-wrap:balance}
.meta{color:var(--muted);font-size:13px;margin-bottom:26px}
.meta b{color:var(--ink);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin-bottom:24px}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--surface);border:1px solid var(--hair);border-radius:14px;padding:15px 16px}
.kpi .v{font-family:var(--mono);font-size:23px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi .k{font-size:12.5px;color:var(--ink);margin-top:4px}
.kpi .note{font-size:10px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.04em}
section{background:var(--surface);border:1px solid var(--hair);border-radius:16px;padding:20px;margin:16px 0}
h2{font-size:15px;font-weight:650;margin:0 0 4px;letter-spacing:-.01em}
.hint{font-size:12.5px;color:var(--muted);margin:2px 0 13px}
.hint i{color:var(--ink);font-style:italic}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:middle;margin:0 3px 0 6px}
.chartbox{background:var(--chart);border:1px solid var(--hair);border-radius:12px;padding:6px 6px 2px;overflow:hidden;margin-top:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px}
.grid2 .chartbox{margin-top:0}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
ul.ins{margin:8px 0 0;padding-left:18px;line-height:1.65}
ul.ins li{margin-bottom:11px}
ul.ins b{color:var(--ink)}
code{font-family:var(--mono);font-size:.86em;background:var(--chip);padding:1px 6px;border-radius:5px}
.tablewrap{overflow-x:auto;border:1px solid var(--hair);border-radius:11px;margin-top:4px}
table.stats{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;font-family:var(--mono)}
table.stats th,table.stats td{border-bottom:1px solid var(--hair);padding:8px 11px;text-align:right;white-space:nowrap}
table.stats tbody tr:last-child td{border-bottom:none}
table.stats th{font-family:var(--sans);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:var(--chip)}
table.stats th:first-child,table.stats td.m{text-align:left;font-family:var(--sans);font-weight:500}
table.stats tr.r-in td.m{border-left:3px solid var(--in)}
table.stats tr.r-out td.m{border-left:3px solid var(--out)}
.foot{color:var(--muted);font-size:12px;margin-top:26px;line-height:1.6}
"""


def build_report(data: Dict[str, Any], run: str, out: str) -> None:
    eps = data["episodes"]
    n = len(eps)
    get = lambda k: [e[k] for e in eps]

    # ---- overall per-episode distributions ----
    cards_defs = [
        ("Student", "student_in", "student_out"),
        ("Teacher (visible)", "teacher_in", "teacher_out"),
        ("Overall", "total_in", "total_out"),
    ]
    stat_rows = []
    for label, ink, outk in cards_defs:
        di, do = dist(get(ink)), dist(get(outk))
        stat_rows.append((label + " input", di))
        stat_rows.append((label + " output", do))

    # totals
    tot_student_in = sum(get("student_in")); tot_student_out = sum(get("student_out"))
    tot_teacher_in = sum(get("teacher_in")); tot_teacher_out = sum(get("teacher_out"))
    tot_teacher_out_stored = sum(get("teacher_out_stored"))
    tot_teacher_in_stored = sum(get("teacher_in_stored"))
    tot_in = tot_student_in + tot_teacher_in
    tot_out = tot_student_out + tot_teacher_out
    reasoning_overhead = tot_teacher_out_stored / tot_teacher_out if tot_teacher_out else 0

    # ---- figures ----
    figs = []
    figs.append(("Student — input vs output tokens per episode",
                 hist_in_out(get("student_in"), get("student_out"), "Student (granite4.1:3b) tokens/episode")))
    figs.append(("Teacher — input vs output tokens per episode",
                 hist_in_out(get("teacher_in"), get("teacher_out"), "Teacher (gpt-oss-120b, visible) tokens/episode")))
    figs.append(("Overall — input vs output tokens per episode",
                 hist_in_out(get("total_in"), get("total_out"), "Overall (student + teacher) tokens/episode")))

    # per-step breakdown (mean tokens by step index, role)
    ps = data["per_step"]
    steps = sorted({k[1] for k in ps})
    def mean_at(role, idx, step):
        v = ps.get((role, step)) or []
        return float(np.mean([x[idx] for x in v])) if v else 0
    fig_step = go.Figure()
    fig_step.add_trace(go.Bar(x=steps, y=[mean_at("student", 0, s) for s in steps], name="student input", marker_color="#93c5fd"))
    fig_step.add_trace(go.Bar(x=steps, y=[mean_at("student", 1, s) for s in steps], name="student output", marker_color="#fdba74"))
    fig_step.add_trace(go.Bar(x=steps, y=[mean_at("teacher", 0, s) for s in steps], name="teacher input", marker_color="#2563eb"))
    fig_step.add_trace(go.Bar(x=steps, y=[mean_at("teacher", 1, s) for s in steps], name="teacher output", marker_color="#ea580c"))
    fig_step.update_layout(barmode="group", title="Mean tokens per call by step (0 = plan stage)",
                           xaxis_title="step index", yaxis_title="mean tokens", template="plotly_white",
                           height=420, legend=dict(orientation="h", y=1.12), margin=dict(t=60))
    figs.append(("Tokens by step — context growth", fig_step))

    # teacher vs student share
    fig_share = go.Figure(go.Bar(
        x=["input tokens", "output tokens (visible)", "output tokens (billed, incl. reasoning)"],
        y=[tot_teacher_in/(tot_teacher_in+tot_student_in)*100,
           tot_teacher_out/(tot_teacher_out+tot_student_out)*100,
           tot_teacher_out_stored/(tot_teacher_out_stored+tot_student_out)*100],
        marker_color=["#2563eb", "#ea580c", "#b91c1c"], text=None))
    fig_share.update_layout(title="Teacher's share of total tokens (%)", yaxis_title="% of tokens from teacher",
                            template="plotly_white", height=360, yaxis_range=[0, 100], margin=dict(t=60))
    figs.append(("Who spends the tokens", fig_share))

    # tokens vs correctness
    corr = [e["total_in"]+e["total_out"] for e in eps if e["correct"]]
    inc = [e["total_in"]+e["total_out"] for e in eps if not e["correct"]]
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Box(y=corr, name=f"correct (n={len(corr)})", marker_color="#16a34a"))
    fig_corr.add_trace(go.Box(y=inc, name=f"incorrect (n={len(inc)})", marker_color="#dc2626"))
    fig_corr.update_layout(title="Total tokens/episode by outcome", yaxis_title="total tokens (in+out)",
                           template="plotly_white", height=380, margin=dict(t=60))
    figs.append(("Does spending more tokens help?", fig_corr))

    # tokens vs steps
    fig_sc = go.Figure(go.Scatter(x=get("used_steps"), y=[e["total_in"]+e["total_out"] for e in eps],
                                  mode="markers", marker=dict(size=4, color="#7c3aed", opacity=0.35)))
    fig_sc.update_layout(title="Total tokens vs steps used", xaxis_title="steps used",
                         yaxis_title="total tokens/episode", template="plotly_white", height=380, margin=dict(t=60))
    figs.append(("Tokens scale with trajectory length", fig_sc))

    # ---- insights ----
    reps_student = sum(1 for c in data["calls"] if c[0] == "student")
    reps_teacher = sum(1 for c in data["calls"] if c[0] == "teacher")
    n_steps_mean = np.mean(get("used_steps"))
    teacher_billed = tot_teacher_in_stored + tot_teacher_out_stored
    student_total = tot_student_in + tot_student_out
    teacher_share_billed = teacher_billed / max(teacher_billed + student_total, 1)
    insights = [
        f"<b>Visible tokens split almost 50/50 — hidden reasoning tips it to the teacher.</b> "
        f"Counted from the text, student and teacher use near-equal tokens "
        f"(input {tot_student_in/1e6:.0f}M vs {tot_teacher_in/1e6:.0f}M; visible output "
        f"{tot_student_out/1e6:.1f}M vs {tot_teacher_out/1e6:.1f}M). But once gpt-oss-120b's hidden "
        f"chain-of-thought is billed, the teacher accounts for <b>{teacher_share_billed:.0%}</b> of all tokens.",
        f"<b>Hidden reasoning is the biggest invisible cost.</b> The teacher's billed output is "
        f"<b>{reasoning_overhead:.1f}×</b> its visible JSON output "
        f"({tot_teacher_out_stored:,.0f} billed vs {tot_teacher_out:,.0f} visible tokens) — "
        f"the rest is gpt-oss-120b's chain-of-thought, paid for but not saved in the traces. "
        f"The student (granite) emits no hidden reasoning, so its tokenized output is its true output.",
        f"<b>Input tokens grow with the trajectory.</b> Both student and teacher prompts accumulate context "
        f"across the {int(n_steps_mean)}-step average episode (see the per-step chart) — later steps cost far "
        f"more input than early ones.",
        f"<b>Student token counts had to be recovered by tokenization</b> — the stored provider <code>usage</code> for the "
        f"Ollama student was 0 for every call, so the ~{tot_student_in+tot_student_out:,.0f} student tokens here "
        f"exist only because we re-tokenized the text.",
        f"<b>Calls include repairs.</b> {reps_student:,} student and {reps_teacher:,} teacher LLM calls across "
        f"{n} episodes (some steps needed a repair retry, each re-sending the prompt).",
    ]

    # ---- HTML (theme-aware; also emits a body-only file for the Claude Artifact host) ----
    import plotly.offline as _po
    plotlyjs = _po.get_plotlyjs()

    def stat_table():
        head = "".join(f"<th>{h}</th>" for h in
                       ["metric", "mean", "median", "sd", "min", "p5", "p25", "p75", "p95", "max", "total"])
        rows = ""
        for label, d in stat_rows:
            role = "in" if "input" in label else "out"
            rows += f"<tr class='r-{role}'><td class='m'>{label}</td>" + "".join(
                f"<td>{d[k]:,.0f}</td>" for k in
                ["mean", "median", "sd", "min", "p5", "p25", "p75", "p95", "max", "sum"]) + "</tr>"
        return (f"<div class='tablewrap'><table class='stats'><thead><tr>{head}</tr></thead>"
                f"<tbody>{rows}</tbody></table></div>")

    cards = [
        ("episodes", f"{n:,}", ""),
        ("LLM calls", f"{reps_student+reps_teacher:,}", "incl. repair retries"),
        ("input tokens", f"{tot_in/1e6:.1f}M", "tokenized"),
        ("visible output", f"{tot_out/1e6:.1f}M", "tokenized"),
        ("reasoning overhead", f"{reasoning_overhead:.1f}×", "teacher billed / visible out"),
        ("teacher token share", f"{teacher_share_billed:.0%}", "of billed total"),
        ("tokens / episode", f"{(tot_in+tot_out)/max(n,1):,.0f}", "mean, visible"),
        ("steps / episode", f"{n_steps_mean:.1f}", "mean"),
    ]
    kpi_html = ""
    for k, v, note in cards:
        note_html = f"<div class='note'>{note}</div>" if note else ""
        kpi_html += f"<div class='kpi'><div class='v'>{v}</div><div class='k'>{k}</div>{note_html}</div>"

    def chartbox(fig):
        return f"<div class='chartbox'>{fig_html(fig)}</div>"

    hist = figs[:3]           # student, teacher, overall
    rest = figs[3:]           # per-step, share, correctness, scatter
    hist_section = (
        "<section><h2>Input vs output tokens per episode</h2>"
        f"<p class='hint'>Distribution across {n:,} episodes of tokens summed per episode "
        "(<span class='sw' style='background:var(--in)'></span>input, "
        "<span class='sw' style='background:var(--out)'></span>output). Teacher counts are the "
        "<i>visible</i> output; hidden reasoning is billed on top (see the overhead card).</p>"
        f"<div class='grid2'>{chartbox(hist[0][1])}{chartbox(hist[1][1])}</div>"
        f"{chartbox(hist[2][1])}</section>"
    )
    rest_html = "".join(
        f"<section><h2>{title}</h2>{chartbox(fig)}</section>" for title, fig in rest)
    ins_html = "".join(f"<li>{x}</li>" for x in insights)

    body = f"""
<div class="eyebrow">Trace-run token economics</div>
<h1>Token analysis — {Path(run).name}</h1>
<div class="meta">student <b>{data['student_model']}</b> &nbsp;&middot;&nbsp; teacher <b>{data['teacher_model']}</b>
 &nbsp;&middot;&nbsp; {n:,} episodes &nbsp;&middot;&nbsp; budget {eps[0]['budget'] if eps else '?'}
 &nbsp;&middot;&nbsp; counted with real tokenizers (<b>granite</b> student, <b>o200k_base</b> teacher) &mdash; no approximation</div>
<div class="kpis">{kpi_html}</div>
<section><h2>What the numbers say</h2><ul class="ins">{ins_html}</ul></section>
{hist_section}
{rest_html}
<section><h2>Per-episode token statistics</h2>
 <p class="hint">Distribution across {n:,} episodes of tokens summed per episode. Teacher rows are visible output only.</p>
 {stat_table()}</section>
<div class="foot">Generated by <code>scripts/analyze_run_tokens.py</code> &middot; every LLM call re-tokenized with granite (student) + tiktoken o200k_base (teacher). Provider <code>usage</code> is used only to expose the teacher's hidden-reasoning tokens.</div>
"""

    title = f"Token analysis — {Path(run).name}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    standalone = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                  f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                  f"<title>{title}</title><style>{_REPORT_CSS}</style><script>{plotlyjs}</script></head>"
                  f"<body><div class='wrap'>{body}</div></body></html>")
    Path(out).write_text(standalone)

    # body-only file for the Artifact host (it wraps in <!doctype><head><body>)
    art_path = Path(str(out).replace(".html", ".artifact.html"))
    artifact = (f"<title>{title}</title>\n<style>{_REPORT_CSS}</style>\n"
                f"<script>{plotlyjs}</script>\n<div class='wrap'>{body}</div>")
    art_path.write_text(artifact)

    Path(out).with_suffix(".json").write_text(json.dumps({
        "run": run, "n_episodes": n, "student_model": data["student_model"], "teacher_model": data["teacher_model"],
        "totals": {"input": tot_in, "visible_output": tot_out, "student_in": tot_student_in,
                   "student_out": tot_student_out, "teacher_in": tot_teacher_in, "teacher_out_visible": tot_teacher_out,
                   "teacher_out_billed": tot_teacher_out_stored, "teacher_in_billed": tot_teacher_in_stored},
        "reasoning_overhead": reasoning_overhead, "teacher_share_billed": teacher_share_billed,
        "per_episode_stats": {label: d for label, d in stat_rows},
    }, indent=2))
    print(f"wrote {out}  ({Path(out).stat().st_size/1e6:.1f} MB)")
    print(f"wrote {art_path}  (body-only, for Artifact)")
    print(f"wrote {Path(out).with_suffix('.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="reports/token_analysis/report.html")
    args = ap.parse_args()
    data = analyze(args.run)
    build_report(data, args.run, args.out)


if __name__ == "__main__":
    main()
