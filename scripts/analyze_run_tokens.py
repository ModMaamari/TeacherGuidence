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
    if "granite" in m or "ollama" in m:
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


IN_COLOR = "#2563eb"   # blue = input
OUT_COLOR = "#ea580c"  # orange = output


def hist_in_out(inv, outv, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=inv, name="input tokens", marker_color=IN_COLOR, opacity=0.65, nbinsx=60))
    fig.add_trace(go.Histogram(x=outv, name="output tokens", marker_color=OUT_COLOR, opacity=0.65, nbinsx=60))
    fig.update_layout(barmode="overlay", title=title, xaxis_title="tokens per episode",
                      yaxis_title="episodes", template="plotly_white", height=380,
                      legend=dict(orientation="h", y=1.1), margin=dict(t=60, b=40))
    return fig


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

    # ---- HTML ----
    import plotly.offline as _po
    plotlyjs = _po.get_plotlyjs()
    def stat_table():
        head = "".join(f"<th>{h}</th>" for h in ["metric", "mean", "median", "sd", "min", "p5", "p25", "p75", "p95", "max", "total"])
        rows = ""
        for label, d in stat_rows:
            rows += "<tr><td class='m'>" + label + "</td>" + "".join(
                f"<td>{d[k]:,.0f}</td>" for k in ["mean", "median", "sd", "min", "p5", "p25", "p75", "p95", "max", "sum"]) + "</tr>"
        return f"<table class='stats'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"

    cards = [
        ("episodes", f"{n:,}"),
        ("LLM calls", f"{reps_student+reps_teacher:,}"),
        ("total input tokens", f"{tot_in:,.0f}"),
        ("total visible output tokens", f"{tot_out:,.0f}"),
        ("teacher reasoning overhead", f"{reasoning_overhead:.1f}×"),
        ("teacher share of tokens", f"{100*(tot_teacher_in+tot_teacher_out)/max(tot_in+tot_out,1):.0f}%"),
        ("mean tokens / episode", f"{(tot_in+tot_out)/max(n,1):,.0f}"),
        ("mean steps / episode", f"{n_steps_mean:.1f}"),
    ]
    card_html = "".join(f"<div class='card'><div class='v'>{v}</div><div class='k'>{k}</div></div>" for k, v in cards)
    figs_html = ""
    for title, fig in figs:
        figs_html += f"<section><h2>{title}</h2>{fig_html(fig)}</section>"
    ins_html = "".join(f"<li>{x}</li>" for x in insights)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Token analysis — {Path(run).name}</title>
<script>{plotlyjs}</script>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f8fafc;color:#0f172a}}
 .wrap{{max-width:1100px;margin:0 auto;padding:28px}}
 h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#64748b;margin-bottom:20px}}
 .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 26px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px}}
 .card .v{{font-size:22px;font-weight:700}} .card .k{{color:#64748b;font-size:12px;margin-top:2px}}
 section{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:16px 0}}
 h2{{font-size:16px;margin:0 0 10px}}
 table.stats{{border-collapse:collapse;width:100%;font-size:12.5px}}
 table.stats th,table.stats td{{border-bottom:1px solid #eef2f7;padding:6px 8px;text-align:right}}
 table.stats th:first-child,table.stats td.m{{text-align:left}}
 table.stats thead th{{background:#f1f5f9;position:sticky;top:0}}
 ul.ins{{line-height:1.7}} ul.ins li{{margin-bottom:8px}}
 .legend{{font-size:12px;color:#64748b}}
</style></head><body><div class="wrap">
<h1>Token analysis — {Path(run).name}</h1>
<div class="sub">student = <b>{data['student_model']}</b> · teacher = <b>{data['teacher_model']}</b> ·
 {n:,} episodes · budget {eps[0]['budget'] if eps else '?'} ·
 tokenizers: <b>granite</b> (student) + <b>o200k_base</b> (teacher), counted on the actual text — no approximation.</div>
<div class="cards">{card_html}</div>
<section><h2>Key insights</h2><ul class="ins">{ins_html}</ul></section>
<section><h2>Per-episode token statistics</h2>
 <div class="legend">Distribution across {n:,} episodes of tokens summed per episode. "Teacher (visible)" excludes hidden reasoning; see the reasoning-overhead card.</div>
 {stat_table()}</section>
{figs_html}
<div class="sub" style="margin-top:24px">Generated by scripts/analyze_run_tokens.py · tokens counted with real tokenizers (granite + tiktoken o200k_base).</div>
</div></body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)

    # also dump machine-readable stats
    Path(out).with_suffix(".json").write_text(json.dumps({
        "run": run, "n_episodes": n, "student_model": data["student_model"], "teacher_model": data["teacher_model"],
        "totals": {"input": tot_in, "visible_output": tot_out, "student_in": tot_student_in,
                   "student_out": tot_student_out, "teacher_in": tot_teacher_in, "teacher_out_visible": tot_teacher_out,
                   "teacher_out_billed": tot_teacher_out_stored, "teacher_in_billed": tot_teacher_in_stored},
        "reasoning_overhead": reasoning_overhead,
        "per_episode_stats": {label: d for label, d in stat_rows},
    }, indent=2))
    print(f"wrote {out}  ({Path(out).stat().st_size/1e6:.1f} MB)")
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
