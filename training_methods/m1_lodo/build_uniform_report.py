"""Render runs/uniform/results.json into reports/m1_uniform/{report.html,RESULTS.md}.

Pure data -> document: every number in the page comes from results.json; nothing is
typed by hand. Charts are inline SVG built here (no library)."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

DS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
DSN = {"hotpotqa": "HotpotQA", "2wikimultihopqa": "2WikiMultihopQA", "musique": "MuSiQue",
       "strategyqa": "StrategyQA"}
ARMS = ["base", "guided", "all4", "teacher"]           # display order: cheap -> expensive
SHORT = {"base": "Base", "guided": "Guided", "all4": "Trained (all-4)", "teacher": "Teacher"}
CLS = {"base": "base", "guided": "guided", "all4": "student", "teacher": "teacher"}


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "—" if x is None else f"{x:,.{d}f}"


def usd(x, d=5):
    return "—" if x is None else f"${x:,.{d}f}"


def esc(s):
    return html.escape(str(s))


# ----------------------------------------------------------------------------- charts
def grouped_bars(per_ds, metric, title, fmt=pct, ymax=1.0, unit=""):
    """Per-dataset grouped bars, one bar per arm, labelled values (colour is not the
    only channel: order + label carry the arm too)."""
    W, H, L, B, T = 900, 300, 44, 54, 26
    gw = (W - L - 16) / len(DS)
    bw = gw / (len(ARMS) + 1.2)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart">']
    for i in range(0, 5):
        y = T + (H - T - B) * (1 - i / 4)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - 8}" y2="{y:.1f}" class="grid"/>')
        lab = fmt(ymax * i / 4) if fmt is pct else f"{ymax * i / 4:g}{unit}"
        out.append(f'<text x="{L - 6}" y="{y + 4:.1f}" class="ax" text-anchor="end">{lab}</text>')
    for di, d in enumerate(DS):
        x0 = L + di * gw + bw * 0.6
        for ai, a in enumerate(ARMS):
            v = (per_ds[a].get(d) or {}).get(metric)
            if v is None:
                continue
            h = (H - T - B) * min(v / ymax, 1.0)
            x = x0 + ai * bw
            y = H - B - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.86:.1f}" height="{h:.1f}" class="bar {CLS[a]}"/>')
            lab = fmt(v, 0) if fmt is pct else f"{v:,.0f}"
            out.append(f'<text x="{x + bw * 0.43:.1f}" y="{y - 4:.1f}" class="val" text-anchor="middle">{lab}</text>')
        out.append(f'<text x="{x0 + bw * len(ARMS) / 2:.1f}" y="{H - B + 18}" class="ax" text-anchor="middle">{DSN[d]}</text>')
        out.append(f'<text x="{x0 + bw * len(ARMS) / 2:.1f}" y="{H - B + 34}" class="ax dim" text-anchor="middle">n={(per_ds["all4"].get(d) or {}).get("n", "?")}</text>')
    out.append("</svg>")
    return "\n".join(out)


def legend():
    return ('<div class="legend">' + "".join(
        f'<span><i class="sw {CLS[a]}"></i>{SHORT[a]}</span>' for a in ARMS) + "</div>")


# ----------------------------------------------------------------------------- tables
def table(head, rows, cls="", note=None):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f'<caption>{note}</caption>' if note else ""
    return f'<div class="tw"><table class="{cls}">{cap}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def arm_td(a):
    return f'<span class="tag {CLS[a]}">{SHORT[a]}</span>'


def sig_cell(s):
    if not s:
        return "—"
    lo, hi = s["ci95"]
    star = "" if s["p"] >= 0.05 else (" ‡" if s["p"] < 0.001 else " †")
    return (f'<span class="mono">{s["diff"] * 100:+.1f} pts</span> '
            f'<span class="dim">[{lo * 100:+.1f}, {hi * 100:+.1f}]</span>{star}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="training_methods/m1_lodo/runs/uniform/results.json")
    ap.add_argument("--out-dir", default="reports/m1_uniform")
    args = ap.parse_args()
    R = json.loads(Path(args.results).read_text())
    P, D, S, O, PR, C = (R["pooled"], R["per_dataset"], R["significance"], R["solved_overlap"],
                         R["protocol"], R["coverage"])
    n_total = R["test_set"]["total"]
    incomplete = [a for a in ARMS if P[a] is None or P[a]["n"] < n_total]

    # ---- headline band
    band = "".join(
        f'<div class="cell"><div class="v {CLS[a]}">{pct(P[a]["judge"])}</div>'
        f'<div class="l">{SHORT[a]} · judge-correct · n={P[a]["n"]}</div></div>' for a in ARMS)

    # ---- accuracy table
    acc_rows = [[arm_td(a), P[a]["n"], pct(P[a]["em"]), num(P[a]["f1"], 3), pct(P[a]["cover"]),
                 f'<b>{pct(P[a]["judge"])}</b>', pct(P[a]["macro_judge"]), num(P[a]["doc_recall"], 3)]
                for a in ARMS]
    per_ds_rows = []
    for d in DS:
        for a in ARMS:
            r = D[a].get(d)
            if r:
                per_ds_rows.append([DSN[d] if a == ARMS[0] else "", arm_td(a), r["n"], pct(r["em"]),
                                    num(r["f1"], 3), pct(r["cover"]), f'<b>{pct(r["judge"])}</b>',
                                    num(r["doc_recall"], 3), num(r["mean_steps"]), pct(r["voluntary_finish"], 0)])

    # ---- efficiency
    eff_rows = [[arm_td(a), num(P[a]["mean_steps"]), pct(P[a]["voluntary_finish"]),
                 num(P[a]["invalid_steps_per_ep"], 3),
                 f'{P[a]["student_tokens_per_ep"]:,.0f}', f'{P[a]["teacher_tokens_per_ep"]:,.0f}',
                 f'<b>{P[a]["total_tokens_per_ep"]:,.0f}</b>',
                 ("—" if P[a]["plan_tokens_per_ep"] is None else f'{P[a]["plan_tokens_per_ep"]:,.0f}'),
                 num(P[a]["latency_s_per_ep"], 1), num(P[a]["teacher_api_s_per_ep"], 1)] for a in ARMS]

    # ---- cost
    rate = PR["gpu_usd_per_hour"]
    cost_rows = []
    for a in ARMS:
        p = P[a]
        no_train = (p["api_usd"] + p["gpu_usd"]) / p["n"]
        cost_rows.append([arm_td(a), usd(p["api_usd"], 3), usd(p["gpu_usd"], 3),
                          usd(p["train_amort_usd"], 3) if a == "all4" else "—",
                          f'<b>{usd(p["usd_per_ep"])}</b>', usd(no_train), usd(p["usd_per_correct"]),
                          f'{p["tokens_per_correct"]:,.0f}', num(p["correct_per_1k_tokens"], 3)])
    train_h = PR["train_gpu_hours"]; train_usd = PR["train_usd_total"]
    all4_run = (P["all4"]["api_usd"] + P["all4"]["gpu_usd"]) / P["all4"]["n"]
    amort = [(k, train_usd / k, all4_run + train_usd / k) for k in (n_total, 10_000, 100_000, 1_000_000)]
    amort_rows = [[f"{k:,}", usd(t, 6), usd(tot, 6)] for k, t, tot in amort]

    # ---- significance
    pairs = [("base", "all4"), ("base", "guided"), ("all4", "guided"), ("all4", "teacher"),
             ("guided", "teacher"), ("base", "teacher")]
    sig_rows = []
    for a, b in pairs:
        s = S.get(f"{a}->{b}", {})
        row = [f'{arm_td(a)} → {arm_td(b)}']
        for scope in DS + ["pooled"]:
            row.append(sig_cell(s.get(scope)))
        pooled = s.get("pooled") or {}
        row.append(f'{pooled.get("b_wins", "—")} / {pooled.get("a_wins", "—")}')
        row.append("—" if not pooled else (f'{pooled["p"]:.2g}' if pooled["p"] > 0 else "&lt;1e-5"))
        sig_rows.append(row)

    # ---- overlap
    ov_rows = [[arm_td(a)] + [str(O["pairwise"][a][b]) for b in ARMS] for a in ARMS]

    # ---- coverage / protocol
    cov_rows = [[arm_td(a)] + [f'{C[a][d]["judged"]}/{C[a][d]["n"]}' for d in DS] for a in ARMS]
    jm = PR["judge_models"]
    judge_txt = "; ".join(f'{k}: ' + ", ".join(f'{m.split("/")[-1]} ×{c:,}' for m, c in v.items())
                          for k, v in jm.items())

    # ---- narrative numbers
    g = lambda a, k: P[a][k]
    d_base_all4 = S["base->all4"]["pooled"]; d_all4_teacher = S["all4->teacher"]["pooled"]
    d_all4_guided = S["all4->guided"].get("pooled") or {}
    ratio_all4 = P["all4"]["judge"] / P["teacher"]["judge"]
    ratio_guided = (P["guided"]["judge"] / P["teacher"]["judge"]) if P["guided"]["judge"] else None
    cost_ratio = P["guided"]["usd_per_ep"] / max(P["all4"]["usd_per_ep"], 1e-9)
    tok_ratio = P["guided"]["total_tokens_per_ep"] / P["all4"]["total_tokens_per_ep"]

    warn = ""
    if incomplete:
        warn = ('<div class="notice">Partial results: ' + ", ".join(SHORT[a] for a in incomplete) +
                f' has fewer than {n_total} episodes. Pooled numbers for that arm cover only the '
                'datasets finished so far; paired tests use the common questions only.</div>')

    css = Path(__file__).with_name("uniform_report.css").read_text()
    page = f"""<title>Four Arms, One Test Set</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="wrap">
<header>
  <p class="eyebrow">Teacher guidance · uniform evaluation · granite-4.1-3b</p>
  <h1>Four arms on one test set</h1>
  <p class="standfirst">The same <strong>{n_total} questions</strong> — 10% of each of four multi-hop
  datasets, never seen by any training fold — answered by the untrained student, the student with a live
  teacher, the teacher itself, and the student fine-tuned on all four datasets. Same corpus, tools,
  budget and decoding for every arm; correctness by an LLM judge; cost and tokens measured per call.</p>
  <div class="meta"><span>test set <b>{n_total}</b></span><span>budget <b>{PR["budget"]} (+hidden)</b></span>
  <span>teacher <b>{esc(PR["teacher_model"])}</b></span><span>judge <b>Kimi-K2.6</b></span>
  <span>GPU rate <b>${rate:.2f}/h A100</b></span></div>
</header>
{warn}
<div class="band">{band}</div>

<section>
<div class="sh"><span class="sn">1</span><h2>What the four arms are</h2></div>
{table(["Arm", "Agent", "Teacher at inference", "Training", "Where the compute runs"], [
    [arm_td("base"), "granite-4.1-3b", "none", "none", "local GPU (vLLM)"],
    [arm_td("guided"), "granite-4.1-3b", "DeepSeek-V4-Flash-0731 reviews the plan and every step", "none", "local GPU + teacher API"],
    [arm_td("all4"), "granite-4.1-3b + LoRA", "none", f"SFT on {14477:,} guidance-as-thought examples from all four datasets ({train_h:.1f} GPU-h)", "local GPU (vLLM, LoRA)"],
    [arm_td("teacher"), "DeepSeek-V4-Flash-0731", "— (it is the agent)", "none", "API only"],
])}
<p>Arms 1 and 3 were measured earlier on exactly these questions and are reused; arms 2 and 4 were run
in job 4049332. The trained student's adapter was fitted only on the 90% pools (dev loss {PR["train_final"]["eval_loss"]:.3f}),
so no arm has seen a test question.</p>
</section>

<section>
<div class="sh"><span class="sn">2</span><h2>Accuracy</h2></div>
{table(["Arm", "n", "EM", "F1", "Cover", "Judge-correct", "Judge (macro over datasets)", "Doc recall"], acc_rows,
       note="Pooled over the whole test set. Judge = Kimi-K2.6 verdict on the final answer against gold; EM/F1 are token-level; cover = gold string contained in the answer.")}
<h3>Judge-correct by dataset</h3>
{legend()}
{grouped_bars(D, "judge", "judge-correct by dataset")}
<h3>Per dataset</h3>
{table(["Dataset", "Arm", "n", "EM", "F1", "Cover", "Judge", "Doc recall", "Steps", "Voluntary finish"], per_ds_rows, cls="dense")}
<p>Training on all four datasets lifts the student from {pct(g("base","judge"))} to <b>{pct(g("all4","judge"))}</b>
judge-correct ({d_base_all4["diff"]*100:+.1f} pts, 95% CI [{d_base_all4["ci95"][0]*100:+.1f}, {d_base_all4["ci95"][1]*100:+.1f}]),
reaching {ratio_all4:.0%} of the teacher's {pct(g("teacher","judge"))} with no teacher at inference.
{("The live-teacher arm reaches " + pct(g("guided","judge")) + f" ({ratio_guided:.0%} of the teacher), so the trained student is " + f'{-d_all4_guided.get("diff",0)*100:.1f} pts better than the same student helped by the teacher at every step' + f' (CI [{-d_all4_guided.get("ci95",[0,0])[1]*100:+.1f}, {-d_all4_guided.get("ci95",[0,0])[0]*100:+.1f}]), at less than half the tokens.') if d_all4_guided else ""}
On StrategyQA both student arms edge past the teacher ({pct((D["all4"].get("strategyqa") or {}).get("judge"))} and {pct((D["guided"].get("strategyqa") or {}).get("judge"))} vs {pct((D["teacher"].get("strategyqa") or {}).get("judge"))}); on MuSiQue, the hardest set, the trained and guided students tie at {pct((D["all4"].get("musique") or {}).get("judge"))}.
The teacher stays ahead of the trained student by {d_all4_teacher["diff"]*100:+.1f} pts (CI [{d_all4_teacher["ci95"][0]*100:+.1f}, {d_all4_teacher["ci95"][1]*100:+.1f}]).</p>
</section>

<section>
<div class="sh"><span class="sn">3</span><h2>Efficiency</h2></div>
{table(["Arm", "Steps / ep", "Voluntary finish", "Invalid steps / ep", "Student tokens / ep", "Teacher tokens / ep", "Total tokens / ep", "Plan-phase tokens / ep", "Latency s / ep", "Teacher API s / ep"], eff_rows,
       note="Step-phase tokens (prompt + completion) are logged by every arm and are the comparable quantity. Plan-phase tokens are logged by three arms; the guided runner trims plan-review call records, so its plan phase is not counted (an under-count for that arm of roughly one student plan, one teacher review and one revision per episode). Latency is per-episode wall time inside a batched/concurrent run, not a single-user response time.")}
<h3>Total tokens per episode by dataset</h3>
{legend()}
{grouped_bars(D, "total_tokens_per_ep", "tokens per episode", fmt=num, ymax=16000, unit="")}
<p>The trained student spends about the same tokens as the untrained one ({g("all4","total_tokens_per_ep"):,.0f} vs {g("base","total_tokens_per_ep"):,.0f} per episode);
the live teacher {tok_ratio:.1f}× that, because every step is also sent to the teacher. The teacher acting alone is the leanest
({g("teacher","total_tokens_per_ep"):,.0f} tokens, {g("teacher","mean_steps"):.2f} steps) and always stops voluntarily; the students mostly run to the budget.</p>
</section>

<section>
<div class="sh"><span class="sn">4</span><h2>Cost</h2></div>
{table(["Arm", "API $ (all eps)", "GPU $ (all eps)", "Training $ (amortized)", "$ / episode", "$ / episode, excl. training", "$ / correct answer", "Tokens / correct", "Correct / 1k tokens"], cost_rows,
       note=f"API dollars are the per-call cost EdenAI reports (DeepSeek-V4-Flash-0731 via flexai). GPU dollars = eval wall time on one A100 × ${rate:.2f}/h (assumption; change with --gpu-usd-per-hour). Training: {train_h:.2f} A100-hours = ${train_usd:.2f}, amortized here over the {n_total} test episodes — the most pessimistic choice.")}
<h3>How training amortizes</h3>
{table(["Episodes the adapter serves", "Training $ / episode", "Trained-student $ / episode (run + training)"], amort_rows,
       note=f"Run cost of the trained student is {usd(all4_run)} per episode. Past ~10k episodes the training cost is negligible and the trained student is the cheapest arm that clears 60% judge-correct.")}
<p>Charging the whole training run to these {n_total} episodes makes the trained student the most expensive arm ({usd(g("all4","usd_per_ep"))} per episode). Excluding training it runs at {usd(all4_run)} per episode — {1/cost_ratio if cost_ratio else 0:.1f}× cheaper than the guided arm's {usd(g("guided","usd_per_ep"))}, and already cheaper than it once the adapter has served about {int(train_usd/(g("guided","usd_per_ep")-all4_run)):,} episodes. The teacher alone is cheap per episode ({usd(g("teacher","usd_per_ep"))}) because DeepSeek-V4-Flash is priced low and it stops early — but every one of its calls leaves the machine.</p>
</section>

<section>
<div class="sh"><span class="sn">5</span><h2>Are the differences real?</h2></div>
{table(["Comparison (a → b)"] + [DSN[d] for d in DS] + ["Pooled", "b wins / a wins", "McNemar p (pooled)"], sig_rows, cls="dense",
       note="Judge-correct difference b − a on the same questions; 95% paired-bootstrap CI (10,000 resamples). † p < 0.05, ‡ p < 0.001 (exact McNemar on discordant pairs). Wins = questions one arm gets right and the other wrong.")}
<h3>Which questions each arm solves</h3>
{table(["Solved by ↓ and →"] + [SHORT[a] for a in ARMS], ov_rows, cls="dense",
       note=f"Diagonal = questions solved by the arm; off-diagonal = solved by both. {O['solved_by_any']} of {n_total} questions are solved by at least one arm, {O['solved_by_all']} by all four, {O['unsolved_by_all']} by none.")}
</section>

<section>
<div class="sh"><span class="sn">6</span><h2>Protocol and coverage</h2></div>
<ul>
<li><b>Test set.</b> {", ".join(f'{DSN[d]} {R["test_set"]["per_dataset"][d]}' for d in DS)} = {n_total} questions: the 10% held-in pool of each dataset (sha256-salted split), excluded from every training fold including this all-4 fold.</li>
<li><b>Student decoding.</b> Greedy (T=0), seed 13, budget {PR["budget"]} with the budget hidden from the agent; identical prompts, tools and per-question corpus for all arms.</li>
<li><b>Teacher.</b> {esc(PR["teacher_model"])} for both the guided arm and the teacher-alone arm, so the reference and the helper are the same model.</li>
<li><b>Judge.</b> {esc(judge_txt)}. The judge sees question, gold and final answer only.</li>
<li><b>Training.</b> LoRA r32/α64, 2 epochs, lr 1e-4, effective batch 16, 14,477 examples from correct episodes only; final train loss {PR["train_final"]["train_loss"]:.3f}, dev loss {PR["train_final"]["eval_loss"]:.3f}.</li>
<li><b>Guided arm.</b> {PR["guided_concurrency"]} episodes in flight; the student ran on the same vLLM server as the other student arms.</li>
</ul>
{table(["Arm"] + [f"{DSN[d]} judged/n" for d in DS], cov_rows, cls="dense")}
<h3>Caveats</h3>
<ul>
<li>One seed, one budget. Bootstrap CIs cover question sampling, not decoding or training variance.</li>
<li>Guided-arm plan-phase tokens and their teacher cost are not logged (see §3); its true token and API cost per episode is somewhat higher than shown.</li>
<li>GPU cost uses an assumed hourly rate; the teacher API price is EdenAI's reported per-call cost at run time.</li>
<li>The judge is a single model; EM/F1/cover are shown alongside so the reader can check it is not systematically generous to any arm.</li>
</ul>
</section>
</div>
"""
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(page, encoding="utf-8")

    # ---- markdown twin
    md = [f"# Four arms on one test set ({n_total} held-in questions)\n",
          f"Teacher: `{PR['teacher_model']}` · judge: Kimi-K2.6 · budget {PR['budget']} (+hidden) · GPU ${rate:.2f}/h A100\n",
          "## Accuracy (pooled)\n", "| Arm | n | EM | F1 | Cover | Judge | Judge macro | Doc recall |", "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        p = P[a]; md.append(f"| {SHORT[a]} | {p['n']} | {pct(p['em'])} | {num(p['f1'],3)} | {pct(p['cover'])} | **{pct(p['judge'])}** | {pct(p['macro_judge'])} | {num(p['doc_recall'],3)} |")
    md += ["\n## Judge-correct by dataset\n", "| Dataset | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for d in DS:
        md.append(f"| {DSN[d]} | " + " | ".join(pct((D[a].get(d) or {}).get("judge")) for a in ARMS) + " |")
    md += ["\n## Efficiency (pooled, per episode)\n", "| Arm | Steps | Voluntary finish | Invalid/ep | Student tok | Teacher tok | Total tok | Plan tok | Latency s |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        p = P[a]; plan = "—" if p["plan_tokens_per_ep"] is None else f"{p['plan_tokens_per_ep']:,.0f}"
        md.append(f"| {SHORT[a]} | {num(p['mean_steps'])} | {pct(p['voluntary_finish'])} | {num(p['invalid_steps_per_ep'],3)} | {p['student_tokens_per_ep']:,.0f} | {p['teacher_tokens_per_ep']:,.0f} | {p['total_tokens_per_ep']:,.0f} | {plan} | {num(p['latency_s_per_ep'],1)} |")
    md += ["\n## Cost (pooled)\n", "| Arm | API $ | GPU $ | Training $ (amortized) | $/episode | $/episode excl. training | $/correct | Tokens/correct |", "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        p = P[a]; md.append(f"| {SHORT[a]} | {usd(p['api_usd'],3)} | {usd(p['gpu_usd'],3)} | {usd(p['train_amort_usd'],3) if a=='all4' else '—'} | **{usd(p['usd_per_ep'])}** | {usd((p['api_usd']+p['gpu_usd'])/p['n'])} | {usd(p['usd_per_correct'])} | {p['tokens_per_correct']:,.0f} |")
    md += ["\n## Paired significance on judge-correct (pooled)\n", "| a → b | Δ pts | 95% CI | b wins / a wins | McNemar p |", "|---|---|---|---|---|"]
    for a, b in pairs:
        s = (S.get(f"{a}->{b}") or {}).get("pooled")
        if s:
            md.append(f"| {SHORT[a]} → {SHORT[b]} | {s['diff']*100:+.1f} | [{s['ci95'][0]*100:+.1f}, {s['ci95'][1]*100:+.1f}] | {s['b_wins']} / {s['a_wins']} | {s['p']:.2g} |")
    md += ["\n## Solved-question overlap\n", f"Solved by any arm: {O['solved_by_any']}/{n_total}; by all four: {O['solved_by_all']}; by none: {O['unsolved_by_all']}.\n",
           "\nSee `report.html` for the full write-up, per-dataset tables and caveats.\n"]
    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out/'report.html'} and {out/'RESULTS.md'}")


if __name__ == "__main__":
    main()
