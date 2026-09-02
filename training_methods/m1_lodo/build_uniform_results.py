"""Four-arm comparison on the uniform test set (747 held-in questions, 10% of each
dataset, never trained on by any model).

    arm      | agent                                    | source of episodes
    ---------|------------------------------------------|-----------------------------------------
    base     | granite-4.1-3b, no teacher               | runs/lodo/eval/*base__heldin_*   (vLLM)
    guided   | granite-4.1-3b + live DeepSeek teacher   | runs/uniform/eval/*guided__heldin_*
    teacher  | DeepSeek-V4-Flash-0731 as the agent      | data/simulation_output/teacher_arm/tarm_heldin_*
    all4     | granite-4.1-3b SFT on all four datasets  | runs/uniform/eval/*all4__heldin_*

Every arm ran the same questions, corpus, tool set, budget 3 (+hidden budget) and greedy
student decoding.  Accuracy = EM / F1 / cover-match / LLM-judge (Kimi-K2.6 router);
efficiency = steps, tokens, latency; cost = API dollars reported per call by EdenAI plus
an explicit, configurable GPU-hour rate for the local student.  Paired bootstrap CIs and
exact McNemar tests are computed on judge-correct across arms (same qids).

    python training_methods/m1_lodo/build_uniform_results.py \
        --out training_methods/m1_lodo/runs/uniform/results.json
"""
from __future__ import annotations
import argparse, collections, glob, json, math, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.episode_cost_report import _calls  # noqa: E402  (per-call usage walker)

DS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
LODO = Path("training_methods/m1_lodo/runs/lodo")
UNI = Path("training_methods/m1_lodo/runs/uniform")
TARM = Path("data/simulation_output/teacher_arm")
ARMS = ["base", "guided", "teacher", "all4"]
ARM_LABEL = {"base": "Base student (no teacher)",
             "guided": "Teacher-guided student (live teacher)",
             "teacher": "Teacher alone (DeepSeek-V4-Flash-0731)",
             "all4": "Trained student (SFT on all 4 datasets)"}


# ----------------------------------------------------------------------------- loaders
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_judge(path):
    """(source-file, qid) -> correct (0/1), plus judge model counts."""
    v, models = {}, collections.Counter()
    if Path(path).exists():
        for r in read_jsonl(path):
            v[(r["source"], r["qid"])] = int(bool(r["verdict"]["correct"]))
            models[r.get("judge_model")] += 1
    return v, models


def eval_dirs(root, tag):
    return sorted(d for d in glob.glob(f"{root}/eval/*{tag}")
                  if Path(d, "episodes.jsonl").exists() and Path(d, "metrics.json").exists())


def student_episodes(root, tag):
    """Student-arm episodes (eval_agent / teacher_eval_agent), merged over shards."""
    eps, mets = [], []
    for d in eval_dirs(root, tag):
        for e in read_jsonl(Path(d, "episodes.jsonl")):
            e["_src"] = str(Path(d, "episodes.jsonl"))
            eps.append(e)
        mets.append(json.loads(Path(d, "metrics.json").read_text()))
    return eps, mets


def teacher_episodes(ds):
    eps = []
    for p in sorted(glob.glob(f"{TARM}/tarm_heldin_{ds}_s*/**/teacher_guidance_episodes.jsonl",
                              recursive=True)):
        for e in read_jsonl(p):
            e["_src"] = p
            eps.append(e)
    return eps


# ----------------------------------------------------------------------------- per-episode rows
def _usage(c):
    u = c.get("usage") or c.get("gen_stats") or {}
    return (u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0,
            float(u.get("cost") or 0.0), float(c.get("elapsed_ms") or 0.0))


def student_row(e, arm, judge):
    """eval_agent (base/all4) and teacher_eval_agent (guided) episodes.

    Token accounting is split into the STEP phase (logged by every arm) and the PLAN
    phase (logged by eval_agent as plan.gen_stats and by the teacher arm; NOT logged by
    teacher_eval_agent, which trims plan-review calls -> None for the guided arm)."""
    fm = e.get("final_metrics") or {}
    steps = e.get("steps") or []
    s_in = s_out = t_in = t_out = 0
    api_cost = t_ms = 0.0
    invalid = 0
    for s in steps:
        if s.get("gen_stats"):                       # eval_agent
            pi, co, _, _ = _usage(s); s_in += pi; s_out += co
        for c in s.get("student_calls") or []:       # teacher_eval_agent
            pi, co, _, _ = _usage(c); s_in += pi; s_out += co
        for c in s.get("teacher_calls") or []:
            pi, co, cost, ms = _usage(c); t_in += pi; t_out += co; api_cost += cost; t_ms += ms
        m = s.get("metrics") or {}
        if s.get("action_valid") is False or m.get("invalid_action"):
            invalid += 1
    plan_in = plan_out = None
    plan = e.get("plan") or {}
    if isinstance(plan, dict) and plan.get("gen_stats"):
        pi, co, _, _ = _usage(plan); plan_in, plan_out = pi, co
    stop = e.get("stop_reason") or ""
    return {
        "qid": e["qid"], "arm": arm,
        "em": int(bool(fm.get("exact_match"))), "f1": float(fm.get("f1") or 0.0),
        "cover": int(bool(fm.get("cover_match", fm.get("answer_correct")))),
        "doc_recall": float(fm.get("doc_recall", fm.get("supporting_doc_recall")) or 0.0),
        "judge": judge.get((e["_src"], e["qid"])),
        "steps": int(e.get("used_steps") or len(steps)),
        "voluntary_finish": int(stop in ("finish", "teacher_accept")),
        "invalid": invalid,
        "student_in": s_in, "student_out": s_out, "teacher_in": t_in, "teacher_out": t_out,
        "plan_in": plan_in, "plan_out": plan_out,
        "api_cost_usd": api_cost, "latency_s": float(e.get("elapsed_s") or 0.0),
        "teacher_api_s": t_ms / 1000.0,
    }


def teacher_row(e, judge):
    """Teacher-alone episodes (agentsim harness): the teacher IS the agent, so its
    calls are the 'student' tokens; all of it goes through the API."""
    fm = e.get("final_metrics") or {}
    s_in = s_out = 0; cost = ms = 0.0
    plan_in = plan_out = 0; plan_cost = plan_ms = 0.0
    step_ids = {id(c) for s in (e.get("steps") or []) for c in (s.get("student_calls") or [])}
    for role, c in _calls(e):
        pi, co, cc, el = _usage(c)
        if id(c) in step_ids:
            s_in += pi; s_out += co; cost += cc; ms += el
        else:                                        # plan-review rounds (student + teacher)
            plan_in += pi; plan_out += co; plan_cost += cc; plan_ms += el
    steps = e.get("steps") or []
    stop = e.get("stop_reason") or ""
    return {
        "qid": e["qid"], "arm": "teacher",
        "em": int(bool(fm.get("exact_match"))), "f1": float(fm.get("f1") or 0.0),
        "cover": int(bool(fm.get("answer_correct"))),
        "doc_recall": float(fm.get("supporting_doc_recall") or 0.0),
        "judge": judge.get((e["_src"], e["qid"])),
        "steps": int(e.get("used_steps") or len(steps)),
        "voluntary_finish": int("finish" in stop or "accept" in stop),
        "invalid": sum(1 for s in steps if (s.get("metrics") or {}).get("invalid_action")),
        "student_in": s_in, "student_out": s_out, "teacher_in": 0, "teacher_out": 0,
        "plan_in": plan_in, "plan_out": plan_out, "plan_api_cost_usd": plan_cost,
        "api_cost_usd": cost, "latency_s": (ms + plan_ms) / 1000.0, "teacher_api_s": ms / 1000.0,
    }


# ----------------------------------------------------------------------------- stats
def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def paired_bootstrap(a, b, iters=10000, seed=13):
    """CI for mean(b) - mean(a) over the same qids (a, b aligned lists of 0/1)."""
    rnd = random.Random(seed); n = len(a)
    if n == 0:
        return None
    diffs = [b[i] - a[i] for i in range(n)]
    obs = sum(diffs) / n
    boots = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += diffs[rnd.randrange(n)]
        boots.append(s / n)
    boots.sort()
    return {"diff": round(obs, 4), "ci95": [round(boots[int(0.025 * iters)], 4),
                                            round(boots[int(0.975 * iters) - 1], 4)], "n": n}


def mcnemar_exact(a, b):
    """Exact two-sided McNemar on discordant pairs."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    m = b01 + b10
    if m == 0:
        return {"b_wins": b01, "a_wins": b10, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {"b_wins": b01, "a_wins": b10, "p": round(p, 5)}


# ----------------------------------------------------------------------------- aggregate
def aggregate(rows, gpu_wall_s, gpu_rate, train_amort_usd):
    n = len(rows)
    if not n:
        return None
    j = [r["judge"] for r in rows if r["judge"] is not None]
    tok_s = sum(r["student_in"] + r["student_out"] for r in rows)
    tok_t = sum(r["teacher_in"] + r["teacher_out"] for r in rows)
    api = sum(r["api_cost_usd"] for r in rows)
    gpu_usd = gpu_wall_s / 3600.0 * gpu_rate
    total_usd = api + gpu_usd + train_amort_usd
    correct = sum(j)
    out = {
        "n": n, "judge_n": len(j),
        "em": round(mean(r["em"] for r in rows), 4), "f1": round(mean(r["f1"] for r in rows), 4),
        "cover": round(mean(r["cover"] for r in rows), 4),
        "judge": round(correct / len(j), 4) if j else None,
        "doc_recall": round(mean(r["doc_recall"] for r in rows), 4),
        "mean_steps": round(mean(r["steps"] for r in rows), 3),
        "voluntary_finish": round(mean(r["voluntary_finish"] for r in rows), 4),
        "invalid_steps_per_ep": round(mean(r["invalid"] for r in rows), 3),
        "student_tokens_per_ep": round(tok_s / n, 1), "teacher_tokens_per_ep": round(tok_t / n, 1),
        "total_tokens_per_ep": round((tok_s + tok_t) / n, 1),
        "student_out_tokens_per_ep": round(sum(r["student_out"] for r in rows) / n, 1),
        "plan_tokens_per_ep": (round(sum((r["plan_in"] or 0) + (r["plan_out"] or 0) for r in rows) / n, 1)
                               if all(r["plan_in"] is not None for r in rows) else None),
        "latency_s_per_ep": round(mean(r["latency_s"] for r in rows), 2),
        "teacher_api_s_per_ep": round(mean(r["teacher_api_s"] for r in rows), 2),
        "gpu_wall_s": round(gpu_wall_s, 1), "gpu_s_per_ep": round(gpu_wall_s / n, 2),
        "api_usd": round(api, 4), "api_usd_per_ep": round(api / n, 6),
        "gpu_usd": round(gpu_usd, 4), "train_amort_usd": round(train_amort_usd, 4),
        "total_usd": round(total_usd, 4), "usd_per_ep": round(total_usd / n, 6),
        "usd_per_correct": round(total_usd / correct, 5) if correct else None,
        "correct_per_1k_tokens": round(1000.0 * correct / (tok_s + tok_t), 4) if tok_s + tok_t else None,
        "tokens_per_correct": round((tok_s + tok_t) / correct, 1) if correct else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(UNI / "results.json"))
    ap.add_argument("--gpu-usd-per-hour", type=float, default=2.0,
                    help="assumed price of one A100-80GB hour for the local student")
    ap.add_argument("--train-gpu-hours", type=float, default=None,
                    help="all4 training GPU hours (default: read train/final_metrics.json)")
    ap.add_argument("--amortize-over", type=int, default=None,
                    help="episodes the training cost is spread over (default: the 747 test episodes)")
    args = ap.parse_args()

    j_lodo, jm1 = load_judge(LODO / "judge/verdicts.jsonl")
    j_uni, jm2 = load_judge(UNI / "judge/verdicts.jsonl")
    j_teach, jm3 = load_judge(LODO / "judge_teacher/verdicts.jsonl")

    train_h = args.train_gpu_hours
    if train_h is None:
        fm = json.loads((UNI / "train/final_metrics.json").read_text())
        train_h = fm["train_runtime"] / 3600.0
    train_meta = json.loads((UNI / "train/final_metrics.json").read_text())

    rows = {a: {d: [] for d in DS} for a in ARMS}
    gpu_wall = {a: {d: 0.0 for d in DS} for a in ARMS}
    server_meta = {}
    for d in DS:
        eps, mets = student_episodes(LODO, f"base__heldin_{d}")
        rows["base"][d] = [student_row(e, "base", j_lodo) for e in eps]
        gpu_wall["base"][d] = sum(m.get("wall_time_s") or 0 for m in mets)
        eps, mets = student_episodes(UNI, f"guided__heldin_{d}")
        rows["guided"][d] = [student_row(e, "guided", j_uni) for e in eps]
        gpu_wall["guided"][d] = sum(m.get("wall_time_s") or 0 for m in mets)
        if mets:
            server_meta.setdefault("guided", mets[0])
        eps, mets = student_episodes(UNI, f"all4__heldin_{d}")
        rows["all4"][d] = [student_row(e, "all4", j_uni) for e in eps]
        gpu_wall["all4"][d] = sum(m.get("wall_time_s") or 0 for m in mets)
        rows["teacher"][d] = [teacher_row(e, j_teach) for e in teacher_episodes(d)]

    # one row per qid (defensive: keep the first if a shard duplicated a question)
    for a in ARMS:
        for d in DS:
            seen, uniq = set(), []
            for r in rows[a][d]:
                if r["qid"] not in seen:
                    seen.add(r["qid"]); uniq.append(r)
            rows[a][d] = uniq

    n_total = sum(len(rows["all4"][d]) for d in DS) or 747
    amort_n = args.amortize_over or n_total
    train_usd_total = train_h * args.gpu_usd_per_hour

    per_ds, pooled = {}, {}
    for a in ARMS:
        per_ds[a] = {}
        for d in DS:
            amort = (train_usd_total / amort_n * len(rows[a][d])) if a == "all4" else 0.0
            per_ds[a][d] = aggregate(rows[a][d], gpu_wall[a][d], args.gpu_usd_per_hour, amort)
        allrows = [r for d in DS for r in rows[a][d]]
        amort = (train_usd_total / amort_n * len(allrows)) if a == "all4" else 0.0
        pooled[a] = aggregate(allrows, sum(gpu_wall[a].values()), args.gpu_usd_per_hour, amort)
        if pooled[a]:
            pooled[a]["macro_judge"] = round(mean(per_ds[a][d]["judge"] for d in DS
                                                  if per_ds[a][d]), 4)

    # paired significance on judge-correct, all arm pairs, per dataset and pooled
    pairs = [("base", "all4"), ("base", "guided"), ("all4", "guided"), ("all4", "teacher"),
             ("guided", "teacher"), ("base", "teacher")]
    sig = {}
    for a, b in pairs:
        sig[f"{a}->{b}"] = {}
        for scope in DS + ["pooled"]:
            dss = DS if scope == "pooled" else [scope]
            ja = {r["qid"]: r["judge"] for d in dss for r in rows[a][d] if r["judge"] is not None}
            jb = {r["qid"]: r["judge"] for d in dss for r in rows[b][d] if r["judge"] is not None}
            common = sorted(set(ja) & set(jb))
            xa = [ja[q] for q in common]; xb = [jb[q] for q in common]
            if common:
                sig[f"{a}->{b}"][scope] = {**paired_bootstrap(xa, xb), **mcnemar_exact(xa, xb)}

    # question-level agreement: how many questions each arm solves that another does not
    qsets = {a: {r["qid"] for d in DS for r in rows[a][d] if r["judge"] == 1} for a in ARMS}
    overlap = {a: {b: len(qsets[a] & qsets[b]) for b in ARMS} for a in ARMS}
    any_arm = set().union(*qsets.values())
    all_arms = set.intersection(*qsets.values())

    coverage = {a: {d: {"n": len(rows[a][d]),
                        "judged": sum(1 for r in rows[a][d] if r["judge"] is not None)}
                    for d in DS} for a in ARMS}
    qid_check = {d: len(set.intersection(*[{r["qid"] for r in rows[a][d]} for a in ARMS]))
                 for d in DS}

    out = {
        "test_set": {"description": "10% held-in pools of each dataset; never used in any training fold",
                     "per_dataset": {d: len(rows["all4"][d]) for d in DS}, "total": n_total,
                     "qids_common_to_all_arms": qid_check},
        "arms": ARM_LABEL,
        "protocol": {"budget": 3, "hidden_budget": True, "student_decoding": "greedy (T=0)",
                     "teacher_model": "edenchat/flexai/DeepSeek-V4-Flash-0731",
                     "judge_models": {"base": dict(jm1), "uniform": dict(jm2), "teacher": dict(jm3)},
                     "gpu_usd_per_hour": args.gpu_usd_per_hour,
                     "train_gpu_hours": round(train_h, 3), "train_usd_total": round(train_usd_total, 3),
                     "train_amortized_over": amort_n,
                     "train_final": {k: train_meta.get(k) for k in ("train_loss", "eval_loss", "epoch")},
                     "guided_concurrency": (server_meta.get("guided") or {}).get("concurrency")},
        "coverage": coverage,
        "per_dataset": per_ds, "pooled": pooled, "significance": sig,
        "solved_overlap": {"per_arm": {a: len(qsets[a]) for a in ARMS}, "pairwise": overlap,
                           "solved_by_any": len(any_arm), "solved_by_all": len(all_arms),
                           "unsolved_by_all": n_total - len(any_arm)},
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    # also dump the per-episode rows for downstream analysis
    Path(args.out).with_name("episode_rows.jsonl").write_text(
        "\n".join(json.dumps({**r, "dataset": d}) for a in ARMS for d in DS for r in rows[a][d]) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"{'arm':8s} {'n':>4s} {'EM':>6s} {'F1':>6s} {'cover':>6s} {'judge':>6s} {'steps':>5s} "
          f"{'tok/ep':>7s} {'$/ep':>9s} {'$/correct':>9s} {'lat s':>6s}")
    for a in ARMS:
        p = pooled[a]
        if p:
            print(f"{a:8s} {p['n']:4d} {p['em']:6.3f} {p['f1']:6.3f} {p['cover']:6.3f} "
                  f"{(p['judge'] or 0):6.3f} {p['mean_steps']:5.2f} {p['total_tokens_per_ep']:7.0f} "
                  f"{p['usd_per_ep']:9.5f} {(p['usd_per_correct'] or 0):9.5f} {p['latency_s_per_ep']:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
