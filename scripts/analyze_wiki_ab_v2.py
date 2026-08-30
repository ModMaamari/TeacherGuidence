"""
Aggregate the wiki-v2 A/B suite results (3 students x 5 reps x baseline/wiki).

Reads data/simulation_output/wikiv2_<tag>_{nowiki,wiki}10_r{1..5}/run episode files,
prints a human-readable summary, and writes reports/wiki_ab_v2/stats.json for the
report-writing step.

Usage:
    python scripts/analyze_wiki_ab_v2.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "wiki_ab_v2"

STUDENTS = [
    ("q08b", "qwen3.5:0.8b"),
    ("q2b", "qwen3.5:2b"),
    ("g3b", "granite4.1:3b"),
]
REPS = [1, 2, 3, 4, 5]
TEACHER_CORRECT_THRESHOLD = 0.40


def load_eps(root: str) -> dict:
    eps = {}
    for f in glob.glob(
        str(REPO_ROOT / "data/simulation_output" / root / "run/**/teacher_guidance_episodes.jsonl"),
        recursive=True,
    ):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                ep = json.loads(line)
                eps[ep["qid"]] = ep
    return eps


def ep_stats(ep: dict) -> dict:
    fm = ep.get("final_metrics", {}) or {}
    ts = fm.get("teacher_answer_score")
    ops = {"applied": 0, "ignored": 0, "rewrites": 0, "keeps": 0}
    for s in ep.get("steps", []):
        eo = s.get("wiki_edit_ops") or {}
        applied = eo.get("applied") or []
        ops["applied"] += len(applied)
        ops["ignored"] += len(eo.get("ignored") or [])
        ops["rewrites"] += sum(1 for a in applied if a == "rewrite")
        ops["keeps"] += sum(1 for a in applied if a == "KEEP")
    gold = (ep.get("gold_answer") or "").strip().lower()
    wiki_final = (ep.get("wiki_final") or "").lower()
    return {
        "cover": int(bool(fm.get("answer_correct"))),
        "teacher": (None if ts is None else int(float(ts) >= TEACHER_CORRECT_THRESHOLD)),
        "steps": len(ep.get("steps", [])),
        "natural": int(ep.get("stop_reason") == "teacher_accept"),
        "unknown": int(str(ep.get("final_answer", "")).strip().lower() == "unknown"),
        "gold_in_wiki": int(bool(gold) and gold in wiki_final),
        "ops": ops,
    }


def rep_row(eps: dict) -> dict:
    ss = [ep_stats(e) for e in eps.values()]
    t = [s["teacher"] for s in ss if s["teacher"] is not None]
    return {
        "n": len(ss),
        "teacher_correct": sum(t),
        "teacher_scored": len(t),
        "teacher_rate": (sum(t) / len(t)) if t else None,
        "cover": sum(s["cover"] for s in ss),
        "mean_steps": round(st.mean([s["steps"] for s in ss]), 2) if ss else None,
        "natural": sum(s["natural"] for s in ss),
        "unknown": sum(s["unknown"] for s in ss),
        "gold_in_wiki_missed": sum(1 for s in ss if s["gold_in_wiki"] and not s["cover"]),
        "edit_ops": {
            k: sum(s["ops"][k] for s in ss) for k in ("applied", "ignored", "rewrites", "keeps")
        },
    }


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return round(st.mean(vals), 4), (round(st.stdev(vals), 4) if len(vals) > 1 else 0.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"threshold": TEACHER_CORRECT_THRESHOLD, "students": {}}

    for tag, model in STUDENTS:
        arms = {}
        for arm, pattern in [("nowiki", f"wikiv2_{tag}_nowiki10_r{{r}}"), ("wiki", f"wikiv2_{tag}_wiki10_r{{r}}")]:
            reps = []
            for r in REPS:
                eps = load_eps(pattern.format(r=r))
                if eps:
                    reps.append({"rep": r, **rep_row(eps)})
            tr_m, tr_s = mean_std([x["teacher_rate"] for x in reps])
            cv_m, cv_s = mean_std([x["cover"] for x in reps])
            arms[arm] = {
                "reps": reps,
                "teacher_rate_mean": tr_m, "teacher_rate_std": tr_s,
                "cover_mean": cv_m, "cover_std": cv_s,
                "steps_mean": mean_std([x["mean_steps"] for x in reps])[0],
                "natural_mean": mean_std([x["natural"] for x in reps])[0],
                "unknown_mean": mean_std([x["unknown"] for x in reps])[0],
            }
        deltas = []
        for i in range(min(len(arms["nowiki"]["reps"]), len(arms["wiki"]["reps"]))):
            a, b = arms["nowiki"]["reps"][i]["teacher_rate"], arms["wiki"]["reps"][i]["teacher_rate"]
            if a is not None and b is not None:
                deltas.append(round(b - a, 4))
        result["students"][tag] = {"model": model, **arms, "paired_teacher_rate_deltas": deltas}

        print(f"\n===== {model} =====")
        for arm in ("nowiki", "wiki"):
            a = arms[arm]
            print(f"  {arm:7}: teacher-rate {a['teacher_rate_mean']} ± {a['teacher_rate_std']} | "
                  f"cover {a['cover_mean']} ± {a['cover_std']} | steps {a['steps_mean']} | "
                  f"natural {a['natural_mean']} | unknown {a['unknown_mean']}")
            for x in a["reps"]:
                print(f"    r{x['rep']}: n={x['n']} teacher {x['teacher_correct']}/{x['teacher_scored']} "
                      f"cover {x['cover']} steps {x['mean_steps']} natural {x['natural']} "
                      f"unknown {x['unknown']} gold-in-wiki-missed {x['gold_in_wiki_missed']} ops {x['edit_ops']}")
        print(f"  paired teacher-rate deltas (wiki - nowiki): {deltas}")

    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
