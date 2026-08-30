"""Per-episode cost, latency and accuracy from teacher-guidance episode files.

Answers the question a collection budget actually turns on: *what does one episode cost,
and where does the money go?* Walks every logged LLM call in an episode -- per-step
student and teacher calls plus every plan-review round -- and attributes tokens, wall
time and USD to the side that spent them. Providers that bill (EdenAI) return a real
per-call ``cost``; free gateways (FAU) return ``None``, which is reported as 0 rather
than silently dropped, so a mixed run's total is still the true spend.

Usage::

    python scripts/episode_cost_report.py --runs data/simulation_output/costprobe/* \
        --scale 10000 120000
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"


def _calls(episode: Dict[str, Any]) -> Iterable[tuple]:
    """Yield ``(role, call)`` for every logged LLM call in one episode.

    Plan-review rounds are split by who spoke: ``initial_plan_calls``/``revision_calls``
    are the student drafting and revising its own plan, ``review_calls`` are the teacher
    critiquing it. Collapsing them would misattribute the planning phase's spend.
    """
    pr = episode.get("plan_review") or {}
    for rnd in pr.get("rounds") or []:
        for key, role in (("initial_plan_calls", "student"),
                          ("revision_calls", "student"),
                          ("review_calls", "teacher")):
            for c in rnd.get(key) or []:
                yield role, c
    for key, role in (("initial_plan_calls", "student"), ("revision_calls", "student"),
                      ("review_calls", "teacher")):
        for c in pr.get(key) or []:
            yield role, c
    for s in episode.get("steps") or []:
        for c in s.get("student_calls") or []:
            yield "student", c
        for c in s.get("teacher_calls") or []:
            yield "teacher", c


def summarize(episode: Dict[str, Any]) -> Dict[str, Any]:
    agg = {r: {"calls": 0, "in": 0, "out": 0, "reasoning": 0, "cost": 0.0, "ms": 0.0,
               "cost_missing": 0, "models": set()} for r in ("student", "teacher")}
    for role, c in _calls(episode):
        a = agg[role]
        u = c.get("usage") or {}
        a["calls"] += 1
        a["in"] += u.get("prompt_tokens") or 0
        a["out"] += u.get("completion_tokens") or 0
        a["reasoning"] += u.get("reasoning_tokens") or 0
        cost = u.get("cost")
        if cost is None:
            a["cost_missing"] += 1
        else:
            a["cost"] += float(cost)
        a["ms"] += c.get("elapsed_ms") or 0.0
        if c.get("model"):
            a["models"].add(c["model"])

    fm = episode.get("final_metrics") or {}
    return {
        "qid": episode.get("qid") or episode.get("question_id"),
        "dataset": episode.get("dataset"),
        "query": episode.get("query"),
        "student_model": episode.get("student_model"),
        "teacher_models": sorted(agg["teacher"]["models"]),
        "steps": episode.get("used_steps", len(episode.get("steps") or [])),
        "budget": episode.get("budget"),
        "stop_reason": episode.get("stop_reason"),
        "exact_match": fm.get("exact_match"),
        "f1": fm.get("f1", fm.get("f1_score")),
        "answer_correct": fm.get("answer_correct"),
        "doc_recall": fm.get("supporting_doc_recall"),
        "answer_grounded": fm.get("answer_grounded"),
        "teacher_answer_score": fm.get("teacher_answer_score"),
        "teacher_answer_correct": fm.get("teacher_answer_correct"),
        "plan_rounds": len(((episode.get("plan_review") or {}).get("rounds")) or []),
        "final_answer": (episode.get("final_answer") or "")[:120],
        "gold_answer": episode.get("gold_answer"),
        "roles": {r: {k: (sorted(v) if isinstance(v, set) else v) for k, v in a.items()}
                  for r, a in agg.items()},
        "total_cost": agg["student"]["cost"] + agg["teacher"]["cost"],
        "wall_s": (agg["student"]["ms"] + agg["teacher"]["ms"]) / 1000.0,
    }


def load_run(run_dir: Path) -> List[Dict[str, Any]]:
    eps = []
    for p in sorted(run_dir.rglob(EPISODE_FILENAME)):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                eps.append(json.loads(line))
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scale", nargs="*", type=int, default=[10000, 120000])
    ap.add_argument("--json-out")
    args = ap.parse_args()

    report = []
    for run in args.runs:
        run_dir = Path(run)
        eps = load_run(run_dir)
        if not eps:
            print(f"!! no episodes under {run_dir}")
            continue
        rows = [summarize(e) for e in eps]
        costs = [r["total_cost"] for r in rows]
        mean_cost = statistics.mean(costs)
        # With a handful of episodes the mean is the projection, but the spread is what
        # decides whether to trust it -- teacher spend per episode varies with how many
        # repair retries the teacher's own output triggers.
        sd = statistics.stdev(costs) if len(costs) > 1 else 0.0
        entry = {
            "run": str(run_dir),
            "n_episodes": len(rows),
            "mean_cost": mean_cost,
            "sd_cost": sd,
            "min_cost": min(costs),
            "max_cost": max(costs),
            "mean_teacher_calls": statistics.mean(r["roles"]["teacher"]["calls"] for r in rows),
            "mean_teacher_in": statistics.mean(r["roles"]["teacher"]["in"] for r in rows),
            "mean_teacher_out": statistics.mean(r["roles"]["teacher"]["out"] for r in rows),
            "mean_api_s": statistics.mean(r["wall_s"] for r in rows),
            "answer_correct": sum(1 for r in rows if r["answer_correct"]),
            "exact_match": sum(1 for r in rows if r["exact_match"]),
            "teacher_correct": sum(1 for r in rows if r["teacher_answer_correct"]),
            "teacher_scored": sum(1 for r in rows if r["teacher_answer_score"] is not None),
            "grounded": sum(1 for r in rows if r["answer_grounded"]),
            "mean_f1": statistics.mean(r["f1"] or 0.0 for r in rows),
            "mean_doc_recall": statistics.mean(r["doc_recall"] or 0.0 for r in rows),
            "episodes": rows,
            "projections": {str(n): mean_cost * n for n in args.scale},
            "projections_hi": {str(n): (mean_cost + sd) * n for n in args.scale},
        }
        report.append(entry)

        r0 = rows[0]
        if len(rows) > 1:
            print(f"\n=== {run_dir.name} · {len(rows)} episodes ===")
            print(f"  teacher        : {', '.join(r0['teacher_models']) or '-'}")
            print(f"  student        : {r0['student_model']}")
            print(f"  cost/episode   : ${mean_cost:.6f}  (sd ${sd:.6f}, "
                  f"min ${min(costs):.6f}, max ${max(costs):.6f})")
            print(f"  teacher/episode: {entry['mean_teacher_calls']:.1f} calls · "
                  f"{entry['mean_teacher_in']:.0f} in / {entry['mean_teacher_out']:.0f} out · "
                  f"{entry['mean_api_s']:.0f}s API time")
            print(f"  accuracy       : answer_correct {entry['answer_correct']}/{len(rows)} · "
                  f"EM {entry['exact_match']}/{len(rows)} · "
                  f"teacher-verdict {entry['teacher_correct']}/{entry['teacher_scored']} scored · "
                  f"grounded {entry['grounded']}/{len(rows)}")
            print(f"  mean F1 {entry['mean_f1']:.3f} · mean doc-recall {entry['mean_doc_recall']:.3f}")
            for n in args.scale:
                print(f"     x{n:,} -> ${entry['projections'][str(n)]:,.2f} "
                      f"(+1sd: ${entry['projections_hi'][str(n)]:,.2f})")
            continue
        print(f"\n=== {run_dir.name} · {len(rows)} episode(s) ===")
        print(f"  teacher        : {', '.join(r0['teacher_models']) or '-'}")
        print(f"  student        : {r0['student_model']}")
        print(f"  steps / stop   : {r0['steps']}/{r0['budget']} · plan rounds {r0['plan_rounds']} · {r0['stop_reason']}")
        print(f"  gold           : {r0['gold_answer']!r}")
        print(f"  answer         : {r0['final_answer']!r}")
        print(f"  EM / F1        : {r0['exact_match']} / {r0['f1']}   answer_correct={r0['answer_correct']}"
              f"  grounded={r0['answer_grounded']}")
        print(f"  doc_recall     : {r0['doc_recall']}   teacher_verdict={r0['teacher_answer_score']} "
              f"(correct={r0['teacher_answer_correct']})")
        for role in ("student", "teacher"):
            a = r0["roles"][role]
            print(f"  {role:<8}: {a['calls']:>3} calls · {a['in']:>7} in / {a['out']:>6} out "
                  f"(reasoning {a['reasoning']}) · {a['ms']/1000:7.1f}s · ${a['cost']:.6f}"
                  + (f"  [{a['cost_missing']} calls unbilled]" if a["cost_missing"] else ""))
        print(f"  TOTAL          : ${entry['mean_cost']:.6f} / episode · {r0['wall_s']:.1f}s of API time")
        for n in args.scale:
            print(f"     x{n:,} episodes -> ${entry['projections'][str(n)]:,.2f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
