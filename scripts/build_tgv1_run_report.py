"""Post-flight report for a tg_v1 collection run.

Answers, per dataset and overall: did every question produce an episode, what did the
student achieve, how did episodes terminate, did any hidden gold reach the student, and
what did it cost. Written to be run unattended right after collection.

The leakage section deliberately separates two things the run validator conflates: a
``leakage_check`` flag means the guard *fired and sanitized*, whereas an exposure means the
gold answer is still readable in ``student_visible_guidance``. Only the second is a defect.
A gold string that already appears in the question (comparison questions contain their own
answer) is not hidden information and is counted separately again.

Usage::

    python scripts/build_tgv1_run_report.py --run-root data/simulation_output/tgv1 \
        --out reports/tgv1/RUN_REPORT.md
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.episode_cost_report import summarize  # noqa: E402

EXPECTED = {"hotpotqa": 2000, "2wikimultihopqa": 2000, "musique": 2000, "strategyqa": 1999}


def visible_text(ep: Dict[str, Any]) -> List[str]:
    out = []
    pr = ep.get("plan_review") or {}
    for rnd in pr.get("rounds") or []:
        out.append(json.dumps(rnd.get("student_visible") or {}))
        out.append(json.dumps(rnd.get("revised_plan") or {}))
    for s in ep.get("steps") or []:
        out.append(json.dumps(s.get("student_visible_guidance") or {}))
    return out


def load(run_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_ds: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for p in run_root.rglob("teacher_guidance_episodes.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ep = json.loads(line)
                by_ds[ep.get("dataset") or "unknown"].append(ep)
    return by_ds


def analyse(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [summarize(e) for e in eps]
    flags = collections.Counter()
    exposures, echo_safe = [], 0
    for e in eps:
        gold = (e.get("gold_answer") or "").strip()
        for s in e.get("steps") or []:
            for f, v in ((s.get("leakage_check") or {}) or {}).items():
                if v is True:
                    flags[f] += 1
        if not gold:
            continue
        pat = re.compile(r"\b" + re.escape(gold) + r"\b", re.I)
        if any(pat.search(v) for v in visible_text(e)):
            if pat.search(e.get("query") or ""):
                echo_safe += 1
            else:
                exposures.append(e.get("qid"))
    qids = [e.get("qid") for e in eps]
    return {
        "n": len(eps),
        "unique_qids": len(set(qids)),
        "duplicates": len(qids) - len(set(qids)),
        "answer_correct": sum(1 for r in rows if r["answer_correct"]),
        "exact_match": sum(1 for r in rows if r["exact_match"]),
        "grounded": sum(1 for r in rows if r["answer_grounded"]),
        "teacher_correct": sum(1 for r in rows if r["teacher_answer_correct"]),
        "teacher_scored": sum(1 for r in rows if r["teacher_answer_score"] is not None),
        "mean_f1": st.mean(r["f1"] or 0.0 for r in rows),
        "mean_doc_recall": st.mean(r["doc_recall"] or 0.0 for r in rows),
        "mean_steps": st.mean(r["steps"] or 0 for r in rows),
        "mean_teacher_calls": st.mean(r["roles"]["teacher"]["calls"] for r in rows),
        "cost": sum(r["total_cost"] for r in rows),
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rows).most_common()),
        "flags": dict(flags),
        "exposures": exposures,
        "echo_safe": echo_safe,
        "provenance": sorted({(e.get("schema_version"), (e.get("framework_commit") or "")[:12],
                              e.get("config_hash")) for e in eps}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="data/simulation_output/tgv1")
    ap.add_argument("--out", default="reports/tgv1/RUN_REPORT.md")
    args = ap.parse_args()

    by_ds = load(Path(args.run_root))
    if not by_ds:
        raise SystemExit(f"no episodes under {args.run_root}")

    L = ["# tg_v1 Collection — Run Report", ""]
    L.append("| Dataset | Episodes | Expected | Correct | EM | Grounded | Mean F1 | Doc recall | Steps | Teacher calls | Cost |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = collections.Counter()
    all_exposures, all_flags = [], collections.Counter()
    for ds in sorted(by_ds):
        a = analyse(by_ds[ds])
        exp = EXPECTED.get(ds, a["n"])
        L.append(f"| {ds} | {a['n']} | {exp} | {a['answer_correct']} ({a['answer_correct']/a['n']:.0%}) | "
                 f"{a['exact_match']} ({a['exact_match']/a['n']:.0%}) | {a['grounded']} ({a['grounded']/a['n']:.0%}) | "
                 f"{a['mean_f1']:.3f} | {a['mean_doc_recall']:.3f} | {a['mean_steps']:.2f} | "
                 f"{a['mean_teacher_calls']:.1f} | ${a['cost']:.4f} |")
        for k in ("n", "answer_correct", "exact_match", "grounded", "duplicates"):
            tot[k] += a[k]
        tot["cost"] += a["cost"]
        all_exposures += a["exposures"]
        all_flags.update(a["flags"])
        L += ["", f"**{ds}** — stop reasons: `{a['stop_reasons']}` · provenance: `{a['provenance']}` · "
                  f"unique qids {a['unique_qids']}/{a['n']}", ""]

    L += ["## Totals", "",
          f"- Episodes: **{tot['n']}**",
          f"- Answer-correct: **{tot['answer_correct']} ({tot['answer_correct']/tot['n']:.1%})**",
          f"- Exact match: {tot['exact_match']} ({tot['exact_match']/tot['n']:.1%})",
          f"- Grounded: {tot['grounded']} ({tot['grounded']/tot['n']:.1%})",
          f"- Duplicate qids: {tot['duplicates']}",
          f"- **Cost: ${tot['cost']:.4f}**", "",
          "## Leakage", "",
          f"- Guard flags fired (detected **and sanitized**): `{dict(all_flags) or 'none'}`",
          f"- **Hidden gold visible to the student: {len(all_exposures)}**"
          + (f" — {all_exposures[:10]}" if all_exposures else " ✅"), ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
