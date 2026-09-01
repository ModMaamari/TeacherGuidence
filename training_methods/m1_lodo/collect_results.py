"""Collect every m1-LODO measurement into one JSON the report renders from.

Merges four sources: eval metrics.json (sharded), the student judge verdicts, the
teacher-arm episodes, and the teacher judge verdicts. Emits per-configuration rows plus
the paired subsets needed for an honest student-vs-teacher comparison (the teacher ran a
~250-question sample of each unseen set, so the students must be filtered to the same qids
before the two are placed side by side).
"""
from __future__ import annotations
import argparse, collections, glob, json, statistics as st
from pathlib import Path

DS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
ROOT = Path("training_methods/m1_lodo/runs/lodo")


def load_eval():
    parts = collections.defaultdict(list)
    for m in (ROOT / "eval").rglob("metrics.json"):
        t = m.parent.name
        t = t.split("_", 1)[1] if "_" in t else t
        parts[t.split("__s")[0]].append(json.loads(m.read_text()))
    out = {}
    for tag, ps in parts.items():
        n = sum(p.get("n", 0) for p in ps)
        if not n:
            continue
        r = {"n": n}
        for k in ("em", "f1", "cover_match", "doc_recall", "mean_steps", "tokens_per_episode"):
            v = [(p.get(k), p.get("n", 0)) for p in ps if p.get(k) is not None]
            r[k] = round(sum(a * b for a, b in v) / n, 4) if v else None
        stc = collections.Counter()
        for p in ps:
            stc.update(p.get("stop_reasons") or {})
        r["stop_reasons"] = dict(stc)
        r["voluntary_finish"] = round(stc.get("finish", 0) / n, 4)
        r["invalid_action_steps"] = sum(p.get("invalid_action_steps") or 0 for p in ps)
        r["total_steps"] = sum(p.get("total_steps") or 0 for p in ps)
        r["invalid_rate"] = round(r["invalid_action_steps"] / max(r["total_steps"], 1), 4)
        out[tag] = r
    return out


def load_judge(path, key_fn):
    """tag -> {qid: correct}"""
    by = collections.defaultdict(dict)
    p = Path(path)
    if not p.exists():
        return by
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by[key_fn(r["source"])][r["qid"]] = r["verdict"]["correct"]
    return by


def student_tag(src: str) -> str:
    t = Path(src).parent.name
    t = t.split("_", 1)[1] if "_" in t else t
    return t.split("__s")[0]


def teacher_tag(src: str) -> str:
    tag = src.split("teacher_arm/")[1].split("/")[0]
    return tag[len("tarm_"):].rsplit("_s", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_methods/m1_lodo/runs/lodo/results.json")
    args = ap.parse_args()

    ev = load_eval()
    sj = load_judge(ROOT / "judge/verdicts.jsonl", student_tag)
    tj = load_judge(ROOT / "judge_teacher/verdicts.jsonl", teacher_tag)

    for tag, r in ev.items():
        v = sj.get(tag, {})
        r["judge_n"] = len(v)
        r["judge_correct"] = round(sum(v.values()) / len(v), 4) if v else None

    # teacher arm, from raw episodes
    teacher = {}
    tep = collections.defaultdict(list)
    for p in glob.glob("data/simulation_output/teacher_arm/**/teacher_guidance_episodes.jsonl",
                       recursive=True):
        name = teacher_tag(p)
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                tep[name].append(json.loads(line))
    for name, eps in tep.items():
        n = len(eps)
        m = [e.get("final_metrics") or {} for e in eps]
        v = tj.get(name, {})
        teacher[name] = {
            "n": n,
            "cover_match": round(sum(bool(x.get("answer_correct")) for x in m) / n, 4),
            "em": round(sum(bool(x.get("exact_match")) for x in m) / n, 4),
            "f1": round(st.mean(x.get("f1") or 0 for x in m), 4),
            "doc_recall": round(st.mean(x.get("supporting_doc_recall") or 0 for x in m), 4),
            "mean_steps": round(st.mean(e.get("used_steps") or 0 for e in eps), 3),
            "judge_n": len(v),
            "judge_correct": round(sum(v.values()) / len(v), 4) if v else None,
            "qids": sorted({e["qid"] for e in eps}),
        }

    # paired subsets: students restricted to the teacher's sampled qids
    paired = {}
    for d in DS:
        for kind in ("heldin", "unseen"):
            name = f"{kind}_{d}"
            t = teacher.get(name)
            if not t:
                continue
            qids = set(t["qids"])
            row = {"teacher_judge": t["judge_correct"], "teacher_cover": t["cover_match"],
                   "n_paired": len(qids)}
            for model in (["base"] + [f"fold_{f}" for f in DS]):
                tag = f"{model}__{name}"
                v = sj.get(tag, {})
                sub = [c for q, c in v.items() if q in qids]
                if sub:
                    row[model] = round(sum(sub) / len(sub), 4)
            paired[name] = row

    out = {"eval": ev, "teacher": {k: {kk: vv for kk, vv in v.items() if kk != "qids"}
                                   for k, v in teacher.items()},
           "paired": paired}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(ev)} student configs, {len(teacher)} teacher sets, "
          f"{len(paired)} paired sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
