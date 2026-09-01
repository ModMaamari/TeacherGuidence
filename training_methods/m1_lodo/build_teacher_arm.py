"""Build the teacher-as-agent evaluation arm.

The students are measured solving questions alone. This arm asks what the *teacher*
scores on the same task under the same protocol -- it is the ceiling the students were
distilled toward, and without it "the student reached 0.64" has no upper reference.

The teacher runs as the AGENT, not as a critic: `skip_teacher: true`, so nothing guides
it. Same budget, same hidden-budget setting, same tools, same per-question retrieval, same
metrics as every student run.

**Why a subset.** EdenAI throttles this model to ~2.6 episodes/min, so all 18,986 test
episodes would take ~122 hours. Instead: every held-in question (747) plus a seeded
250-question sample of each unseen set (1,000). The sample is drawn by a fixed hash so the
same questions can be pulled out of the students' completed episode files, making the
comparison paired on identical items rather than across different samples.

Usage::

    python training_methods/m1_lodo/build_teacher_arm.py --unseen-sample 250
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402
from scripts.gen_fau_smoke_template import build_fau_smoke_template  # noqa: E402

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
CORPUS = {d: f"./data/datasets/tg_v1/{d}/{d}_train_corpus.jsonl" for d in DATASETS}
CORPUS["hotpotqa"] = "./data/datasets/tg_v1/hotpotqa/hotpotqa_train_corpus.jsonl"
TEACHER = "edenchat/flexai/DeepSeek-V4-Flash-0731"
DATA = REPO / "training_methods/m1_lodo/data"
OUT_Q = DATA / "teacher_arm"
TPL = REPO / "templates/simulations"


def sampled(qid: str, k: int, total: int, salt: str = "m1lodo-teacher") -> bool:
    """Deterministic k-of-total sample: keeps the same questions on every rebuild, and
    lets the students' episodes be filtered to the identical set afterwards."""
    h = int(hashlib.sha256(f"{salt}:{qid}".encode()).hexdigest(), 16) % 10_000
    return h < (k / max(total, 1)) * 10_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseen-sample", type=int, default=250)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()
    OUT_Q.mkdir(parents=True, exist_ok=True)

    plan = []
    for ds in DATASETS:
        for kind in ("heldin", "unseen"):
            src = DATA / "tests" / f"{kind}_{ds}_questions.jsonl"
            rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
            if kind == "unseen":
                rows = [r for r in rows if sampled(str(r["id"]), args.unseen_sample, len(rows))]
            name = f"{kind}_{ds}"
            (OUT_Q / f"{name}_questions.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            plan.append((name, ds, len(rows)))

    total = 0
    for name, ds, n in plan:
        for s in range(args.shards):
            shard = [l for i, l in enumerate((OUT_Q / f"{name}_questions.jsonl")
                                             .read_text(encoding="utf-8").splitlines()) if i % args.shards == s]
            if not shard:
                continue
            qp = OUT_Q / f"{name}_s{s}.jsonl"
            qp.write_text("\n".join(shard) + "\n", encoding="utf-8")
            tid = f"tarm_{name}_s{s}"
            t = build_fau_smoke_template(
                template_id=tid, student_model=TEACHER, teacher_model=TEACHER,
                teacher_router=[TEACHER], num_samples=len(shard),
                questions_path="./" + str(qp.relative_to(REPO)),
                corpus_path=CORPUS[ds],
                output_dir=f"./data/simulation_output/teacher_arm/{tid}",
                budget=3, workflow="hotpot_teacher_guided_b3_plan_review",
                planning_steps=3, max_plan_steps=6, disclose_budget=False,
                student_use_response_schema=False)
            # The teacher IS the agent here: nothing critiques it.
            t["mode_config"]["skip_teacher"] = True
            (TPL / f"{tid}.yaml").write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
        total += n
        print(f"  {name:<28} {n:>5} questions -> {args.shards} shards")
    print(f"\nteacher arm: {total} episodes, model {TEACHER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
