"""exp_teacher_only PART C: build the m1 SFT dataset from teacher-only expert traces.

Unlike ``m1_sft/build_dataset.py`` (which assembles a second-person ``teacher_guidance``
block from the guidance recipe), here the trajectory IS the teacher's own reasoning: each
training completion is the 120b-as-student's own thought+action in the plain m1 action
format with NO guidance block -- the same shape ``m2_rft/filter_rollouts.episode_to_examples``
emits. On top of that base builder this adds, for this run:

  * the G-2 empty-category strip (``decision.category == ""`` is rejected by the action
    schema at eval and zeroes every action; the harness stores an omitted category as "");
  * the §6 / G-4 leak gate on the model-*generated* text (the step ``thought`` and the
    plan) -- rejects a turn only if it states the gold answer without the gold already
    being in the question or the student-visible context, catching parametric-knowledge
    guesses. It deliberately does NOT gate the ``finish`` answer itself: that answer is the
    point of a correct episode, and word-boundary containment would over-reject grounded
    answers that phrase the gold slightly differently than the retrieved text;
  * a qid-hash 3% dev split mirroring ``episode_lib.qid_split`` (never splits a question).

Prompts are kept VERBATIM from collection (hidden budget, "This is step N"), exactly how
the guidance-m1 dataset was built, so the ONLY difference between the two recipes' training
data is the trace content, not the prompt rendering (PLAN.md §7). Eval discloses budget 4
for both recipes, so that (small) train/eval wording gap is identical across recipes and
cancels in the within-model paired comparison.

Output (the same files the m1 trainer consumes):
    train.jsonl / dev.jsonl   {"prompt":[system,user],"completion":[assistant],"metadata":{}}
    dev_questions.jsonl       question rows for held-out dev qids (agent-loop eval)
    stats.json                counts + gate/telemetry

Usage:
    .venv_train/bin/python training_methods/exp_teacher_only/build_dataset.py \
        --episodes training_methods/exp_teacher_only/data/selected_correct.jsonl \
        --out      training_methods/exp_teacher_only/data
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402
from training_methods.common.episode_lib import leak_gate_ok, qid_split  # noqa: E402
from agentsim.teacher_guidance.sft_export import DEFAULT_SYSTEM  # noqa: E402
from agentsim.teacher_guidance.json_utils import parse_student_action, parse_student_plan  # noqa: E402

DEFAULT_QUESTIONS = (REPO_ROOT / "data" / "datasets" / "hotpot_teacher_guidance_train3000"
                     / "hotpot_distractor_train_questions.jsonl")


def _visible_context(ep: Dict[str, Any], upto_step_idx: int) -> str:
    """Everything the 120b had seen by the END of step ``upto_step_idx`` (0-based): the
    question, its plan, and every prior student output + tool observation. This is the
    context in which the thought at the NEXT step was written."""
    parts: List[str] = [ep.get("query") or ""]
    plan = ep.get("plan") or {}
    if plan.get("raw"):
        parts.append(str(plan["raw"]))
    for s in (ep.get("steps") or [])[: upto_step_idx + 1]:
        parts.append(json.dumps(s.get("student_raw") or "", ensure_ascii=False))
        parts.append(json.dumps(s.get("tool_observation") or {}, ensure_ascii=False))
    return "\n".join(parts)


def _strip_empty_category(action: Dict[str, Any]) -> Dict[str, Any]:
    """G-2: the harness stores an omitted decision.category as "", but the action schema
    rejects "" (it accepts a *missing* category). Cloning it verbatim teaches an invalid
    value that fails every action at eval, so drop the key when empty."""
    dec = action.get("decision")
    if isinstance(dec, dict) and dec.get("category") == "":
        action = dict(action)
        action["decision"] = {k: v for k, v in dec.items() if k != "category"}
    return action


def _example(user: str, target: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": [{"role": "system", "content": DEFAULT_SYSTEM},
                   {"role": "user", "content": user}],
        "completion": [{"role": "assistant",
                        "content": json.dumps(target, ensure_ascii=False)}],
        "metadata": meta,
    }


def episode_to_examples(ep: Dict[str, Any], stats: collections.Counter) -> List[Dict[str, Any]]:
    """m1-format SFT rows from one teacher-only episode; only structurally valid, non-leaky
    turns are kept."""
    rows: List[Dict[str, Any]] = []
    qid = ep.get("qid")
    gold = ep.get("gold_answer") or ""
    question = ep.get("query") or ""

    # Plan turn: the 120b's own plan, authored before any retrieval.
    plan = ep.get("plan") or {}
    if plan.get("valid") and plan.get("raw") and plan.get("prompt"):
        obj, info = parse_student_plan(plan["raw"])
        if info.get("json_valid") and isinstance(obj, dict) and obj.get("steps"):
            plan_text = json.dumps(obj, ensure_ascii=False)
            # A plan precedes retrieval: gold is safe only if it is in the question.
            if leak_gate_ok(plan_text, gold, question, question):
                rows.append(_example(plan["prompt"], obj,
                                     {"qid": qid, "kind": "plan", "step": 0,
                                      "source": "teacher_only"}))
            else:
                stats["plan_gated_out"] += 1

    # One example per structurally-valid tool step.
    for i, s in enumerate(ep.get("steps") or []):
        if not s.get("action_valid"):
            stats["step_action_invalid"] += 1
            continue
        action, info = parse_student_action(s.get("student_raw") or "")
        if not info.get("action_valid"):
            stats["step_reparse_invalid"] += 1
            continue
        raw_action = action.to_dict()
        if (raw_action.get("decision") or {}).get("category") == "":
            stats["empty_category_stripped"] += 1
        target = _strip_empty_category(raw_action)
        thought = str(target.get("thought") or "")
        if (target.get("decision") or {}).get("parametric_knowledge_used"):
            stats["parametric_knowledge_used_steps"] += 1
        # Gate the model-generated thought: reject only if it states the gold answer
        # without the gold being in the question or the context visible when it was
        # written (the state through the previous step).
        ctx = _visible_context(ep, i - 1)
        if not leak_gate_ok(thought, gold, question, ctx):
            stats["step_thought_gated_out"] += 1
            continue
        tool = (target.get("action") or {}).get("tool")
        rows.append(_example(s["student_prompt"], target,
                             {"qid": qid, "kind": "action", "step": s.get("t", i + 1),
                              "tool": tool, "source": "teacher_only"}))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", required=True,
                    help="selected-correct episodes jsonl (full trace lines)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                    help="the 3000-question file, to emit dev_questions.jsonl rows")
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("teacher_only_build", out / "build.log")

    stats = collections.Counter()
    seen_qids = set()
    examples: List[Dict[str, Any]] = []
    for line in open(args.episodes):
        if not line.strip():
            continue
        ep = json.loads(line)
        qid = ep.get("qid")
        if qid in seen_qids:
            stats["dup_qid_skipped"] += 1
            continue
        seen_qids.add(qid)
        stats["episodes"] += 1
        rows = episode_to_examples(ep, stats)
        stats["examples"] += len(rows)
        examples.extend(rows)

    # qid-hash train/dev split (never splits a question across sides).
    train, dev = [], []
    for e in examples:
        (dev if qid_split(e["metadata"]["qid"], args.dev_fraction) == "dev" else train).append(e)

    with open(out / "train.jsonl", "w") as w:
        for e in train:
            w.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "dev.jsonl", "w") as w:
        for e in dev:
            w.write(json.dumps(e, ensure_ascii=False) + "\n")

    # dev question rows for agent-loop evaluation (exact eval row format).
    dev_qids = {e["metadata"]["qid"] for e in dev}
    n_devq = 0
    with open(out / "dev_questions.jsonl", "w") as w, open(args.questions) as fh:
        for line in fh:
            if json.loads(line)["id"] in dev_qids:
                w.write(line)
                n_devq += 1

    tools = collections.Counter(e["metadata"].get("tool") for e in examples
                                if e["metadata"]["kind"] == "action")
    kinds = collections.Counter(e["metadata"]["kind"] for e in examples)
    summary = {
        "train_examples": len(train),
        "dev_examples": len(dev),
        "dev_questions": n_devq,
        "unique_qids": len(seen_qids),
        "episodes": int(stats["episodes"]),
        "examples": int(stats["examples"]),
        "empty_category_stripped": int(stats["empty_category_stripped"]),
        "step_thought_gated_out": int(stats["step_thought_gated_out"]),
        "plan_gated_out": int(stats["plan_gated_out"]),
        "step_action_invalid": int(stats["step_action_invalid"]),
        "step_reparse_invalid": int(stats["step_reparse_invalid"]),
        "parametric_knowledge_used_steps": int(stats["parametric_knowledge_used_steps"]),
        "dup_qid_skipped": int(stats["dup_qid_skipped"]),
        "kinds": dict(kinds),
        "tools": dict(tools),
        "prompt_rendering": "verbatim_hidden_budget (matches guidance-m1; no re-render)",
    }
    write_json(out / "stats.json", summary)
    log.info(f"DONE: {json.dumps(summary)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
