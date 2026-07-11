"""m3_kto_dpo dataset builder: step-level KTO labels + same-prompt DPO pairs.

KTO (primary). Every teacher-scored step across ALL six runs becomes a standalone
(prompt, completion, label) example -- no pairing needed, which fits our data: one
trajectory per (question, run), heterogeneous prompts across runs.
  * completion format = m1's guidance-as-internal-thought target (so KTO stacks
    cleanly on top of the m1 SFT adapter)
  * label True  <=> the step's own teacher score >= --pos-threshold (default 0.9)
  * label False <=> score <= --neg-threshold (default 0.2); middle scores skipped
  * this is the only offline method that LEARNS FROM THE ~40k rejected steps.

DPO (secondary). Pairs require an identical prompt for chosen/rejected, which our
runs almost never give naturally (budget lines differ). Two honest constructions:
  * plan pairs: chosen = plan of a teacher-correct episode; rejected = plan of a
    teacher-wrong episode of the SAME question, presented under the chosen prompt
    (plans are budget-capped lists; mild approximation, documented).
  * finish pairs: prompt = the correct episode's final-step state; chosen = its
    finish action; rejected = the same action JSON with the answer replaced by a
    wrong episode's final answer (synthetic negative isolating the answer choice).

Outputs in --out (default training_methods/m3_kto_dpo/data):
    kto_train.jsonl / kto_dev.jsonl   {"prompt", "completion", "label", "metadata"}
    dpo_train.jsonl / dpo_dev.jsonl   {"prompt", "chosen", "rejected", "metadata"}
    stats.json
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402
from training_methods.common.episode_lib import (  # noqa: E402
    RUN_ROOTS,
    build_plan_example,
    build_step_example,
    iter_run_episodes,
    qid_split,
    teacher_correct,
)
from agentsim.teacher_guidance.metrics import cover_match  # noqa: E402


def step_score(episode, i):
    g = (episode.get("steps") or [])[i].get("student_visible_guidance") or {}
    s = g.get("score")
    return float(s) if isinstance(s, (int, float)) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--pos-threshold", type=float, default=0.9)
    ap.add_argument("--neg-threshold", type=float, default=0.2)
    ap.add_argument("--max-per-label", type=int, default=20000)
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("m3_build", out / "build.log")
    rng = random.Random(args.seed)

    kto_pos, kto_neg = [], []
    by_qid: dict = collections.defaultdict(lambda: {"correct": [], "wrong": []})
    stats = collections.Counter()

    for run, root in RUN_ROOTS.items():
        n_eps = 0
        for _, ep in iter_run_episodes(REPO_ROOT / root):
            n_eps += 1
            bucket = "correct" if teacher_correct(ep) else "wrong"
            by_qid[ep.get("qid")][bucket].append((run, ep))
            for i in range(len(ep.get("steps") or [])):
                s = step_score(ep, i)
                if s is None:
                    stats["unscored_steps"] += 1
                    continue
                if s >= args.pos_threshold:
                    label = True
                elif s <= args.neg_threshold:
                    label = False
                else:
                    stats["midscore_skipped"] += 1
                    continue
                ex = build_step_example(ep, i, run)
                if ex is None:
                    stats["gated_out"] += 1
                    continue
                ex["label"] = label
                ex["metadata"]["step_score"] = s
                (kto_pos if label else kto_neg).append(ex)
        log.info(f"scanned {run}: {n_eps} episodes | pos={len(kto_pos)} neg={len(kto_neg)}")

    rng.shuffle(kto_pos)
    rng.shuffle(kto_neg)
    kto = kto_pos[: args.max_per_label] + kto_neg[: args.max_per_label]
    rng.shuffle(kto)
    log.info(f"KTO: {len(kto)} examples "
             f"(pos {min(len(kto_pos), args.max_per_label)}, neg {min(len(kto_neg), args.max_per_label)})")

    # ------------------------------------------------------------------ DPO pairs
    dpo = []
    for qid, d in by_qid.items():
        if not d["correct"] or not d["wrong"]:
            continue
        # best correct = highest teacher score then fewest steps; worst wrong = lowest
        c_run, c_ep = max(d["correct"], key=lambda t: ((t[1].get("final_metrics") or {}).get("teacher_answer_score") or 0, -len(t[1].get("steps") or [])))
        w_run, w_ep = min(d["wrong"], key=lambda t: ((t[1].get("final_metrics") or {}).get("teacher_answer_score") or 0))

        # plan pair
        c_plan = build_plan_example(c_ep, c_run)
        w_plan = build_plan_example(w_ep, w_run)
        if c_plan and w_plan and c_plan["completion"][0]["content"] != w_plan["completion"][0]["content"]:
            dpo.append({
                "prompt": c_plan["prompt"],
                "chosen": c_plan["completion"],
                "rejected": w_plan["completion"],
                "metadata": {"qid": qid, "kind": "plan", "chosen_run": c_run, "rejected_run": w_run},
            })

        # finish pair (synthetic negative: same state, wrong answer substituted)
        steps = c_ep.get("steps") or []
        fin_i = next((i for i in range(len(steps) - 1, -1, -1)
                      if ((steps[i].get("student_action") or {}).get("action") or {}).get("tool") == "finish"), None)
        wrong_answer = (w_ep.get("final_answer") or "").strip()
        gold = c_ep.get("gold_answer") or ""
        if fin_i is None or not wrong_answer or cover_match(wrong_answer, gold):
            stats["finish_pair_skipped"] += 1
            continue
        ex = build_step_example(c_ep, fin_i, c_run)
        if ex is None:
            stats["finish_pair_skipped"] += 1
            continue
        chosen_obj = json.loads(ex["completion"][0]["content"])
        params = ((chosen_obj.get("action") or {}).get("params") or {})
        if not (params.get("answer") or "").strip():
            stats["finish_pair_skipped"] += 1
            continue
        rejected_obj = copy.deepcopy(chosen_obj)
        rejected_obj["action"]["params"]["answer"] = wrong_answer
        dpo.append({
            "prompt": ex["prompt"],
            "chosen": [{"role": "assistant", "content": json.dumps(chosen_obj, ensure_ascii=False)}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected_obj, ensure_ascii=False)}],
            "metadata": {"qid": qid, "kind": "finish", "chosen_run": c_run,
                         "rejected_run": w_run, "wrong_answer": wrong_answer[:120]},
        })

    log.info(f"DPO: {len(dpo)} pairs "
             f"({sum(1 for p in dpo if p['metadata']['kind']=='plan')} plan, "
             f"{sum(1 for p in dpo if p['metadata']['kind']=='finish')} finish)")

    def dump(rows, name):
        tr, dv = [], []
        for r in rows:
            (dv if qid_split(r["metadata"]["qid"], args.dev_fraction) == "dev" else tr).append(r)
        for split, data in (("train", tr), ("dev", dv)):
            with open(out / f"{name}_{split}.jsonl", "w") as w:
                for r in data:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(tr), len(dv)

    kto_n = dump(kto, "kto")
    dpo_n = dump(dpo, "dpo")

    summary = {
        "kto_train": kto_n[0], "kto_dev": kto_n[1],
        "kto_pos_total": len(kto_pos), "kto_neg_total": len(kto_neg),
        "pos_threshold": args.pos_threshold, "neg_threshold": args.neg_threshold,
        "dpo_train": dpo_n[0], "dpo_dev": dpo_n[1],
        "dpo_plan_pairs": sum(1 for p in dpo if p["metadata"]["kind"] == "plan"),
        "dpo_finish_pairs": sum(1 for p in dpo if p["metadata"]["kind"] == "finish"),
        "questions_with_both": sum(1 for d in by_qid.values() if d["correct"] and d["wrong"]),
        **{k: int(v) for k, v in stats.items()},
    }
    write_json(out / "stats.json", summary)
    log.info(f"DONE: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
