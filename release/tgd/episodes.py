"""Episode-file utilities: the publishable view of a raw harness episode, and the
consolidation of many per-question run directories into one file.

A raw episode written by the harness keeps everything needed to debug a run: full
provider response bodies, per-call cost, the fallback chain. The *publishable* view keeps
what research needs -- every prompt, every model output, the teacher's guidance, the
leakage checks, token counts and latencies, all metrics -- and drops:

* ``raw_response`` on every call (provider ids, duplicated text, megabytes per episode)
* ``usage.cost`` (a property of the provider that served the call, not of the data)
* ``teacher_router`` (the fallback chain; ``teacher_models_used`` records what answered)
* ``framework_commit`` (a source-control id of the generating checkout)

The privileged teacher fields (``teacher_prompt``, ``teacher_raw``,
``teacher_private_diagnosis`` and the plan-review equivalents) CONTAIN THE GOLD ANSWER.
They are kept in the published episodes because they are the teacher's actual behaviour;
the SFT builder (``episode_lib``) never copies them into a training target -- it uses only
the student-visible guidance and applies a leakage gate.
"""
from __future__ import annotations

import copy
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"
CALL_DROP = ("raw_response",)
USAGE_DROP = ("cost",)
EPISODE_DROP = ("teacher_router", "framework_commit")
PRIVILEGED_STEP = ("teacher_prompt", "teacher_raw", "teacher_private_diagnosis")
PRIVILEGED_PLAN = ("teacher_plan_review_prompt", "teacher_plan_review_raw", "teacher_plan_review_full")


def _clean_calls(calls: Any) -> Any:
    if not isinstance(calls, list):
        return calls
    out = []
    for c in calls:
        if not isinstance(c, dict):
            out.append(c)
            continue
        c = {k: v for k, v in c.items() if k not in CALL_DROP}
        if isinstance(c.get("usage"), dict):
            c["usage"] = {k: v for k, v in c["usage"].items() if k not in USAGE_DROP}
        out.append(c)
    return out


def publishable(episode: Dict[str, Any], *, strip_privileged: bool = False) -> Dict[str, Any]:
    """Return the publishable form of one raw episode (see module docstring)."""
    ep = copy.deepcopy(episode)
    for key in EPISODE_DROP:
        ep.pop(key, None)
    for step in ep.get("steps") or []:
        for key in ("student_calls", "teacher_calls"):
            if key in step:
                step[key] = _clean_calls(step[key])
        if isinstance(step.get("wiki_update_call"), dict):
            step["wiki_update_call"] = _clean_calls([step["wiki_update_call"]])[0]
        if strip_privileged:
            for key in PRIVILEGED_STEP:
                step.pop(key, None)
    plan = ep.get("plan_review")
    if isinstance(plan, dict):
        for key in ("initial_plan_calls", "revision_calls", "review_calls"):
            if key in plan:
                plan[key] = _clean_calls(plan[key])
        for rnd in plan.get("rounds") or []:
            for key in ("initial_plan_calls", "review_calls", "revision_calls"):
                if key in rnd:
                    rnd[key] = _clean_calls(rnd[key])
            if strip_privileged:
                for key in PRIVILEGED_PLAN:
                    rnd.pop(key, None)
        if strip_privileged:
            for key in PRIVILEGED_PLAN:
                plan.pop(key, None)
    return ep


MODEL_FIELDS = ("student_model", "teacher_model")
MODEL_LIST_FIELDS = ("teacher_models_used",)


def normalize_models(episode: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Rewrite model ids everywhere they occur (episode fields, per-step
    ``teacher_models_used``, every call record's ``model``). Use it to strip provider
    routing prefixes before publishing, e.g. ``{"vllm/student": "org/model-name"}``."""
    if not mapping:
        return episode

    def fix(v):
        return mapping.get(v, v) if isinstance(v, str) else v

    def fix_calls(calls):
        for c in calls or []:
            if isinstance(c, dict) and "model" in c:
                c["model"] = fix(c["model"])

    for k in MODEL_FIELDS:
        if k in episode:
            episode[k] = fix(episode[k])
    for k in MODEL_LIST_FIELDS:
        if isinstance(episode.get(k), list):
            episode[k] = sorted({fix(v) for v in episode[k]})
    for step in episode.get("steps") or []:
        if isinstance(step.get("teacher_models_used"), list):
            step["teacher_models_used"] = sorted({fix(v) for v in step["teacher_models_used"]})
        for k in ("student_calls", "teacher_calls"):
            fix_calls(step.get(k))
    pr = episode.get("plan_review") or {}
    if isinstance(pr, dict):
        for k in ("initial_plan_calls", "revision_calls", "review_calls"):
            fix_calls(pr.get(k))
        for rnd in pr.get("rounds") or []:
            for k in ("initial_plan_calls", "revision_calls", "review_calls"):
                fix_calls(rnd.get(k))
    return episode


def iter_run_episodes(run_roots: Iterable[str | Path]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(file, episode)`` for every episode under the given run directories."""
    for root in run_roots:
        for f in sorted(glob.glob(str(Path(root) / "**" / EPISODE_FILENAME), recursive=True)):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield f, json.loads(line)


def is_error(ep: Dict[str, Any]) -> bool:
    return str(ep.get("stop_reason", "")).startswith("error") or bool(ep.get("error"))


def summary_row(ep: Dict[str, Any]) -> Dict[str, Any]:
    fm = ep.get("final_metrics") or {}
    return {
        "dataset": ep.get("dataset"), "qid": ep.get("qid"),
        "correct": bool(fm.get("answer_correct")), "exact_match": bool(fm.get("exact_match")),
        "f1": fm.get("f1"), "grounded": bool(fm.get("answer_grounded")),
        "steps": ep.get("used_steps"), "stop_reason": ep.get("stop_reason"),
        "student_model": ep.get("student_model"),
        "teacher_models_used": ep.get("teacher_models_used"),
    }
