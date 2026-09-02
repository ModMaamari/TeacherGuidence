"""Shared episode loading + internal-thought example assembly.

Turns teacher-guided episodes into *teacherless* training examples in which the
teacher's step feedback is kept verbatim (second person) as a ``teacher_guidance``
block that the student itself must generate BEFORE its thought/action -- the
"guidance as internal thoughts" format:

    input   = stored student prompt with the ``Previous teacher guidance:`` block removed
    output  = {"teacher_guidance": {...}, "thought": ..., "decision": ..., "action": ...,
               "new_facts_extracted": [...]}

Key rules (documented in the method reports):
  * guidance about step t is attached to step t+1 (causality); step 1 gets the
    plan-review feedback; the final step's outgoing feedback is unused.
  * ``[answer hidden]`` placeholders are restored from the gold answer ONLY when
    echo-safe -- the answer already appeared in the question, the student's own
    prior output, or a retrieved document. Otherwise the placeholder sentence is
    dropped entirely (never train on the literal placeholder token).
  * a leakage gate rejects any example whose guidance/thought mentions the gold
    answer without being echo-safe.

The runtime action parser (``StudentActionModel``, extra="ignore") tolerates the
extra ``teacher_guidance`` key, so trained models remain drop-in compatible with
the environment.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.leakage import _contains  # noqa: E402
from agentsim.teacher_guidance.sft_internalize import strip_teacher_guidance_block  # noqa: E402
from agentsim.teacher_guidance.sft_export import DEFAULT_SYSTEM  # noqa: E402

PLACEHOLDER = "[answer hidden]"

TEACHER_CORRECT_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------
def iter_run_episodes(run_root: str | Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(sample_dir, episode)`` from a consolidated run directory."""
    pattern = str(Path(run_root) / "run" / "hotpot_questions" / "sample_*" /
                  "teacher_guidance_episodes.jsonl")
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    yield str(Path(f).parent), json.loads(line)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def teacher_score(episode: Dict[str, Any]) -> Optional[float]:
    fm = episode.get("final_metrics") or {}
    v = fm.get("teacher_answer_score")
    return float(v) if v is not None else None


def teacher_correct(episode: Dict[str, Any]) -> bool:
    s = teacher_score(episode)
    return s is not None and s >= TEACHER_CORRECT_THRESHOLD


# ---------------------------------------------------------------------------
# Echo-safe restoration of the [answer hidden] placeholder
# ---------------------------------------------------------------------------
def visible_context_through_step(episode: Dict[str, Any], step_idx: int) -> str:
    """Everything the student had seen by the END of step ``step_idx`` (0-based):
    the question, its own raw outputs, and all tool observations so far. This is
    the context in which the guidance produced at ``step_idx`` was rendered."""
    parts = [episode.get("query") or ""]
    pr = episode.get("plan_review") or {}
    for key in ("initial_student_plan_raw", "revised_student_plan_raw"):
        if pr.get(key):
            parts.append(str(pr[key]))
    for s in (episode.get("steps") or [])[: step_idx + 1]:
        parts.append(json.dumps(s.get("student_raw") or "", ensure_ascii=False))
        parts.append(json.dumps(s.get("tool_observation") or {}, ensure_ascii=False))
    return "\n".join(parts)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def restore_placeholder(feedback: str, gold: str, visible_context: str) -> Tuple[str, Dict[str, int]]:
    """Restore ``[answer hidden]`` from ``gold`` when echo-safe, else drop the
    sentence containing it. Returns (clean_feedback, stats)."""
    stats = {"restored": 0, "dropped_sentences": 0}
    if PLACEHOLDER not in feedback:
        return feedback, stats
    if _contains(visible_context, gold):
        stats["restored"] = feedback.count(PLACEHOLDER)
        return feedback.replace(PLACEHOLDER, gold), stats
    kept = []
    for sent in _SENT_SPLIT.split(feedback):
        if PLACEHOLDER in sent:
            stats["dropped_sentences"] += 1
        else:
            kept.append(sent)
    return " ".join(kept).strip(), stats


def leak_gate_ok(text: str, gold: str, question: str, visible_context: str) -> bool:
    """True when ``text`` is safe to train on: it may mention the gold answer only
    if the answer is already in the question or the student-visible context."""
    if not gold or not gold.strip():
        return True
    if not _contains(text, gold):
        return True
    return _contains(question, gold) or _contains(visible_context, gold)


# ---------------------------------------------------------------------------
# Internal-thought example assembly
# ---------------------------------------------------------------------------
def _guidance_into_step(episode: Dict[str, Any], i: int) -> Optional[Dict[str, Any]]:
    """Guidance the student saw going INTO step i (0-based): the plan-review
    feedback for step 0, the previous step's feedback otherwise."""
    steps = episode.get("steps") or []
    if i <= 0:
        pr = episode.get("plan_review") or {}
        return pr.get("student_visible_plan_feedback")
    return steps[i - 1].get("student_visible_guidance")


def _ordered_target(guidance: Optional[Dict[str, Any]], action: Dict[str, Any]) -> Dict[str, Any]:
    """Target JSON with teacher_guidance FIRST (generation order matters)."""
    target: Dict[str, Any] = {}
    if guidance:
        target["teacher_guidance"] = guidance
    for key in ("thought", "decision", "action", "new_facts_extracted"):
        if key in action:
            target[key] = action[key]
    # storage records an omitted decision.category as ""; the action schema rejects
    # the empty string, so don't teach the model to emit it
    dec = target.get("decision")
    if isinstance(dec, dict) and dec.get("category") == "":
        target["decision"] = {k: v for k, v in dec.items() if k != "category"}
    for key, val in action.items():  # keep any remaining keys, stable order
        if key not in target:
            target[key] = val
    return target


def build_step_example(
    episode: Dict[str, Any],
    i: int,
    run: str = "",
    keep_score: bool = True,
) -> Optional[Dict[str, Any]]:
    """One prompt/completion example for step ``i`` (0-based), or None if the
    step is unusable (no action, or fails the leakage gate)."""
    steps = episode.get("steps") or []
    if i >= len(steps):
        return None
    step = steps[i]
    action = step.get("student_action") or {}
    if not (action.get("action") or {}).get("tool"):
        return None

    gold = episode.get("gold_answer") or ""
    question = episode.get("query") or ""
    prompt = strip_teacher_guidance_block(step.get("student_prompt") or "")
    if not prompt:
        return None

    # Context in which the incoming guidance was written: end of step i-1
    # (plan-stage context for step 0).
    ctx = visible_context_through_step(episode, i - 1)

    guidance_in = _guidance_into_step(episode, i)
    clean_guidance: Optional[Dict[str, Any]] = None
    restore_stats = {"restored": 0, "dropped_sentences": 0}
    if isinstance(guidance_in, dict) and (guidance_in.get("feedback") or "").strip():
        fb, restore_stats = restore_placeholder(str(guidance_in["feedback"]), gold, ctx)
        if fb.strip():
            clean_guidance = {"feedback": fb.strip()}
            # step 0's incoming guidance is plan-review feedback, whose stored score
            # is a 0.0 default (not a judgment) -- teaching it miscalibrates the model
            if keep_score and i > 0 and isinstance(guidance_in.get("score"), (int, float)):
                clean_guidance = {"score": guidance_in["score"], "feedback": fb.strip()}

    # Leakage gate on the parts the model must generate unprompted: guidance + thought.
    gate_text = json.dumps(clean_guidance or {}, ensure_ascii=False) + "\n" + str(action.get("thought") or "")
    if not leak_gate_ok(gate_text, gold, question, ctx):
        return None

    target = _ordered_target(clean_guidance, action)
    # The student occasionally parrots the literal placeholder from its prompt into
    # its own thought/params; never train on that token.
    if PLACEHOLDER in json.dumps(target, ensure_ascii=False):
        return None
    return {
        "prompt": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ],
        "metadata": {
            "qid": episode.get("qid"),
            "run": run,
            "kind": "action",
            "step": step.get("t", i + 1),
            "tool": (action.get("action") or {}).get("tool"),
            "step_teacher_score": (step.get("student_visible_guidance") or {}).get("score"),
            "restored": restore_stats["restored"],
            "dropped_sentences": restore_stats["dropped_sentences"],
            "had_guidance": clean_guidance is not None,
        },
    }


def build_plan_example(episode: Dict[str, Any], run: str = "") -> Optional[Dict[str, Any]]:
    """Plan turn: teacher-free plan prompt -> final (revised, teacher-approved) plan."""
    pr = episode.get("plan_review") or {}
    prompt = pr.get("initial_student_plan_prompt")
    plan = pr.get("revised_student_plan") or pr.get("initial_student_plan")
    if not (pr.get("enabled") and prompt and plan):
        return None
    gold = episode.get("gold_answer") or ""
    question = episode.get("query") or ""
    plan_text = json.dumps(plan, ensure_ascii=False)
    # A plan is written before any retrieval: mentioning the gold answer there is
    # only safe if the question itself contains it. Revised plans can also parrot
    # the literal [answer hidden] placeholder from sanitized plan feedback.
    if PLACEHOLDER in plan_text or not leak_gate_ok(plan_text, gold, question, question):
        return None
    return {
        "prompt": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": strip_teacher_guidance_block(prompt)},
        ],
        "completion": [{"role": "assistant", "content": plan_text}],
        "metadata": {"qid": episode.get("qid"), "run": run, "kind": "plan", "step": 0},
    }


def build_episode_examples(episode: Dict[str, Any], run: str = "") -> List[Dict[str, Any]]:
    out = []
    plan_ex = build_plan_example(episode, run)
    if plan_ex:
        out.append(plan_ex)
    for i in range(len(episode.get("steps") or [])):
        ex = build_step_example(episode, i, run)
        if ex:
            out.append(ex)
    return out


def qid_split(qid: str, dev_fraction: float = 0.03) -> str:
    """Deterministic train/dev assignment by qid hash (never split one question
    across train and dev)."""
    h = int(hashlib.sha256(str(qid).encode()).hexdigest(), 16) % 10_000
    return "dev" if h < dev_fraction * 10_000 else "train"
