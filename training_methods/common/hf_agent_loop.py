"""Teacherless agent loop driven by a HuggingFace transformers policy.

Re-uses the EXACT environment pieces of the simulation harness -- prompt renderer
(``build_student_prompt``), deterministic tool executor (``execute_student_tool``),
per-question local retrieval (``HotpotLocalRetriever``) and the HotpotQA metrics --
but generates student actions with a local HF model (base checkpoint or base+LoRA
adapter) instead of Ollama, and with NO teacher anywhere. This is the evaluation
and rollout engine shared by every training method:

  * m1/m3/m5 evaluation (exp100 subset, golden-100 hardest questions)
  * m2 rejection-sampling rollout generation
  * m4 GRPO on-policy rollouts

Episodes mirror the harness export shape (qid/query/gold_answer/final_answer/
steps/stop_reason/used_steps + final_metrics with em/f1/cover/doc_recall).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentsim.workflow.context import WorkflowContext  # noqa: E402
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever  # noqa: E402
from agentsim.teacher_guidance.prompts import (  # noqa: E402
    build_initial_plan_prompt,
    build_student_prompt,
    build_student_visible_state,
)
from agentsim.teacher_guidance.json_utils import (  # noqa: E402
    parse_student_action,
    parse_student_plan,
)
from agentsim.teacher_guidance.tool_executor import (  # noqa: E402
    derive_final_answer,
    execute_student_tool,
)
from agentsim.teacher_guidance.schemas import GuidanceConfig, PlanReviewConfig  # noqa: E402
from agentsim.teacher_guidance.metrics import (  # noqa: E402
    cover_match,
    exact_match,
    f1_score,
    supporting_doc_recall,
)
from agentsim.teacher_guidance.sft_export import DEFAULT_SYSTEM  # noqa: E402

INVALID_RETRY_NOTE = (
    "\n\nYour previous reply was not a valid action JSON. Output ONLY one JSON object "
    "with the exact schema shown above."
)


def _safe_parse_action(raw: str):
    """parse_student_action, hardened against degenerate shapes (e.g. "action" being
    a string) that make StudentAction.from_dict raise instead of flagging invalid.

    Also normalizes decision.category == "" to an omitted field before validation:
    the harness stores an omitted category as "" (StudentAction.from_dict default),
    the training targets reproduce that verbatim, and the pydantic schema accepts a
    missing category but rejects the empty string.
    """
    from agentsim.teacher_guidance.json_utils import parse_json_object, validate_student_action
    from agentsim.teacher_guidance.schemas import StudentAction

    try:
        obj, info = parse_json_object(raw)
        if info.get("json_valid") and isinstance(obj, dict):
            dec = obj.get("decision")
            if isinstance(dec, dict) and dec.get("category") == "":
                dec.pop("category")
        valid, errors = validate_student_action(obj) if info["json_valid"] else (False, info["errors"])
        info["action_valid"] = valid
        info["errors"] = list(info.get("errors", [])) + [e for e in errors if e not in info.get("errors", [])]
        return StudentAction.from_dict(obj), info
    except Exception as exc:  # noqa: BLE001 -- any malformed output is just invalid
        return None, {"action_valid": False, "errors": [f"unparseable: {type(exc).__name__}"]}


class PolicyModel:
    """A HF causal-LM policy (optionally with a PEFT/LoRA adapter on top)."""

    def __init__(
        self,
        model_path: str = "ibm-granite/granite-4.1-3b",
        adapter_path: Optional[str] = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype), device_map=device
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.device = device

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 700,
        temperature: float = 0.0,
    ) -> str:
        import torch

        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        return self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


def _messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def run_episode(
    policy: PolicyModel,
    question_row: Dict[str, Any],
    retriever: HotpotLocalRetriever,
    budget: int = 4,
    disclose_budget: bool = True,
    with_plan: bool = True,
    temperature: float = 0.0,
    max_new_tokens: int = 700,
    logger=None,
) -> Dict[str, Any]:
    """Run one teacherless episode; returns an episode dict with metrics."""
    qid = question_row["id"]
    gold = question_row.get("answer", "") or (question_row.get("gold") or {}).get("answer", "")
    gold_doc_ids = set((question_row.get("gold") or {}).get("gold_doc_ids", []) or [])

    ctx = WorkflowContext(
        task_id=qid,
        query=question_row["query"],
        metadata={
            "retrieval_scope": question_row.get("retrieval_scope", {}),
            "disclose_budget": disclose_budget,
        },
    )
    started = time.time()
    steps: List[Dict[str, Any]] = []
    plan_record: Optional[Dict[str, Any]] = None

    if with_plan:
        state = build_student_visible_state(ctx, 0, budget)
        pr_cfg = PlanReviewConfig(
            enabled=True, max_initial_plan_steps=budget, max_revised_plan_steps=budget
        )
        plan_prompt = build_initial_plan_prompt(state, pr_cfg)
        plan_raw = policy.generate(_messages(plan_prompt), max_new_tokens, temperature)
        plan_obj, plan_info = parse_student_plan(plan_raw)
        if plan_info.get("json_valid") and isinstance(plan_obj, dict) and plan_obj.get("steps"):
            ctx.metadata["revised_plan"] = plan_obj
        plan_record = {"prompt": plan_prompt, "raw": plan_raw, "valid": bool(plan_info.get("json_valid"))}

    stop_reason = "budget_forced_finish"
    final_answer = ""
    for t in range(1, budget + 1):
        force_finish = t == budget
        state = build_student_visible_state(ctx, t, budget)
        prompt = build_student_prompt(state, GuidanceConfig(), force_finish)
        raw = policy.generate(_messages(prompt), max_new_tokens, temperature)
        action, info = _safe_parse_action(raw)
        if not info.get("action_valid"):
            # one retry with an explicit format reminder
            raw = policy.generate(_messages(prompt + INVALID_RETRY_NOTE), max_new_tokens, temperature)
            action, info = _safe_parse_action(raw)

        step_rec: Dict[str, Any] = {
            "t": t,
            "student_prompt": prompt,
            "student_raw": raw,
            "student_action": action.to_dict() if action is not None else {},
            "action_valid": bool(info.get("action_valid")) and action is not None,
        }
        if step_rec["action_valid"]:
            obs = execute_student_tool(ctx, action, retriever)
            step_rec["tool_observation"] = obs
            tool = action.action.tool
        else:
            step_rec["tool_observation"] = {"tool": None, "status": "invalid_action", "errors": info.get("errors", [])}
            tool = None
        steps.append(step_rec)
        if logger:
            logger.info(f"qid={qid} step={t}/{budget} tool={tool} valid={step_rec['action_valid']}")

        if tool == "finish":
            final_answer = ctx.metadata.get("final_answer", "")
            stop_reason = "finish" if not force_finish else "budget_forced_finish"
            break
        if force_finish:
            # student failed to finish on the forced step: derive best-effort answer
            final_answer = derive_final_answer(ctx)
            stop_reason = "budget_forced_finish_no_finish"

    if not final_answer:
        final_answer = ctx.metadata.get("final_answer") or derive_final_answer(ctx)

    retrieved_ids = set(ctx.metadata.get("retrieved_doc_ids", []) or [])
    metrics = {
        "exact_match": bool(exact_match(final_answer, gold)),
        "f1": round(f1_score(final_answer, gold), 4),
        "cover_match": bool(cover_match(final_answer, gold)),
        "doc_recall": round(supporting_doc_recall(retrieved_ids, gold_doc_ids), 4) if gold_doc_ids else None,
    }
    return {
        "qid": qid,
        "query": question_row["query"],
        "gold_answer": gold,
        "final_answer": final_answer,
        "budget": budget,
        "used_steps": len(steps),
        "stop_reason": stop_reason,
        "plan": plan_record,
        "steps": steps,
        "final_metrics": metrics,
        "elapsed_s": round(time.time() - started, 2),
    }


def load_questions(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows
