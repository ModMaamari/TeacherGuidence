"""
Prompt builders for the Teacher Guidance pipeline.

Hard visibility rule, enforced here and by the renderer/leakage checker:

* student prompts are built only from student-visible state
  (no gold answer, no hidden gold doc ids/titles/spans, no teacher diagnosis);
* teacher prompts receive the gold metadata.

All prompts demand a single JSON object as output and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentsim.teacher_guidance.schemas import GuidanceConfig, PlanReviewConfig


_TOOL_REFERENCE = """Available tools (action.tool) and their params:
- decompose: {"sub_questions": ["..."]}
- reformulate: {"queries": ["..."], "reformulation_type": "conceptual|syntactic"}
- search: {"query": "...", "k": 5, "backend": "hotpot_local"}
- extract: {"doc_ids": ["..."], "target_facts": ["verbatim substring from a retrieved doc"]}
- verify: {"claim": "...", "query": "...", "k": 5}
- synthesize: {}
- finish: {"answer": "...", "citations": [{"doc_id": "...", "span": "..."}]}"""

_STUDENT_ACTION_SCHEMA = """Return ONLY a JSON object with this shape (no prose outside JSON):
{
  "thought": "brief private reasoning",
  "decision": {"category": "need_decomposition|need_retrieval|need_reformulation|sufficient_evidence|synthesize|verify|finish",
               "parametric_knowledge_used": false},
  "action": {"tool": "decompose|reformulate|search|extract|verify|synthesize|finish", "params": {}},
  "new_facts_extracted": [{"doc_id": "...", "span": "verbatim span", "fact": "grounded fact"}]
}"""

# A single neutral example, included only to reinforce the JSON FORMAT. Its content is
# generic and unrelated to any dataset question, so it cannot leak gold information.
_STUDENT_EXAMPLE = """Format example only (your content and tool will differ):
{
  "thought": "I should retrieve evidence about the topic before answering.",
  "decision": {"category": "need_retrieval", "parametric_knowledge_used": false},
  "action": {"tool": "search", "params": {"query": "<your search terms>", "k": 5}},
  "new_facts_extracted": []
}"""


def build_student_visible_state(context: Any, step_index: int, budget: int) -> Dict[str, Any]:
    """Assemble the student-visible state dict from a workflow context.

    Reads only student-safe keys from ``context.metadata``; gold metadata is never
    included.
    """
    md = context.metadata
    state: Dict[str, Any] = {
        "question": context.query,
        "step": step_index,
        "budget": budget,
        "disclose_budget": md.get("disclose_budget", True),
        "previous_actions": md.get("previous_actions", []),
        "retrieved_docs": md.get("retrieved_docs", []),
        "extracted_facts": md.get("extracted_facts", []),
        "draft_answer": md.get("draft_answer"),
        "sub_questions": md.get("sub_questions", []),
        "previous_teacher_guidance": md.get("last_teacher_guidance_for_student"),
    }
    if md.get("revised_plan") is not None:
        state["revised_plan"] = md.get("revised_plan")
    return state


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_student_prompt(
    state: Dict[str, Any], guidance_config: GuidanceConfig, force_finish: bool
) -> str:
    parts: List[str] = []
    parts.append(
        "You are an information-seeking retrieval agent solving a question using tools. "
        "You must ground every fact in retrieved documents and must NOT answer from memory."
    )
    parts.append(_TOOL_REFERENCE)
    parts.append("Retrieval backend is 'hotpot_local' (search only the current question's documents).")
    parts.append(f"Question: {state.get('question', '')}")
    # Budget disclosure: in the default mode the student sees its exact step budget; in the
    # hidden-budget mode it only sees the current step number and is told to be efficient
    # and answer as soon as it can (the budget is revealed only on the forced final step).
    if state.get("disclose_budget", True):
        parts.append(f"Step {state.get('step')} of budget {state.get('budget')}.")
    else:
        parts.append(
            f"This is step {state.get('step')}. Work efficiently: retrieve only what you need "
            "and use \"finish\" with your answer as soon as you are confident — do not waste "
            "steps once you can answer."
        )

    if state.get("revised_plan") is not None:
        parts.append("Your current revised plan (follow it unless evidence requires deviating):")
        parts.append(_json(state["revised_plan"]))

    if state.get("expected_plan_step") is not None:
        parts.append(
            "The plan step you are expected to execute now (your action.tool should "
            "match its intended_tool unless tool results require otherwise):"
        )
        parts.append(_json(state["expected_plan_step"]))

    parts.append("Previous actions: " + _json(state.get("previous_actions", [])))
    parts.append("Retrieved documents so far (previews only): " + _json(state.get("retrieved_docs", [])))
    parts.append("Extracted facts so far: " + _json(state.get("extracted_facts", [])))
    if state.get("sub_questions"):
        parts.append("Sub-questions: " + _json(state.get("sub_questions")))
    if state.get("draft_answer"):
        parts.append(f"Current draft answer: {state.get('draft_answer')}")

    prev = state.get("previous_teacher_guidance")
    if prev:
        parts.append("Previous teacher guidance: " + _json(prev))

    if force_finish:
        parts.append(
            "You are at the final budget step. You MUST use action.tool = \"finish\" and "
            "you MUST include a non-empty \"answer\" string with your best answer, even if "
            "you are uncertain — never leave the answer empty. Base it on the "
            "retrieved/extracted evidence."
        )

    parts.append(_STUDENT_ACTION_SCHEMA)
    parts.append(_STUDENT_EXAMPLE)
    return "\n\n".join(parts)


_RAW_TEXT_TRUNCATE = 1500


def build_forced_answer_prompt(state: Dict[str, Any]) -> str:
    """Last-ditch, plain-text final-answer prompt for the forced-finish step.

    Used only when the normal force-finish step failed to yield a committed answer (the
    student emitted another search instead of a ``finish``, and there was no prior
    candidate/draft/extracted fact to fall back on -- which is exactly the case that used
    to leave ``final_answer == "unknown"``). Rather than fabricating "unknown", we ask the
    student model one more time to turn whatever it has already retrieved into its single
    best answer, in free text (no JSON, no tool call) so nothing structural can go wrong.
    """
    parts: List[str] = []
    parts.append(
        "You have run out of retrieval steps. You must now give your single best final "
        "answer to the question, based ONLY on the evidence you already gathered below. "
        "You cannot search again."
    )
    parts.append(f"Question: {state.get('question', '')}")
    parts.append("Retrieved documents (previews): " + _json(state.get("retrieved_docs", [])))
    parts.append("Extracted facts: " + _json(state.get("extracted_facts", [])))
    if state.get("draft_answer"):
        parts.append(f"Draft answer so far: {state.get('draft_answer')}")
    parts.append(
        "Output ONLY your final answer as a short phrase -- no JSON, no code fences, no "
        "explanation, no surrounding quotes. If the evidence is incomplete, still give your "
        "single best answer; do not reply \"unknown\" or \"I don't know\"."
    )
    return "\n\n".join(parts)


def build_teacher_prompt(
    state: Dict[str, Any],
    gold: Dict[str, Any],
    student_action: Dict[str, Any],
    tool_observation: Dict[str, Any],
    guidance_config: GuidanceConfig,
    student_raw: Optional[str] = None,
    student_action_valid: bool = True,
    is_final_answer: bool = False,
) -> str:
    parts: List[str] = []
    parts.append(
        "You are a teacher evaluating one step of a student's retrieval-agent trajectory. "
        "You can see the gold answer and gold supporting facts, which are PRIVATE."
    )
    parts.append(f"Question: {state.get('question', '')}")
    parts.append("Gold metadata (PRIVATE — never reveal hidden gold values to the student):")
    parts.append(_json(gold))
    parts.append(f"Current step {state.get('step')} of budget {state.get('budget')}.")
    if student_action_valid or not student_raw:
        parts.append("Student action: " + _json(student_action))
    else:
        # The student's output failed to parse into a valid action even after a
        # repair retry -- showing the empty/default parsed object here would give the
        # teacher zero signal about what actually happened. Show the real raw text
        # instead so the teacher can still give a specific, useful evaluation.
        truncated = student_raw[:_RAW_TEXT_TRUNCATE]
        if len(student_raw) > _RAW_TEXT_TRUNCATE:
            truncated += "... [truncated]"
        parts.append(
            "Student action: FAILED TO PARSE as a valid action, even after a correction "
            "retry. Here is the student's raw output verbatim -- evaluate it as best you "
            "can and penalize the malformed output in your scoring:\n" + truncated
        )
    parts.append("Tool observation: " + _json(tool_observation))
    parts.append("Retrieved documents so far: " + _json(state.get("retrieved_docs", [])))
    parts.append("Extracted facts so far: " + _json(state.get("extracted_facts", [])))

    parts.append(
        f"Active guidance level: {guidance_config.level}. Score the step with both a binary "
        f"(0/1) and a continuous (0.0-1.0) score. Keep student-visible feedback under "
        f"{guidance_config.max_feedback_words} words and do NOT mention any hidden gold "
        f"answer, title, doc id, or span the student has not retrieved."
    )

    final_judgment_field = ""
    if is_final_answer:
        parts.append(
            "This is the student's FINAL answer. You can see the gold answer, so also judge "
            "whether the student's final answer is correct by comparing it to the gold answer "
            "(accept a correct answer wrapped in explanation or extra words; judge on meaning, "
            "not exact string match). Put your verdict in private_diagnosis as "
            "\"final_answer_correct\" (1 if the final answer is correct, else 0) and "
            "\"final_answer_score\" (0.0-1.0 continuous correctness). These two fields are "
            "PRIVATE and must never be shown to the student."
        )
        final_judgment_field = (
            '"final_answer_correct": 0, "final_answer_score": 0.0, '
        )

    parts.append(
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "guidance_level": <int>,\n'
        '  "student_visible": {"score_binary": 0, "score_continuous": 0.0, "feedback": "...", "hint": null},\n'
        '  "private_diagnosis": {"json_valid": true, "action_valid": true, "tool_valid": true, '
        '"step_correct": false, "retrieved_gold_doc": false, "extracted_gold_fact": false, '
        '"answer_supported": false, ' + final_judgment_field + '"main_error": "...", '
        '"recommended_next_tool": "...", "recommended_next_focus": "...", "gold_answer_leaked": false},\n'
        '  "teacher_decision": "continue|accept_finish|reject_finish|force_finish"\n'
        "}"
    )
    return "\n\n".join(parts)


def _plan_step_cap(state: Dict[str, Any], plan_review_config: PlanReviewConfig) -> int:
    """Cap the plan length at the smaller of the configured max and the step budget."""
    budget = int(state.get("budget") or 0)
    cap = plan_review_config.max_initial_plan_steps
    return min(cap, budget) if budget > 0 else cap


def _budget_clause(state: Dict[str, Any]) -> str:
    budget = int(state.get("budget") or 0)
    if budget <= 0:
        return ""
    return (
        f"You have a budget of {budget} tool-use steps for this question. Your plan MUST be "
        f"executable within {budget} steps (at most {budget} steps; fewer is fine), and the "
        "final step must be 'finish'."
    )


def build_initial_plan_prompt(state: Dict[str, Any], plan_review_config: PlanReviewConfig) -> str:
    parts: List[str] = []
    parts.append(
        "You are a retrieval agent. Before using any tools, write a retrieval-oriented plan. "
        "You have NOT retrieved any evidence yet and must NOT answer from memory."
    )
    parts.append(_TOOL_REFERENCE)
    parts.append(f"Question: {state.get('question', '')}")
    cap = _plan_step_cap(state, plan_review_config)
    if state.get("disclose_budget", True):
        budget_clause = _budget_clause(state)
        if budget_clause:
            parts.append(budget_clause)
        parts.append(
            f"Produce at most {cap} steps describing how to retrieve and verify evidence "
            "(not a final answer)."
        )
    else:
        # Hidden-budget mode: don't reveal the budget; ask for a minimal, efficient plan.
        parts.append(
            "Write the SHORTEST efficient plan that retrieves and verifies just enough "
            f"evidence to answer, then finishes — aim for as few steps as possible (no more "
            f"than {cap}). The final step must be 'finish'. Do not pad the plan."
        )
    parts.append(
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "plan_summary": "...",\n'
        '  "steps": [{"step_id": 1, "goal": "...", "intended_tool": "search", "rationale": "...", "depends_on": []}],\n'
        '  "uncertainties": ["..."],\n'
        '  "stop_condition": "..."\n'
        "}"
    )
    return "\n\n".join(parts)


def build_teacher_plan_prompt(
    state: Dict[str, Any],
    gold: Dict[str, Any],
    guidance_config: GuidanceConfig,
    plan_review_config: PlanReviewConfig,
) -> str:
    """Teacher-planner mode: the teacher authors the full plan the student will follow.

    The plan is shown to the student, so it must NOT contain the gold answer, hidden
    gold titles, or hidden doc ids (the code sanitizes the plan afterwards regardless).
    """
    parts: List[str] = []
    parts.append(
        "You are a teacher writing a complete, retrieval-oriented plan FOR THE STUDENT to "
        "follow. You can see the gold metadata, which is PRIVATE: do NOT reveal the gold "
        "answer, hidden gold titles, or hidden doc ids in the plan — describe how to "
        "retrieve and verify evidence, not the answer itself."
    )
    parts.append(_TOOL_REFERENCE)
    parts.append(f"Question: {state.get('question', '')}")
    parts.append("Gold metadata (PRIVATE — for your planning only):")
    parts.append(_json(gold))
    budget_clause = _budget_clause(state)
    if budget_clause:
        parts.append(budget_clause)
    parts.append(
        f"Produce at most {_plan_step_cap(state, plan_review_config)} ordered steps the "
        "student should execute with the tools above."
    )
    parts.append(
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "plan_summary": "...",\n'
        '  "steps": [{"step_id": 1, "goal": "...", "intended_tool": "search", "rationale": "...", "depends_on": []}],\n'
        '  "uncertainties": ["..."],\n'
        '  "stop_condition": "..."\n'
        "}"
    )
    return "\n\n".join(parts)


def build_plan_review_prompt(
    state: Dict[str, Any],
    gold: Dict[str, Any],
    initial_plan: Dict[str, Any],
    guidance_config: GuidanceConfig,
    plan_review_config: PlanReviewConfig,
) -> str:
    parts: List[str] = []
    parts.append(
        "You are a teacher reviewing the student's initial plan before any tool use. "
        "You can see gold metadata, which is PRIVATE."
    )
    parts.append(f"Question: {state.get('question', '')}")
    parts.append("Gold metadata (PRIVATE):")
    parts.append(_json(gold))
    parts.append("Student initial plan:")
    parts.append(_json(initial_plan))
    budget = int(state.get("budget") or 0)
    if budget > 0:
        parts.append(
            f"The student has a budget of {budget} tool-use steps. A plan that needs more "
            f"than {budget} steps cannot be fully executed — if so, ask the student to "
            "tighten it to fit the budget."
        )
    allowed = []
    if plan_review_config.allow_teacher_to_suggest_tools:
        allowed.append("suggest tools")
    if plan_review_config.allow_teacher_to_suggest_queries:
        allowed.append("suggest queries")
    parts.append(
        f"Active plan-review guidance level: {guidance_config.level}. You MAY: "
        f"{', '.join(allowed) or 'give feedback only'}. Do NOT reveal the gold answer, "
        "gold titles, or gold doc ids unless explicitly allowed."
    )
    parts.append(
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "plan_review_enabled": true,\n'
        f'  "review_guidance_level": {guidance_config.level},\n'
        '  "student_visible": {"score_binary": 0, "score_continuous": 0.0, "feedback": "...", "hint": null},\n'
        '  "private_diagnosis": {"plan_valid": true, "uses_tools": true, "premature_answering_risk": false, '
        '"covers_decomposition": true, "covers_retrieval": true, "covers_extraction": true, '
        '"covers_verification": false, "covers_synthesis": true, "missing_capabilities": [], '
        '"gold_answer_leaked": false, "main_error": "...", "recommended_plan_changes": ["..."]},\n'
        '  "teacher_decision": "accept_plan|revise_plan|reject_plan"\n'
        "}"
    )
    return "\n\n".join(parts)


def build_revised_plan_prompt(
    state: Dict[str, Any],
    initial_plan: Dict[str, Any],
    rendered_plan_feedback: Dict[str, Any],
    plan_review_config: PlanReviewConfig,
) -> str:
    parts: List[str] = []
    parts.append(
        "Revise your plan using the teacher's feedback. You still have NOT retrieved "
        "evidence and must NOT answer from memory."
    )
    parts.append(f"Question: {state.get('question', '')}")
    parts.append("Your initial plan:")
    parts.append(_json(initial_plan))
    parts.append("Teacher feedback (student-visible only):")
    parts.append(_json(rendered_plan_feedback))
    budget_clause = _budget_clause(state)
    if budget_clause:
        parts.append(budget_clause)
    revised_cap = min(plan_review_config.max_revised_plan_steps, int(state.get("budget") or 0)) \
        if int(state.get("budget") or 0) > 0 else plan_review_config.max_revised_plan_steps
    parts.append(
        f"Produce at most {revised_cap} steps.\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "revision_summary": "...",\n'
        '  "plan_summary": "...",\n'
        '  "steps": [{"step_id": 1, "goal": "...", "intended_tool": "search", "rationale": "...", "depends_on": []}],\n'
        '  "teacher_feedback_used": ["..."],\n'
        '  "stop_condition": "..."\n'
        "}"
    )
    return "\n\n".join(parts)
