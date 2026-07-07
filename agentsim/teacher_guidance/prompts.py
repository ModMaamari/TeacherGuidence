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

# Wiki-enabled variants: identical to the above plus the wiki_read/wiki_write tools
# (and the manage_wiki decision category). The baseline text stays byte-identical so
# wiki-disabled runs are unaffected.
_WIKI_TOOL_REFERENCE_EXTRA = """- wiki_read: {}  (read your wiki.md notes)
- wiki_write: {"content": "..."}  (replace your wiki.md notes with new content)"""

_STUDENT_ACTION_SCHEMA_WIKI = """Return ONLY a JSON object with this shape (no prose outside JSON):
{
  "thought": "brief private reasoning",
  "decision": {"category": "need_decomposition|need_retrieval|need_reformulation|sufficient_evidence|synthesize|verify|manage_wiki|finish",
               "parametric_knowledge_used": false},
  "action": {"tool": "decompose|reformulate|search|extract|verify|synthesize|wiki_read|wiki_write|finish", "params": {}},
  "new_facts_extracted": [{"doc_id": "...", "span": "verbatim span", "fact": "grounded fact"}]
}"""

# Auto wiki mode (wiki_mode == "auto"): the wiki is not a tool. It is read into every
# step's prompt automatically, and after every step a dedicated call asks the student to
# rewrite it. No budget steps are consumed and the action grammar stays the baseline one.
_WIKI_AUTO_NOTE = (
    "You keep a personal wiki (wiki.md): a MINIMAL notes file that persists across all "
    "your steps on this question. READ it below before choosing your action -- it is "
    "your memory of everything you have learned so far. After each step you will be "
    "asked to edit it: correct anything it got wrong, add what you just learned, and "
    "keep your best answer up to date."
)

# Fixed three-line format for wiki.md. Small students write junk (raw action JSON,
# rambling paragraphs) into a free-form wiki; a rigid fill-in template keeps the notes
# usable and easy to both write and read back. ANSWER must always name a concrete best
# guess: an earlier "ANSWER: ?" variant taught small models to answer "unknown" even
# when their own FACTS contained the gold answer (observed with qwen3.5:2b).
_WIKI_TEMPLATE = """FACTS:
- <one confirmed fact> (doc_id)
ANSWER: <your current best guess -- always name one, never "?" or "unknown">
NEXT: <the single next thing to do>"""

_WIKI_EXAMPLE = """Example of a good wiki.md:
FACTS:
- Jill Stein ran for president in 2012 and 2016 (q1::doc2)
- She represented the Green Party (q1::doc2)
ANSWER: Green Party
NEXT: verify the party name, then finish"""

_WIKI_INSTRUCTIONS = (
    "You also have a personal wiki (wiki.md): a private notes file that starts empty and "
    "persists across all your steps on this question. Use it as an information bank -- "
    "when you learn something important (key entities, confirmed facts, partial answers, "
    "what to look up next), save it with wiki_write; recall it later with wiki_read. "
    "Keep it MINIMAL (a few short lines, not full documents) -- wiki_write replaces the "
    "whole file. You decide when to read and write; each wiki operation uses a step, so "
    "use it only when it helps you answer."
)

_WIKI_AUTO_USES = (
    "Use the wiki to: remember facts from documents you already searched, keep your "
    "current best answer, and note what to look up next -- so you never repeat a search "
    "or forget a fact you already found."
)

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
    if md.get("wiki_enabled"):
        state["wiki_enabled"] = True
        state["wiki_mode"] = md.get("wiki_mode", "tools")
        if state["wiki_mode"] == "auto":
            # Auto mode: the wiki is read into every step's prompt.
            state["wiki_content"] = str(md.get("wiki", "") or "")
        else:
            state["wiki_chars"] = len(str(md.get("wiki", "") or ""))
            # One-shot content surfacing: present only on the step right after a
            # wiki_read (execute_student_tool clears it when the next action runs).
            if md.get("wiki_just_read") is not None:
                state["wiki_content"] = md.get("wiki_just_read")
    return state


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_student_prompt(
    state: Dict[str, Any], guidance_config: GuidanceConfig, force_finish: bool
) -> str:
    parts: List[str] = []
    wiki_auto = bool(state.get("wiki_enabled")) and state.get("wiki_mode") == "auto"
    wiki_tools = bool(state.get("wiki_enabled")) and not wiki_auto
    parts.append(
        "You are an information-seeking retrieval agent solving a question using tools. "
        "You must ground every fact in retrieved documents and must NOT answer from memory."
    )
    if wiki_tools:
        parts.append(_TOOL_REFERENCE + "\n" + _WIKI_TOOL_REFERENCE_EXTRA)
        parts.append(_WIKI_INSTRUCTIONS)
    else:
        parts.append(_TOOL_REFERENCE)
        if wiki_auto:
            parts.append(_WIKI_AUTO_NOTE)
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

    if wiki_auto:
        parts.append(_WIKI_AUTO_USES)
        parts.append("Your wiki.md:\n" + (state.get("wiki_content") or "(empty)"))
    elif wiki_tools:
        if state.get("wiki_content") is not None:
            content = state.get("wiki_content") or "(empty)"
            parts.append("Your wiki.md (from your wiki_read):\n" + content)
        elif state.get("wiki_chars"):
            parts.append(
                f"Your wiki.md has {state.get('wiki_chars')} characters of saved notes "
                "(use wiki_read to view them)."
            )
        else:
            parts.append("Your wiki.md is currently empty.")

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

    parts.append(_STUDENT_ACTION_SCHEMA_WIKI if wiki_tools else _STUDENT_ACTION_SCHEMA)
    parts.append(_STUDENT_EXAMPLE)
    return "\n\n".join(parts)


def build_wiki_update_prompt(
    state: Dict[str, Any], student_action: Dict[str, Any], tool_observation: Dict[str, Any]
) -> str:
    """Auto-wiki-mode prompt: after each step, ask the student to rewrite wiki.md from
    what it just did/observed. Free text output (no JSON) so nothing structural can fail.

    The prompt pins wiki.md to a rigid three-line template (FACTS / ANSWER / NEXT) with
    a filled example -- small students degrade into raw action-JSON or rambling prose
    when the format is left open (observed with qwen3.5:0.8b)."""
    parts: List[str] = []
    parts.append(
        "You maintain a personal wiki (wiki.md) of MINIMAL notes that helps you solve a "
        "question over multiple retrieval steps. You just completed a step; EDIT the "
        "wiki now so your next step can rely on it."
    )
    parts.append(f"Question: {state.get('question', '')}")
    parts.append("Current wiki.md:\n" + (state.get("wiki_content") or "(empty)"))
    parts.append("Action you just took: " + _json(student_action))
    parts.append("Tool observation: " + _json(tool_observation))
    parts.append(
        "Edit the wiki like a careful reviewer:\n"
        "1. Check every existing line against the observation above -- CORRECT or DELETE "
        "any line that is wrong, outdated, or contradicted by the new evidence.\n"
        "2. ADD any new fact from the observation that helps answer the question "
        "(with its doc_id). Do not duplicate facts already listed.\n"
        "3. UPDATE the ANSWER line to your current best guess. Always commit to one "
        "concrete candidate -- never write \"?\", \"unknown\", or leave it blank; if the "
        "FACTS name a plausible answer, use it.\n"
        "4. UPDATE the NEXT line to the single most useful next step."
    )
    parts.append("Output wiki.md using EXACTLY this format:\n" + _WIKI_TEMPLATE)
    parts.append(_WIKI_EXAMPLE)
    parts.append(
        "Output ONLY the new wiki.md content in that format -- no JSON, no code fences, "
        "no commentary. At most 5 FACTS lines; keep each line short."
    )
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
    # Wiki-enabled episodes: the notes often already contain the answer (its ANSWER
    # line is the student's own running best guess), so surface them here where the
    # final answer is committed.
    if state.get("wiki_content"):
        parts.append("Your wiki.md notes:\n" + state.get("wiki_content"))
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
