"""
Pydantic models for the student/teacher JSON contracts.

These serve two purposes:

1. ``.model_json_schema()`` is fed to Ollama's structured-output ``format`` field
   (see ``LLMClient.get_completion(response_schema=...)``), which grammar-constrains
   generation so the model is physically unable to emit invalid JSON or an
   out-of-vocabulary ``tool``/``category``/``teacher_decision`` value in the first
   place -- prevention instead of post-hoc retries.
2. ``.model_validate(obj)`` gives precise, structured validation errors, used by
   ``json_utils.validate_*`` to build clearer repair-prompt messages than the previous
   hand-rolled checks.

The ``TOOLS``/``DECISION_CATEGORIES``/``TEACHER_DECISIONS``/``PLAN_TEACHER_DECISIONS``
vocabularies are imported from ``schemas.py`` (single source of truth) rather than
duplicated here.

Generation vs. validation strictness: making every free-text field optional (to stay
lenient for post-hoc validation of already-produced JSON, including older data) turned
out to actively hurt quality once fed to Ollama as a *generation* schema -- grammar-
constrained decoding takes the shortest grammatically-valid path once all *required*
fields are satisfied, so an optional "thought" field gets skipped entirely and the
model jumps straight to a bare, under-reasoned action (observed in production: empty
`thought`, empty `action.params`, and a degenerate first `decompose` call with no
`sub_questions` -- a no-op that burns a step for nothing). The ``*GenerationModel``
subclasses below tighten exactly the free-text fields that carry the model's reasoning
(non-empty, with a minimum length) and are used *only* to build the schema handed to
Ollama for student-routed calls -- the base models stay lenient and are what
``validate_*`` continues to check post-hoc, so existing/older data and OpenRouter-
routed teacher calls (whose provider never receives the schema payload, only a loose
"valid JSON" flag -- see ``LLMClient._custom_completion``) are unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentsim.teacher_guidance.schemas import (
    TOOLS,
    DECISION_CATEGORIES,
    TEACHER_DECISIONS,
    PLAN_TEACHER_DECISIONS,
)

Tool = Literal[TOOLS]  # type: ignore[valid-type]
DecisionCategory = Literal[DECISION_CATEGORIES]  # type: ignore[valid-type]
TeacherDecision = Literal[TEACHER_DECISIONS]  # type: ignore[valid-type]
PlanTeacherDecision = Literal[PLAN_TEACHER_DECISIONS]  # type: ignore[valid-type]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Student action (per tool-use step)
# ---------------------------------------------------------------------------
class DecisionModel(_Strict):
    category: Optional[DecisionCategory] = None
    parametric_knowledge_used: bool = False


class ToolCallModel(_Strict):
    tool: Tool
    params: Dict[str, Any] = {}


class ExtractedFactModel(_Strict):
    doc_id: str = ""
    span: str = ""
    fact: str = ""


class StudentActionModel(_Strict):
    thought: str = ""
    decision: DecisionModel = DecisionModel()
    action: ToolCallModel
    new_facts_extracted: List[ExtractedFactModel] = []

    @model_validator(mode="after")
    def _finish_requires_answer(self) -> "StudentActionModel":
        if self.action.tool == "finish":
            answer = self.action.params.get("answer")
            if not (isinstance(answer, str) and answer.strip()):
                raise ValueError("finish_missing_answer")
        return self


class StudentActionGenerationModel(StudentActionModel):
    """Stricter variant used only for Ollama's response_schema: thought must be a
    real, substantive reasoning trace, not skippable."""

    thought: str = Field(..., min_length=15)


# ---------------------------------------------------------------------------
# Student plan (preflight plan-review phase)
# ---------------------------------------------------------------------------
class PlanStepModel(_Strict):
    step_id: int = 0
    goal: str = ""
    intended_tool: Tool
    rationale: str = ""
    depends_on: List[int] = []


class PlanStepGenerationModel(PlanStepModel):
    goal: str = Field(..., min_length=5)
    rationale: str = Field(..., min_length=5)


class StudentPlanModel(_Strict):
    plan_summary: str = ""
    steps: List[PlanStepModel] = []
    uncertainties: List[str] = []
    stop_condition: str = ""


class StudentPlanGenerationModel(StudentPlanModel):
    plan_summary: str = Field(..., min_length=10)
    steps: List[PlanStepGenerationModel] = Field(..., min_length=1)
    stop_condition: str = Field(..., min_length=3)


class RevisedStudentPlanModel(_Strict):
    revision_summary: str = ""
    plan_summary: str = ""
    steps: List[PlanStepModel] = []
    teacher_feedback_used: List[str] = []
    stop_condition: str = ""


class RevisedStudentPlanGenerationModel(RevisedStudentPlanModel):
    revision_summary: str = Field(..., min_length=5)
    plan_summary: str = Field(..., min_length=10)
    steps: List[PlanStepGenerationModel] = Field(..., min_length=1)
    stop_condition: str = Field(..., min_length=3)


# ---------------------------------------------------------------------------
# Teacher evaluation (per tool-use step)
# ---------------------------------------------------------------------------
class TeacherEvaluationModel(_Strict):
    guidance_level: int = 0
    student_visible: Dict[str, Any] = {}
    private_diagnosis: Dict[str, Any] = {}
    teacher_decision: TeacherDecision = "continue"


# ---------------------------------------------------------------------------
# Teacher plan review (preflight plan-review phase)
# ---------------------------------------------------------------------------
class TeacherPlanReviewModel(_Strict):
    plan_review_enabled: bool = True
    review_guidance_level: int = 0
    student_visible: Dict[str, Any] = {}
    private_diagnosis: Dict[str, Any] = {}
    teacher_decision: PlanTeacherDecision = "revise_plan"
