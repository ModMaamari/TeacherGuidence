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
``validate_*`` continues to check post-hoc, so existing/older data and provider-
routed teacher calls (whose provider never receives the schema payload, only a loose
"valid JSON" flag -- see ``LLMClient._custom_completion``) are unaffected.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

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
            params = self.action.params
            answer = params.get("answer") if isinstance(params, dict) else getattr(params, "answer", None)
            if not (isinstance(answer, str) and answer.strip()):
                raise ValueError("finish_missing_answer")
        return self


# ---------------------------------------------------------------------------
# Per-tool params, generation-only: mirrors what tool_executor.py actually reads
# per tool (see execute_student_tool). A freeform params dict let the model get away
# with an empty {} for tools that need real content -- e.g. a "decompose" with no
# sub_questions or a "search" with no query is a silent no-op that burns a step for
# nothing. A discriminated union on `tool` requires the right keys for the right
# tool. This is used only by StudentActionGenerationModel; the lenient base
# ToolCallModel above (freeform params) is what validate_student_action still checks
# post-hoc, so it accepts any already-produced JSON regardless of tool.
# ---------------------------------------------------------------------------
class DecomposeParams(_Strict):
    sub_questions: List[str] = Field(..., min_length=1)


class ReformulateParams(_Strict):
    queries: List[str] = Field(..., min_length=1)
    reformulation_type: Optional[str] = None


class SearchParams(_Strict):
    query: str = Field(..., min_length=1)
    k: int = 5


class ExtractParams(_Strict):
    doc_ids: List[str] = Field(..., min_length=1)
    target_facts: List[str] = Field(..., min_length=1)


class VerifyParams(_Strict):
    claim: str = Field(..., min_length=1)
    query: Optional[str] = None
    k: int = 5


class SynthesizeParams(_Strict):
    pass


class FinishParams(_Strict):
    answer: str = Field(..., min_length=1)
    citations: List[Dict[str, Any]] = []


class DecomposeCall(_Strict):
    tool: Literal["decompose"]
    params: DecomposeParams


class ReformulateCall(_Strict):
    tool: Literal["reformulate"]
    params: ReformulateParams


class SearchCall(_Strict):
    tool: Literal["search"]
    params: SearchParams


class ExtractCall(_Strict):
    tool: Literal["extract"]
    params: ExtractParams


class VerifyCall(_Strict):
    tool: Literal["verify"]
    params: VerifyParams


class SynthesizeCall(_Strict):
    tool: Literal["synthesize"]
    params: SynthesizeParams = SynthesizeParams()


class FinishCall(_Strict):
    tool: Literal["finish"]
    params: FinishParams


class WikiReadParams(_Strict):
    pass


class WikiWriteParams(_Strict):
    content: str = Field(..., min_length=1)


class WikiReadCall(_Strict):
    tool: Literal["wiki_read"]
    params: WikiReadParams = WikiReadParams()


class WikiWriteCall(_Strict):
    tool: Literal["wiki_write"]
    params: WikiWriteParams


ToolCallUnion = Annotated[
    Union[
        DecomposeCall, ReformulateCall, SearchCall, ExtractCall,
        VerifyCall, SynthesizeCall, FinishCall,
    ],
    Field(discriminator="tool"),
]

# Wiki-enabled runs get the same union extended with the two wiki tools, so the
# baseline (wiki-disabled) generation grammar stays byte-identical to before.
WikiToolCallUnion = Annotated[
    Union[
        DecomposeCall, ReformulateCall, SearchCall, ExtractCall,
        VerifyCall, SynthesizeCall, FinishCall, WikiReadCall, WikiWriteCall,
    ],
    Field(discriminator="tool"),
]


class StudentActionGenerationModel(StudentActionModel):
    """Stricter variant used only for Ollama's response_schema: thought must be a
    real, substantive reasoning trace (not skippable), and action.params must match
    the shape the chosen tool actually needs (not an empty no-op {})."""

    thought: str = Field(..., min_length=15)
    action: ToolCallUnion


class StudentActionWikiGenerationModel(StudentActionGenerationModel):
    """Generation schema for wiki-enabled runs: identical to the standard one plus
    the wiki_read/wiki_write tools in the action grammar."""

    action: WikiToolCallUnion


class StudentFinishActionGenerationModel(_Strict):
    """Finish-only variant used as Ollama's response_schema on the *final* (force-finish)
    step. Grammar-constraining generation to a FinishCall (tool == "finish", non-empty
    ``params.answer``) makes the model turn its retrieved context into an actual answer
    instead of burning the last step on yet another search -- which previously left the
    system fabricating a "Budget exhausted" finish with answer "unknown".

    ``action`` is declared *before* ``thought`` on purpose: Ollama emits fields in schema
    order, so a small model that rambles in ``thought`` would truncate the answer away
    before reaching it (observed on qwen3.5:2b). Emitting the answer first makes the
    forced finish robust to an over-long reasoning trace; ``thought`` still follows and is
    parsed leniently by ``parse_student_action`` regardless of field order. FinishParams
    already enforces a non-empty ``answer``, so no separate validator is needed here."""

    action: FinishCall
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
