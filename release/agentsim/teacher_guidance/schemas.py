"""
Dataclass schemas for the Teacher Guidance pipeline.

These mirror the JSON contracts described in the Teacher Guidance report. We use
dataclasses (not Pydantic) to match the upstream AgentSim style and to keep the
package importable without extra dependencies. Validation is intentionally
lightweight and lives in ``json_utils`` so that parse/validation failures can be
recorded as dataset labels rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# Canonical vocabularies (kept as tuples so they can be imported for validation).
TOOLS = (
    "decompose",
    "reformulate",
    "search",
    "extract",
    "verify",
    "synthesize",
    "finish",
    # Wiki notes tools (only offered to the student when wiki_enabled is set on the
    # run; always accepted here so post-hoc validation of wiki-run data never fails).
    "wiki_read",
    "wiki_write",
)

DECISION_CATEGORIES = (
    "need_decomposition",
    "need_retrieval",
    "need_reformulation",
    "sufficient_evidence",
    "synthesize",
    "verify",
    "finish",
    # Category for wiki_read/wiki_write actions (wiki-enabled runs only).
    "manage_wiki",
)

TEACHER_DECISIONS = ("continue", "accept_finish", "reject_finish", "force_finish")
PLAN_TEACHER_DECISIONS = ("accept_plan", "revise_plan", "reject_plan")

STOP_REASONS = (
    "teacher_accept",
    "budget_forced_finish",
    "invalid_action",
    "error",
)


# ---------------------------------------------------------------------------
# Student action
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    category: str = ""
    parametric_knowledge_used: bool = False


@dataclass
class ToolCall:
    tool: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedFact:
    doc_id: str = ""
    span: str = ""
    fact: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedFact":
        data = data or {}
        return cls(
            doc_id=str(data.get("doc_id", "")),
            span=str(data.get("span", "")),
            fact=str(data.get("fact", "")),
        )


@dataclass
class StudentAction:
    thought: str = ""
    decision: Decision = field(default_factory=Decision)
    action: ToolCall = field(default_factory=ToolCall)
    new_facts_extracted: List[ExtractedFact] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudentAction":
        data = data or {}
        decision_raw = data.get("decision", {}) or {}
        action_raw = data.get("action", {}) or {}
        facts_raw = data.get("new_facts_extracted", []) or []
        return cls(
            thought=str(data.get("thought", "")),
            decision=Decision(
                category=str(decision_raw.get("category", "")),
                parametric_knowledge_used=bool(
                    decision_raw.get("parametric_knowledge_used", False)
                ),
            ),
            action=ToolCall(
                tool=str(action_raw.get("tool", "")),
                params=dict(action_raw.get("params", {}) or {}),
            ),
            new_facts_extracted=[ExtractedFact.from_dict(f) for f in facts_raw],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "decision": {
                "category": self.decision.category,
                "parametric_knowledge_used": self.decision.parametric_knowledge_used,
            },
            "action": {"tool": self.action.tool, "params": self.action.params},
            "new_facts_extracted": [asdict(f) for f in self.new_facts_extracted],
        }


# ---------------------------------------------------------------------------
# Tool observation
# ---------------------------------------------------------------------------
@dataclass
class ToolObservation:
    tool: str = ""
    status: str = "ok"
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "status": self.status, **self.data}


# ---------------------------------------------------------------------------
# Teacher evaluation
# ---------------------------------------------------------------------------
@dataclass
class StudentVisibleGuidance:
    score_binary: int = 0
    score_continuous: float = 0.0
    feedback: Optional[str] = None
    hint: Optional[Dict[str, Any]] = None


@dataclass
class PrivateDiagnosis:
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherEvaluation:
    guidance_level: int = 0
    student_visible: Dict[str, Any] = field(default_factory=dict)
    private_diagnosis: Dict[str, Any] = field(default_factory=dict)
    teacher_decision: str = "continue"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherEvaluation":
        data = data or {}
        return cls(
            guidance_level=int(data.get("guidance_level", 0) or 0),
            student_visible=dict(data.get("student_visible", {}) or {}),
            private_diagnosis=dict(data.get("private_diagnosis", {}) or {}),
            teacher_decision=str(data.get("teacher_decision", "continue")),
        )


# ---------------------------------------------------------------------------
# Plan review
# ---------------------------------------------------------------------------
@dataclass
class PlanStep:
    step_id: int = 0
    goal: str = ""
    intended_tool: str = ""
    rationale: str = ""
    depends_on: List[int] = field(default_factory=list)


@dataclass
class StudentPlan:
    plan_summary: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    stop_condition: str = ""


@dataclass
class TeacherPlanReview:
    plan_review_enabled: bool = True
    review_guidance_level: int = 0
    student_visible: Dict[str, Any] = field(default_factory=dict)
    private_diagnosis: Dict[str, Any] = field(default_factory=dict)
    teacher_decision: str = "revise_plan"


@dataclass
class RevisedStudentPlan:
    revision_summary: str = ""
    plan_summary: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    teacher_feedback_used: List[str] = field(default_factory=list)
    stop_condition: str = ""


@dataclass
class PlanReviewRecord:
    enabled: bool = False
    initial_student_plan_prompt: Optional[str] = None
    initial_student_plan_raw: Optional[str] = None
    initial_plan_calls: List[Dict[str, Any]] = field(default_factory=list)
    initial_plan_call_ms: Optional[float] = None
    initial_student_plan: Optional[Dict[str, Any]] = None
    teacher_plan_review_prompt: Optional[str] = None
    teacher_plan_review_raw: Optional[str] = None
    teacher_plan_review_full: Optional[Dict[str, Any]] = None
    student_visible_plan_feedback: Optional[Dict[str, Any]] = None
    revised_student_plan_prompt: Optional[str] = None
    revised_student_plan_raw: Optional[str] = None
    revised_student_plan: Optional[Dict[str, Any]] = None
    leakage_check: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    plan_review_started_at: Optional[str] = None
    plan_review_ended_at: Optional[str] = None
    plan_review_elapsed_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    t: int = 0
    student_prompt: str = ""
    student_raw: str = ""
    student_calls: List[Dict[str, Any]] = field(default_factory=list)
    student_call_ms: Optional[float] = None
    student_action: Dict[str, Any] = field(default_factory=dict)
    tool_observation: Dict[str, Any] = field(default_factory=dict)
    teacher_prompt: str = ""
    teacher_raw: str = ""
    teacher_calls: List[Dict[str, Any]] = field(default_factory=list)
    teacher_call_ms: Optional[float] = None
    teacher_full: Dict[str, Any] = field(default_factory=dict)
    student_visible_guidance: Dict[str, Any] = field(default_factory=dict)
    leakage_check: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stop_condition: str = "CONTINUE"
    step_started_at: Optional[str] = None
    step_ended_at: Optional[str] = None
    step_elapsed_ms: Optional[float] = None


@dataclass
class EpisodeRecord:
    episode_id: str = ""
    qid: str = ""
    query: str = ""
    gold_answer: str = ""
    # Source provenance -- set from the question row (every converter stamps them).
    dataset: str = "hotpotqa"
    split: str = "validation"
    gold_granularity: str = "sentence"  # "sentence" | "paragraph"
    answer_type: str = "span"           # "span" | "boolean"
    num_hops: Optional[int] = None
    question_type: str = ""
    # Record provenance -- what schema, what code, what configuration produced this.
    schema_version: str = ""
    framework_commit: str = ""
    config_hash: str = ""
    generated_at: str = ""
    budget: int = 5
    guidance_level: int = 0
    student_model: str = ""
    teacher_model: str = ""
    plan_review: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Config blocks (parsed from simulation mode_config)
# ---------------------------------------------------------------------------
@dataclass
class GuidanceConfig:
    level: int = 0
    name: Optional[str] = None
    score_mode: str = "continuous"  # "binary" | "continuous"
    max_feedback_words: int = 60
    expose_next_action_hint: bool = False
    expose_tool_hint: bool = False
    expose_query_hint: bool = False
    expose_doc_title_hint: bool = False
    expose_gold_answer_hint: bool = False
    leak_policy: str = "strict"  # "strict" | "permissive"

    @classmethod
    def from_mode_config(cls, mode_config: Dict[str, Any]) -> "GuidanceConfig":
        block = dict((mode_config or {}).get("guidance", {}) or {})
        return cls(
            level=int(block.get("level", 0) or 0),
            name=block.get("name"),
            score_mode=str(block.get("score_mode", "continuous")),
            max_feedback_words=int(block.get("max_feedback_words", 60) or 60),
            expose_next_action_hint=bool(block.get("expose_next_action_hint", False)),
            expose_tool_hint=bool(block.get("expose_tool_hint", False)),
            expose_query_hint=bool(block.get("expose_query_hint", False)),
            expose_doc_title_hint=bool(block.get("expose_doc_title_hint", False)),
            expose_gold_answer_hint=bool(block.get("expose_gold_answer_hint", False)),
            leak_policy=str(block.get("leak_policy", "strict")),
        )


@dataclass
class PlanReviewConfig:
    enabled: bool = False
    planner: str = "student"  # "student" | "teacher"
    planning_steps: int = 1  # max student review->revise rounds before tool use
    formal_plan: bool = False  # track student adherence to the plan programmatically
    review_guidance_level: Optional[int] = None  # None means reuse guidance.level
    max_initial_plan_steps: int = 6
    max_revised_plan_steps: int = 6
    consume_budget: bool = False
    include_revised_plan_in_student_context: bool = True
    allow_teacher_to_suggest_tools: bool = True
    allow_teacher_to_suggest_queries: bool = False
    allow_teacher_to_reveal_gold_titles: bool = False
    allow_teacher_to_reveal_gold_answer: bool = False

    @classmethod
    def from_mode_config(cls, mode_config: Dict[str, Any]) -> "PlanReviewConfig":
        block = dict((mode_config or {}).get("plan_review", {}) or {})
        rgl = block.get("review_guidance_level", None)
        return cls(
            enabled=bool(block.get("enabled", False)),
            planner=str(block.get("planner", "student")),
            planning_steps=max(1, int(block.get("planning_steps", 1) or 1)),
            formal_plan=bool(block.get("formal_plan", False)),
            review_guidance_level=(int(rgl) if rgl is not None else None),
            max_initial_plan_steps=int(block.get("max_initial_plan_steps", 6) or 6),
            max_revised_plan_steps=int(block.get("max_revised_plan_steps", 6) or 6),
            consume_budget=bool(block.get("consume_budget", False)),
            include_revised_plan_in_student_context=bool(
                block.get("include_revised_plan_in_student_context", True)
            ),
            allow_teacher_to_suggest_tools=bool(
                block.get("allow_teacher_to_suggest_tools", True)
            ),
            allow_teacher_to_suggest_queries=bool(
                block.get("allow_teacher_to_suggest_queries", False)
            ),
            allow_teacher_to_reveal_gold_titles=bool(
                block.get("allow_teacher_to_reveal_gold_titles", False)
            ),
            allow_teacher_to_reveal_gold_answer=bool(
                block.get("allow_teacher_to_reveal_gold_answer", False)
            ),
        )
