"""
JSON parsing and validation helpers for Teacher Guidance.

Model outputs are expected to be a single JSON object, but models often wrap the
object in prose or ```` ```json ```` fences. We extract the first balanced JSON
object and parse it. We deliberately do *not* aggressively repair malformed JSON:
parse and validation outcomes are recorded as ``(obj, info)`` so the pipeline can use
them as dataset labels (``json_valid``, ``errors``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from agentsim.teacher_guidance.schemas import (
    StudentAction,
    TeacherEvaluation,
    TOOLS,
    DECISION_CATEGORIES,
    TEACHER_DECISIONS,
    PLAN_TEACHER_DECISIONS,
)


def extract_first_json_object(raw: str) -> Optional[str]:
    """Return the substring of the first top-level ``{...}`` JSON object, or None.

    Tolerates code fences and surrounding prose. Respects strings/escapes so that
    braces inside string literals do not break brace matching.
    """
    if not raw:
        return None

    text = raw.strip()
    # Strip a leading ```json / ``` fence if present.
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_object(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse the first JSON object from ``raw``.

    Returns ``(obj, info)`` where ``info`` has ``json_valid`` and ``errors``.
    On failure ``obj`` is ``{}``.
    """
    info: Dict[str, Any] = {"json_valid": False, "errors": []}

    candidate = extract_first_json_object(raw)
    if candidate is None:
        info["errors"].append("no_json_object_found")
        return {}, info

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        info["errors"].append(f"json_decode_error: {exc}")
        return {}, info

    if not isinstance(obj, dict):
        info["errors"].append("json_not_object")
        return {}, info

    info["json_valid"] = True
    return obj, info


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_student_action(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["not_an_object"]

    action = obj.get("action")
    if not isinstance(action, dict):
        errors.append("missing_action")
    else:
        tool = action.get("tool")
        if tool not in TOOLS:
            errors.append(f"invalid_tool:{tool}")
        if "params" in action and not isinstance(action.get("params"), dict):
            errors.append("params_not_object")

    decision = obj.get("decision")
    if isinstance(decision, dict):
        category = decision.get("category")
        if category is not None and category not in DECISION_CATEGORIES:
            errors.append(f"invalid_category:{category}")
    # decision is optional-but-recommended; absence is not fatal.

    facts = obj.get("new_facts_extracted", [])
    if facts is not None and not isinstance(facts, list):
        errors.append("new_facts_extracted_not_list")

    return len(errors) == 0, errors


def validate_teacher_evaluation(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["not_an_object"]

    if not isinstance(obj.get("student_visible", {}), dict):
        errors.append("student_visible_not_object")
    if not isinstance(obj.get("private_diagnosis", {}), dict):
        errors.append("private_diagnosis_not_object")

    decision = obj.get("teacher_decision")
    if decision is not None and decision not in TEACHER_DECISIONS:
        errors.append(f"invalid_teacher_decision:{decision}")

    return len(errors) == 0, errors


def validate_teacher_plan_review(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    if not isinstance(obj.get("student_visible", {}), dict):
        errors.append("student_visible_not_object")
    decision = obj.get("teacher_decision")
    if decision is not None and decision not in PLAN_TEACHER_DECISIONS:
        errors.append(f"invalid_plan_decision:{decision}")
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Typed parse helpers (obj, info)
# ---------------------------------------------------------------------------
def parse_student_action(raw: str) -> Tuple[StudentAction, Dict[str, Any]]:
    obj, info = parse_json_object(raw)
    valid, errors = validate_student_action(obj) if info["json_valid"] else (False, info["errors"])
    info["action_valid"] = valid
    info["errors"] = list(info.get("errors", [])) + [e for e in errors if e not in info.get("errors", [])]
    return StudentAction.from_dict(obj), info


def parse_teacher_evaluation(raw: str) -> Tuple[TeacherEvaluation, Dict[str, Any]]:
    obj, info = parse_json_object(raw)
    valid, errors = validate_teacher_evaluation(obj) if info["json_valid"] else (False, info["errors"])
    info["eval_valid"] = valid
    info["errors"] = list(info.get("errors", [])) + [e for e in errors if e not in info.get("errors", [])]
    return TeacherEvaluation.from_dict(obj), info


def parse_student_plan(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj, info = parse_json_object(raw)
    return obj, info


def parse_teacher_plan_review(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj, info = parse_json_object(raw)
    if info["json_valid"]:
        valid, errors = validate_teacher_plan_review(obj)
        info["review_valid"] = valid
        info["errors"] = list(info.get("errors", [])) + [e for e in errors if e not in info.get("errors", [])]
    return obj, info


def parse_revised_plan(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj, info = parse_json_object(raw)
    return obj, info
