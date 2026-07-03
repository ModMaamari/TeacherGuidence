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
import re
from typing import Any, Dict, List, Optional, Tuple

import json_repair

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


def _repair_json(text: str) -> str:
    """Apply safe, common repairs for small-model JSON glitches.

    Conservative fixes that do not change well-formed JSON: normalise curly quotes to
    straight quotes, remove trailing commas before ``}``/``]``, and drop backslashes
    before characters that are not valid JSON escapes (e.g. ``\\'`` -- small models
    routinely escape apostrophes out of Python/JS habit, but JSON only allows
    ``\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX``, so this is fatal to json.loads until
    fixed). Safe to apply globally: valid JSON never has a bare backslash outside of a
    string escape sequence in the first place.
    """
    repaired = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', "", repaired)
    return repaired


def parse_json_object(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse the first JSON object from ``raw``.

    Returns ``(obj, info)`` where ``info`` has ``json_valid``, ``errors``, and
    ``repaired`` (True if a repair pass was needed). On failure ``obj`` is ``{}``.

    Three tiers, each only attempted if the previous one failed: (1) strict
    ``json.loads``, (2) the conservative regex repairs above, (3) the ``json_repair``
    library (handles the wider range of LLM JSON glitches: unescaped quotes, missing
    brackets, stray commentary, etc.) as a last resort before giving up.
    """
    info: Dict[str, Any] = {"json_valid": False, "errors": [], "repaired": False}

    candidate = extract_first_json_object(raw)
    if candidate is None:
        # No balanced {...} found -- typically a response truncated mid-object (e.g. a
        # reasoning model running out of tokens). extract_first_json_object requires a
        # closing brace, but json_repair can often complete an unbalanced structure
        # directly, so give it a shot before giving up entirely.
        start = raw.find("{") if raw else -1
        if start != -1:
            try:
                repaired_obj = json_repair.loads(raw[start:])
            except Exception:
                repaired_obj = None
            if isinstance(repaired_obj, dict) and repaired_obj:
                info["json_valid"] = True
                info["repaired"] = True
                return repaired_obj, info
        info["errors"].append("no_json_object_found")
        return {}, info

    obj = None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            obj = json.loads(_repair_json(candidate))
            info["repaired"] = True
        except json.JSONDecodeError:
            try:
                obj = json_repair.loads(candidate)
                info["repaired"] = True
            except Exception as exc:  # pragma: no cover - json_repair rarely raises
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
        params = action.get("params")
        if "params" in action and not isinstance(params, dict):
            errors.append("params_not_object")
        # A finish action must carry a non-empty answer.
        if tool == "finish":
            answer = (params or {}).get("answer") if isinstance(params, dict) else None
            if not (isinstance(answer, str) and answer.strip()):
                errors.append("finish_missing_answer")

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
    valid, errors = validate_teacher_plan_review(obj) if info["json_valid"] else (False, info["errors"])
    info["review_valid"] = valid
    info["errors"] = list(info.get("errors", [])) + [e for e in errors if e not in info.get("errors", [])]
    return obj, info


def parse_revised_plan(raw: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj, info = parse_json_object(raw)
    return obj, info
