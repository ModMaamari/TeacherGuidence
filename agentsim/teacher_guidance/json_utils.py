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
from pydantic import ValidationError

from agentsim.teacher_guidance.schemas import StudentAction, TeacherEvaluation
from agentsim.teacher_guidance.pydantic_schemas import (
    StudentActionModel,
    TeacherEvaluationModel,
    TeacherPlanReviewModel,
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
# Validation (delegates to the Pydantic models in pydantic_schemas.py, translating
# ValidationError.errors() into the same short error-code strings callers already
# depend on, e.g. "invalid_tool:<value>", "finish_missing_answer".)
# ---------------------------------------------------------------------------
def _pydantic_errors_to_strings(exc: ValidationError, code_map: Dict[str, str]) -> List[str]:
    """Convert pydantic errors into short, greppable error-code strings.

    ``code_map`` maps a dotted field path (e.g. ``"action.tool"``) to a code prefix
    (e.g. ``"invalid_tool"``); the offending input value is appended. A custom
    ``model_validator`` that raises ``ValueError("some_code")`` (e.g.
    ``finish_missing_answer``) is passed through as-is -- the message *is* the code.
    Anything unmapped still gets a readable, if generic, code.
    """
    out: List[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        if err["type"] == "value_error":
            out.append(str(err["ctx"]["error"]))
            continue
        if err["type"] == "missing":
            out.append(f"missing_{loc.replace('.', '_')}")
            continue
        prefix = code_map.get(loc, f"invalid_{loc.replace('.', '_')}")
        out.append(f"{prefix}:{err.get('input')}")
    return out


def validate_student_action(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    try:
        StudentActionModel.model_validate(obj)
        return True, []
    except ValidationError as exc:
        errors = _pydantic_errors_to_strings(exc, {
            "action": "missing_action",
            "action.tool": "invalid_tool",
            "action.params": "params_not_object",
            "decision.category": "invalid_category",
            "new_facts_extracted": "new_facts_extracted_not_list",
        })
        return False, errors


def validate_teacher_evaluation(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    try:
        TeacherEvaluationModel.model_validate(obj)
        return True, []
    except ValidationError as exc:
        errors = _pydantic_errors_to_strings(exc, {
            "student_visible": "student_visible_not_object",
            "private_diagnosis": "private_diagnosis_not_object",
            "teacher_decision": "invalid_teacher_decision",
        })
        return False, errors


def validate_teacher_plan_review(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(obj, dict):
        return False, ["not_an_object"]
    try:
        TeacherPlanReviewModel.model_validate(obj)
        return True, []
    except ValidationError as exc:
        errors = _pydantic_errors_to_strings(exc, {
            "student_visible": "student_visible_not_object",
            "teacher_decision": "invalid_plan_decision",
        })
        return False, errors


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
