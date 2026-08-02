"""Tests for the publishable-dataset builder.

Two things must hold for every release: the published records must not carry cost or
provider internals, and the ``train_safe`` view must not carry the teacher's gold-bearing
reasoning. Both are checked here, along with the credential scanner -- including the
false positive that a naive pattern hits on real corpus text.
"""

from __future__ import annotations

import json

from scripts.build_release import (
    scan_for_secrets,
    strip_episode,
)


def _episode() -> dict:
    call = {
        "attempt": 1,
        "model": "fau/gpt-oss-120b",
        "elapsed_ms": 512.0,
        "prompt": "PROMPT",
        "response_text": "RESPONSE",
        "raw_response": {"id": "chatcmpl-1", "system_fingerprint": "vllm-abc"},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                  "cost": 0.00089},
    }
    return {
        "qid": "q1",
        "dataset": "hotpotqa",
        "teacher_router": ["fau/gpt-oss-120b", "custom/openai/gpt-oss-120b"],
        "teacher_models_used": ["fau/gpt-oss-120b"],
        "gold_answer": "Delhi",
        "final_metrics": {"answer_correct": True},
        "steps": [
            {
                "t": 1,
                "student_prompt": "STUDENT PROMPT",
                "student_calls": [dict(call)],
                "teacher_prompt": "TEACHER PROMPT revealing gold Delhi",
                "teacher_raw": '{"private": "the answer is Delhi"}',
                "teacher_private_diagnosis": {"main_error": "none"},
                "student_visible_guidance": {"score": 0.7, "feedback": "Good."},
                "teacher_calls": [dict(call)],
                "wiki_update_call": dict(call),
            }
        ],
        "plan_review": {
            "enabled": True,
            "initial_plan_calls": [dict(call)],
            "teacher_plan_review_prompt": "PLAN PROMPT with gold Delhi",
            "teacher_plan_review_raw": "{...}",
            "teacher_plan_review_full": {"private_diagnosis": {}},
            "student_visible_plan_feedback": {"feedback": "Tighten step 2."},
            "rounds": [
                {
                    "round": 1,
                    "review_calls": [dict(call)],
                    "revision_calls": [dict(call)],
                    "teacher_plan_review_raw": "{...}",
                    "student_visible_plan_feedback": {"feedback": "ok"},
                }
            ],
        },
    }


def _all_calls(ep: dict):
    for step in ep.get("steps") or []:
        yield from step.get("student_calls") or []
        yield from step.get("teacher_calls") or []
        if isinstance(step.get("wiki_update_call"), dict):
            yield step["wiki_update_call"]
    plan = ep.get("plan_review") or {}
    yield from plan.get("initial_plan_calls") or []
    for rnd in plan.get("rounds") or []:
        yield from rnd.get("review_calls") or []
        yield from rnd.get("revision_calls") or []


def test_cost_and_provider_internals_are_removed_everywhere():
    published = strip_episode(_episode(), train_safe=False)
    calls = list(_all_calls(published))
    assert calls, "fixture should contain calls"
    for call in calls:
        assert "raw_response" not in call
        assert "cost" not in call["usage"]
        # token telemetry is research-relevant and must survive
        assert call["usage"]["prompt_tokens"] == 10
        assert call["response_text"] == "RESPONSE"
    assert "teacher_router" not in published
    assert published["teacher_models_used"] == ["fau/gpt-oss-120b"]


def test_full_view_keeps_privileged_teacher_reasoning():
    published = strip_episode(_episode(), train_safe=False)
    step = published["steps"][0]
    assert step["teacher_prompt"]
    assert step["teacher_private_diagnosis"]
    assert published["plan_review"]["teacher_plan_review_raw"]


def test_train_safe_view_removes_every_gold_bearing_teacher_field():
    published = strip_episode(_episode(), train_safe=True)
    step = published["steps"][0]
    for key in ("teacher_prompt", "teacher_raw", "teacher_private_diagnosis"):
        assert key not in step
    plan = published["plan_review"]
    for key in ("teacher_plan_review_prompt", "teacher_plan_review_raw",
                "teacher_plan_review_full"):
        assert key not in plan
        assert key not in plan["rounds"][0]
    # the sanitized, student-visible guidance is the whole point -- it must remain
    assert step["student_visible_guidance"]["feedback"] == "Good."
    assert plan["student_visible_plan_feedback"]
    assert published["gold_answer"] == "Delhi"  # the label, not a leak


def test_keep_raw_response_opt_in():
    published = strip_episode(_episode(), train_safe=False, keep_raw_response=True)
    assert all("raw_response" in c for c in _all_calls(published))
    # cost is still removed even when raw bodies are kept
    assert all("cost" not in c["usage"] for c in _all_calls(published))


def test_stripping_does_not_mutate_the_input():
    original = _episode()
    strip_episode(original, train_safe=True)
    assert "teacher_router" in original
    assert "raw_response" in original["steps"][0]["teacher_calls"][0]


# --- credential scanning ------------------------------------------------------------
def test_scanner_detects_real_key_shapes():
    assert scan_for_secrets('{"k": "sk-4H4B6ybMUK5H2HqlWgYQpA"}')
    assert scan_for_secrets('{"k": "sk-or-v1-e870aba71039861369cf02bbceb9646daed6505161"}')
    assert scan_for_secrets('{"h": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghij"}')


def test_scanner_detects_configured_literals():
    blob = '{"text": "some content with MY-SECRET-VALUE-123 inside"}'
    assert scan_for_secrets(blob, ["MY-SECRET-VALUE-123"])
    assert not scan_for_secrets(blob, ["A-DIFFERENT-SECRET"])


def test_scanner_does_not_fire_on_corpus_text():
    """Regression: an earlier pattern matched 'Yakutsk-Kirensk-Krasnoyarsk' -- real
    HotpotQA text about a Siberian air route -- and aborted a 6000-episode release."""
    corpus_text = (
        '{"text": "This airfield was part of the Yakutsk-Kirensk-Krasnoyarsk leg of the '
        'ALSIB route. Nearby towns include Ust-Kut and Novosibirsk-Tolmachevo."}'
    )
    assert scan_for_secrets(corpus_text) == []


def test_scanner_ignores_hyphenated_words_without_digits():
    assert scan_for_secrets('{"t": "Minsk-Smolensk-Bryansk-Kursk-Belgorod-Kharkiv"}') == []


def test_published_record_is_json_serializable():
    published = strip_episode(_episode(), train_safe=True)
    assert json.loads(json.dumps(published, ensure_ascii=False))["qid"] == "q1"
