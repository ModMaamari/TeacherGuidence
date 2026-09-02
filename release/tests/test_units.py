"""Unit tests (CPU, seconds): python -m pytest tests/ -q"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tgd  # noqa: F401,E402
from tgd.splits import is_dev, pool_of  # noqa: E402
from tgd.metrics import aggregate, episode_tokens  # noqa: E402


def test_pool_is_deterministic_and_roughly_ten_percent():
    qids = [f"q{i}" for i in range(20000)]
    a = [pool_of(q) for q in qids]
    b = [pool_of(q) for q in qids]
    assert a == b
    frac = sum(x == "heldout_test" for x in a) / len(a)
    assert 0.09 < frac < 0.11
    assert pool_of("abc", salt="other") in ("heldout_test", "trainable")
    assert sum(is_dev(q) for q in qids) / len(qids) < 0.05


def test_provider_resolution(monkeypatch):
    from agentsim.config import config
    assert config.get_provider_from_model_id("oai/x") == "oai"
    assert config.get_provider_from_model_id("oai-judge/org/model") == "oai"
    assert config.get_provider_from_model_id("mock/anything") == "mock"
    assert config.get_provider_from_model_id("vllm/student") == "vllm"
    monkeypatch.setenv("OAI_JUDGE_BASE_URL", "http://example/v1")
    base, key, name = config.oai_endpoint("oai-judge/org/model")
    assert (base, key, name) == ("http://example/v1", None, "org/model")
    assert config.provider_available("oai-judge/org/model")
    monkeypatch.delenv("OAI_BASE_URL", raising=False)
    assert not config.provider_available("oai/unconfigured")


def test_mock_outputs_validate_against_schemas():
    from agentsim.clients.mock_provider import mock_completion
    from agentsim.teacher_guidance.json_utils import parse_student_action, parse_teacher_evaluation
    first = mock_completion("You are an information-seeking retrieval agent ...\nQuestion: Who?\nPrevious actions: []\n")
    action, meta = parse_student_action(first)
    assert action.action.tool == "search"
    later = mock_completion("You are an information-seeking retrieval agent ...\nQuestion: Who?\nPrevious actions: [{\"tool\": \"search\"}]\n")
    action, meta = parse_student_action(later)
    assert action.action.tool == "finish"
    ev, _ = parse_teacher_evaluation(mock_completion("You are a teacher evaluating one step of ...\nQuestion: Who?\n"))
    assert ev.teacher_decision == "continue"
    assert json.loads(mock_completion("You are grading one answer\nQuestion: q\nGold answer: Paris\nModel answer: paris, france\n"))["correct"] == 1


def test_judge_parse():
    from scripts.judge import parse_verdict
    assert parse_verdict('text {"correct": 1, "reason": "ok"} trailing')["correct"] == 1
    assert parse_verdict('{"correct": "0"}')["correct"] == 0
    assert parse_verdict("no json") is None
    assert parse_verdict('{"score": 1}') is None


def test_metrics_aggregate_and_tokens():
    ep = {"qid": "a", "final_metrics": {"exact_match": True, "f1": 1.0, "cover_match": True, "doc_recall": 0.5},
          "used_steps": 2, "stop_reason": "finish", "elapsed_s": 1.0,
          "plan": {"gen_stats": {"prompt_tokens": 10, "completion_tokens": 5}},
          "steps": [{"gen_stats": {"prompt_tokens": 100, "completion_tokens": 20}, "action_valid": True},
                    {"student_calls": [{"usage": {"prompt_tokens": 50, "completion_tokens": 5, "cost": 0.001}}],
                     "teacher_calls": [{"usage": {"prompt_tokens": 70, "completion_tokens": 30}}], "action_valid": False}]}
    t = episode_tokens(ep)
    assert (t["student_in"], t["student_out"], t["teacher_in"], t["teacher_out"], t["plan_in"]) == (150, 25, 70, 30, 10)
    agg = aggregate([ep], judge={"a": 1})
    assert agg["em"] == 1.0 and agg["invalid_action_steps"] == 1 and agg["judge_correct"] == 1.0
    assert agg["total_tokens_per_ep"] == 290.0 and abs(agg["api_cost_usd"] - 0.001) < 1e-9


def test_hybrid_client_routes_local_student():
    from tgd.guided_loop import LOCAL_STUDENT, HybridLLMClient
    from tgd.mock_policy import MockPolicy

    class FakeTeacher:
        async def get_completion(self, prompt, model=None, **kw):
            return f"teacher:{model}"

        async def get_completion_with_fallback(self, models, **kw):
            return f"teacher:{models[0]}", models[0]

    c = HybridLLMClient(MockPolicy(), FakeTeacher(), serialize_gpu=False)
    out = asyncio.run(c.get_completion("Question: x\nPrevious actions: []\n", model=LOCAL_STUDENT, return_raw=True))
    assert "search" in out["text"] and out["usage"]["prompt_tokens"] > 0
    assert asyncio.run(c.get_completion("p", model="mock/t")) == "teacher:mock/t"
