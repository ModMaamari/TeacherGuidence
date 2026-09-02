"""Aggregate metrics over an episodes.jsonl file (any arm).

One place computes accuracy, efficiency and token accounting so every arm is summarised
identically. Token accounting reads, per step, ``gen_stats`` (local student loop) or the
``usage`` of ``student_calls`` / ``teacher_calls`` (harness-driven arms), plus the
plan-phase calls when they are recorded.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional


def _usage(call: Dict[str, Any]):
    u = call.get("usage") or call.get("gen_stats") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), float(u.get("cost") or 0.0)


def episode_tokens(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Step-phase and plan-phase token counts + API cost of one episode."""
    s_in = s_out = t_in = t_out = 0
    p_in = p_out = 0
    cost = 0.0
    for s in ep.get("steps") or []:
        if s.get("gen_stats"):
            a, b, _ = _usage(s); s_in += a; s_out += b
        for c in s.get("student_calls") or []:
            a, b, cc = _usage(c); s_in += a; s_out += b; cost += cc
        for c in s.get("teacher_calls") or []:
            a, b, cc = _usage(c); t_in += a; t_out += b; cost += cc
    plan = ep.get("plan") or {}
    if isinstance(plan, dict) and plan.get("gen_stats"):
        a, b, _ = _usage(plan); p_in += a; p_out += b
    pr = ep.get("plan_review") or {}
    if isinstance(pr, dict):
        for key in ("initial_plan_calls", "revision_calls", "review_calls"):
            for c in pr.get(key) or []:
                a, b, cc = _usage(c); p_in += a; p_out += b; cost += cc
        for rnd in pr.get("rounds") or []:
            for key in ("initial_plan_calls", "revision_calls", "review_calls"):
                for c in rnd.get(key) or []:
                    a, b, cc = _usage(c); p_in += a; p_out += b; cost += cc
    return {"student_in": s_in, "student_out": s_out, "teacher_in": t_in, "teacher_out": t_out,
            "plan_in": p_in, "plan_out": p_out, "api_cost_usd": cost}


def invalid_steps(ep: Dict[str, Any]) -> int:
    n = 0
    for s in ep.get("steps") or []:
        m = s.get("metrics") or {}
        if s.get("action_valid") is False or m.get("invalid_action"):
            n += 1
    return n


def aggregate(episodes: List[Dict[str, Any]], judge: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {"n": 0}
    fm = [e.get("final_metrics") or {} for e in episodes]
    toks = [episode_tokens(e) for e in episodes]
    doc = [m["doc_recall"] for m in fm if m.get("doc_recall") is not None]
    stop = {}
    for e in episodes:
        stop[e.get("stop_reason")] = stop.get(e.get("stop_reason"), 0) + 1
    voluntary = sum(1 for e in episodes if e.get("stop_reason") in ("finish", "teacher_accept"))
    out = {
        "n": n,
        "em": round(sum(bool(m.get("exact_match")) for m in fm) / n, 4),
        "f1": round(sum(float(m.get("f1") or 0) for m in fm) / n, 4),
        "cover_match": round(sum(bool(m.get("cover_match", m.get("answer_correct"))) for m in fm) / n, 4),
        "doc_recall": round(statistics.mean(doc), 4) if doc else None,
        "mean_steps": round(statistics.mean(int(e.get("used_steps") or 0) for e in episodes), 3),
        "voluntary_finish": round(voluntary / n, 4),
        "stop_reasons": stop,
        "invalid_action_steps": sum(invalid_steps(e) for e in episodes),
        "total_steps": sum(int(e.get("used_steps") or 0) for e in episodes),
        "student_tokens_per_ep": round(sum(t["student_in"] + t["student_out"] for t in toks) / n, 1),
        "teacher_tokens_per_ep": round(sum(t["teacher_in"] + t["teacher_out"] for t in toks) / n, 1),
        "plan_tokens_per_ep": round(sum(t["plan_in"] + t["plan_out"] for t in toks) / n, 1),
        "total_tokens_per_ep": round(sum(sum(v for k, v in t.items() if k != "api_cost_usd") for t in toks) / n, 1),
        "api_cost_usd": round(sum(t["api_cost_usd"] for t in toks), 6),
        "latency_s_per_ep": round(statistics.mean(float(e.get("elapsed_s") or 0) for e in episodes), 2),
    }
    if judge is not None:
        js = [judge.get(e["qid"]) for e in episodes]
        js = [j for j in js if j is not None]
        out["judge_n"] = len(js)
        out["judge_correct"] = round(sum(js) / len(js), 4) if js else None
    return out
