"""Deterministic offline stand-in for any model (``mock/<anything>``).

Used by the smoke tests so the complete pipeline -- collection, the four evaluation
arms, the judge -- can run without a GPU or an API key. It recognises the harness's
prompt kinds by their opening sentence and returns a syntactically valid answer:

* student action prompt  -> a ``search`` for the question, then ``finish`` with a
                            canned answer (so an episode takes two steps)
* student plan prompt    -> a two-step plan; revised-plan prompt -> the same, revised
* teacher step prompt    -> ``continue`` with score 0.7 (``accept_finish`` on a finish)
* teacher plan review    -> ``accept_plan``
* judge prompt           -> ``{"correct": 1}`` when the model answer contains the gold

Nothing here is meant to be accurate; it exercises code paths and file formats.
"""
from __future__ import annotations

import json
import re


def _question(prompt: str) -> str:
    m = re.search(r"^Question: (.*)$", prompt, re.M)
    return (m.group(1) if m else "the question").strip()


def mock_completion(prompt: str) -> str:
    p = prompt.lstrip()
    q = _question(prompt)
    if p.startswith("You are grading one answer"):
        gold = re.search(r"^Gold answer: (.*)$", prompt, re.M)
        ans = re.search(r"^Model answer: (.*)$", prompt, re.M)
        ok = bool(gold and ans and gold.group(1).strip().lower() in ans.group(1).lower())
        return json.dumps({"correct": 1 if ok else 0, "reason": "mock judge"})
    if p.startswith("You are a teacher reviewing the student's initial plan"):
        return json.dumps({"plan_review_enabled": True, "review_guidance_level": 3,
                           "student_visible": {"score_binary": 1, "score_continuous": 0.8,
                                               "feedback": "The plan is reasonable; search first, then finish."},
                           "private_diagnosis": {"note": "mock"}, "teacher_decision": "accept_plan"})
    if p.startswith("You are a teacher evaluating one step"):
        finishing = '"tool": "finish"' in prompt
        return json.dumps({"guidance_level": 3,
                           "student_visible": {"score_binary": 1, "score_continuous": 0.7,
                                               "feedback": "Reasonable step. Keep grounding every claim in a retrieved document."},
                           "private_diagnosis": {"note": "mock"},
                           "teacher_decision": "accept_finish" if finishing else "continue"})
    if p.startswith("You are a retrieval agent. Before using any tools") or p.startswith("Revise your plan"):
        plan = {"plan_summary": f"Search the local documents for '{q[:60]}' and answer from them.",
                "steps": [{"step_id": 1, "goal": "Find the documents that mention the entities in the question",
                           "intended_tool": "search", "rationale": "Evidence must come from retrieved text", "depends_on": []},
                          {"step_id": 2, "goal": "Answer from the retrieved evidence",
                           "intended_tool": "finish", "rationale": "Commit once the evidence is in hand", "depends_on": [1]}],
                "uncertainties": ["entity names may be ambiguous"], "stop_condition": "an answer is supported by a retrieved document"}
        if p.startswith("Revise"):
            plan = {"revision_summary": "Kept the plan; the teacher accepted it.", "teacher_feedback_used": ["plan accepted"], **plan}
        return json.dumps(plan)
    # student action: search on the first step (no previous actions), finish afterwards
    first_step = re.search(r"Previous actions: \[\]", prompt) is not None
    if first_step:
        return json.dumps({"thought": "I have no evidence yet, so I search the local documents for the question entities.",
                           "decision": {"category": "need_retrieval", "parametric_knowledge_used": False},
                           "action": {"tool": "search", "params": {"query": q[:200], "k": 5}},
                           "new_facts_extracted": []})
    return json.dumps({"thought": "I have retrieved documents; I commit to an answer supported by them.",
                       "decision": {"category": "finish", "parametric_knowledge_used": False},
                       "action": {"tool": "finish", "params": {"answer": "mock answer", "citations": []}},
                       "new_facts_extracted": []})
