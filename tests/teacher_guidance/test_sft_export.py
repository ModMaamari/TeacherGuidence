"""Tests for chat-format SFT example construction."""

import json

from agentsim.teacher_guidance.sft_export import build_examples_from_episode, build_dataset


def _episode():
    return {
        "qid": "q1",
        "student_model": "ollama/qwen3.5:4b",
        "plan_review": {
            "enabled": True,
            "initial_student_plan_prompt": "Write a plan.\n\nQuestion: who?",
            "initial_student_plan_raw": '{"plan_summary": "search then finish"}',
        },
        "steps": [
            {
                "t": 1,
                "student_prompt": "Question: who?\n\nStep 1 of 5.\n\nReturn ONLY JSON.",
                "student_action": {"thought": "I will search.",
                                    "action": {"tool": "search", "params": {"query": "who"}}},
                "student_visible_guidance": {"score": 0.9, "feedback": "You found the right doc."},
            },
            {
                "t": 2,
                "student_prompt": ("Question: who?\n\nStep 2 of 5.\n\n"
                                   "Previous teacher guidance: {\"feedback\": \"You found the right doc.\"}\n\n"
                                   "Return ONLY JSON."),
                "student_action": {"thought": "Now I extract.",
                                   "action": {"tool": "extract", "params": {"doc_ids": ["d1"], "target_facts": ["f"]}}},
                "student_visible_guidance": {"score": 1.0, "feedback": "done"},
            },
        ],
    }


def test_builds_plan_plus_per_step_examples():
    ex = build_examples_from_episode(_episode())
    kinds = [e["metadata"]["kind"] for e in ex]
    assert kinds == ["plan", "action", "action"]
    # Each example is a system/user/assistant chat triple.
    for e in ex:
        roles = [m["role"] for m in e["messages"]]
        assert roles == ["system", "user", "assistant"]


def test_step2_input_is_teacher_free_and_thought_internalized():
    ex = build_examples_from_episode(_episode())
    step2 = ex[2]  # plan, action(t1), action(t2)
    user = step2["messages"][1]["content"]
    assistant = json.loads(step2["messages"][2]["content"])
    # Teacher block stripped from the input the model will be trained on.
    assert "Previous teacher guidance" not in user
    # The guidance the student saw going into step 2 (step 1's feedback) is internalized.
    assert assistant["thought"].startswith("Reflecting on my progress so far: I found the right doc.")
    assert assistant["thought"].endswith("Now I extract.")
    assert assistant["action"]["tool"] == "extract"


def test_first_step_has_no_internalized_guidance():
    ex = build_examples_from_episode(_episode())
    step1 = ex[1]
    assistant = json.loads(step1["messages"][2]["content"])
    assert assistant["thought"] == "I will search."  # no prior guidance to fold in


def test_build_dataset_concatenates_episodes():
    data = build_dataset([_episode(), _episode()])
    assert len(data) == 6  # (plan + 2 actions) x 2
