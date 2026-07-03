"""Guards the hidden-budget workflow's shape: exactly 20 agent-step components, each
displaying budget 5 (never revealing the real 20-step cap), with force_finish only on
the last one. A hand-edit typo here would silently break the "hidden max" ablation.
"""

from agentsim.workflow.loader import WorkflowLoader


def test_bmax20_workflow_has_20_hidden_budget_steps():
    workflow = WorkflowLoader().load_workflow("hotpot_teacher_guided_bmax20_plan_review")

    assert workflow.components[0]["type"] == "teacher_guided_plan_review"

    agent_steps = [c for c in workflow.components if c["type"] == "teacher_guided_agent_step"]
    assert len(agent_steps) == 20
    assert [c["config"]["step_index"] for c in agent_steps] == list(range(1, 21))

    # Every step displays "budget 5" -- the real 20-step cap is never surfaced anywhere.
    assert all(c["config"]["budget"] == 5 for c in agent_steps)

    force_finish_flags = [c["config"]["force_finish"] for c in agent_steps]
    assert force_finish_flags == [False] * 19 + [True]

    assert workflow.config["max_steps"] == len(workflow.components)
