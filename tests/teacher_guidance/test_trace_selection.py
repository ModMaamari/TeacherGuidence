"""Tests for canonical (one optimal trace per question) selection."""

from agentsim.teacher_guidance.trace_selection import select_canonical


def _ep(qid, eid, steps, eff=1.0, wasted=0, f1=1.0, model="m"):
    return {
        "qid": qid, "episode_id": eid, "used_steps": steps, "student_model": model,
        "final_metrics": {"f1": f1},
        "path_optimality": {"step_efficiency": eff, "wasted_step_count": wasted},
    }


def test_one_trace_per_qid():
    eps = [_ep("q1", "a", 3), _ep("q1", "b", 5), _ep("q2", "c", 4)]
    canon = select_canonical(eps)
    assert {e["qid"] for e in canon} == {"q1", "q2"}


def test_prefers_fewest_steps():
    eps = [_ep("q1", "long", 6), _ep("q1", "short", 3)]
    assert select_canonical(eps)[0]["episode_id"] == "short"


def test_ties_broken_by_efficiency_then_f1():
    # Same step count: higher efficiency wins.
    eps = [_ep("q1", "sloppy", 4, eff=0.5), _ep("q1", "clean", 4, eff=1.0)]
    assert select_canonical(eps)[0]["episode_id"] == "clean"
    # Same steps and efficiency: higher F1 wins.
    eps2 = [_ep("q2", "vague", 4, eff=1.0, f1=0.3), _ep("q2", "precise", 4, eff=1.0, f1=0.9)]
    assert select_canonical(eps2)[0]["episode_id"] == "precise"


def test_output_sorted_by_qid_and_deterministic():
    eps = [_ep("q2", "a", 3), _ep("q1", "b", 3)]
    canon = select_canonical(eps)
    assert [e["qid"] for e in canon] == ["q1", "q2"]


def test_deterministic_tie_break_on_model_and_id():
    # Fully tied on quality -> stable pick by (model, episode_id).
    eps = [_ep("q1", "z", 3, model="b"), _ep("q1", "a", 3, model="a")]
    assert select_canonical(eps)[0]["episode_id"] == "a"
