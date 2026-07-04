"""Tests for SFT diversity metrics and near-duplicate capping."""

from agentsim.teacher_guidance.sft_diversity import (
    signature,
    diversity_report,
    cap_by_signature,
    jaccard_similarity,
)


def _ep(qid, tools, used=None, eff=1.0):
    return {
        "qid": qid,
        "used_steps": used if used is not None else len(tools),
        "path_optimality": {"step_efficiency": eff},
        "steps": [{"student_action": {"action": {"tool": t}}} for t in tools],
    }


def test_signature_is_tool_sequence():
    assert signature(_ep("q", ["search", "extract", "finish"])) == "search>extract>finish"


def test_diversity_report_counts_unique_signatures():
    eps = [
        _ep("q1", ["search", "finish"]),
        _ep("q2", ["search", "finish"]),
        _ep("q3", ["search", "extract", "finish"]),
    ]
    rep = diversity_report(eps)
    assert rep["total"] == 3
    assert rep["unique_signatures"] == 2
    assert rep["top_signatures"][0] == ("search>finish", 2)


def test_cap_by_signature_limits_and_keeps_shortest():
    eps = [
        _ep("q1", ["search", "finish"], used=2),
        _ep("q2", ["search", "finish"], used=5),   # same signature, longer
        _ep("q3", ["search", "finish"], used=3),
        _ep("q4", ["decompose", "search", "finish"], used=3),
    ]
    kept = cap_by_signature(eps, max_per_signature=1)
    # One per signature; for the crowded signature the 2-step trace is kept.
    kept_by_sig = {signature(e): e for e in kept}
    assert len(kept) == 2
    assert kept_by_sig["search>finish"]["qid"] == "q1"


def test_cap_zero_is_noop():
    eps = [_ep("q1", ["search", "finish"]), _ep("q2", ["search", "finish"])]
    assert len(cap_by_signature(eps, 0)) == 2


def test_jaccard_similarity():
    assert jaccard_similarity("the cat sat on mat", "the cat sat on mat") == 1.0
    assert jaccard_similarity("the cat sat on mat", "a dog ran very fast today") == 0.0
    assert 0.0 < jaccard_similarity(
        "the cat sat on the mat", "the cat sat on the rug"
    ) < 1.0
