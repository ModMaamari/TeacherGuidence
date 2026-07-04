"""Tests for the SFT report + leakage healthcheck."""

from agentsim.teacher_guidance.sft_report import (
    leakage_healthcheck,
    tool_distribution,
    stratum_balance,
    build_report,
)


def _episode(qid, gold, question, guidances):
    """guidances[i] = the student_visible_guidance produced at step i."""
    steps = []
    for i, g in enumerate(guidances):
        steps.append({
            "t": i + 1,
            "student_visible_guidance": g,
            "leakage_check": {"gold_answer_leaked": False},
            "student_action": {"action": {"tool": "search"}},
        })
    return {"qid": qid, "gold_answer": gold, "query": question, "steps": steps}


def test_healthcheck_clean_when_no_leak():
    ep = _episode("q1", "Delhi", "Where is the Oberoi Group based?",
                  [{"feedback": "You retrieved the right doc; keep going."},
                   {"feedback": "Now extract the location fact."}])
    hc = leakage_healthcheck([ep])
    assert hc["clean"] is True
    assert hc["reflection_leaks"] == []


def test_healthcheck_flags_reflection_leak():
    # Step 2's reflection comes from step 1's guidance, which names the gold answer.
    ep = _episode("q1", "Delhi", "Where is the Oberoi Group based?",
                  [{"feedback": "The answer is Delhi, now confirm it."},
                   {"feedback": "done"}])
    hc = leakage_healthcheck([ep])
    assert hc["clean"] is False
    assert hc["reflection_leaks"][0]["qid"] == "q1"


def test_healthcheck_answer_in_question_is_not_a_leak():
    # Gold answer appears in the question -> the teacher echoing it is not a leak.
    ep = _episode("q1", "Roger Federer", "Who won more, Roger Federer or Murray?",
                  [{"feedback": "Good, Roger Federer is the one to focus on."},
                   {"feedback": "done"}])
    hc = leakage_healthcheck([ep])
    assert hc["clean"] is True


def test_healthcheck_counts_flagged_steps():
    ep = _episode("q1", "Delhi", "q", [{"feedback": "ok"}])
    ep["steps"][0]["leakage_check"]["gold_answer_leaked"] = True
    hc = leakage_healthcheck([ep])
    assert hc["flagged_gold_answer_leaked_steps"] == 1
    assert hc["clean"] is False


def test_tool_distribution_and_stratum_balance():
    examples = [
        {"metadata": {"kind": "action", "tool": "search"}},
        {"metadata": {"kind": "action", "tool": "search"}},
        {"metadata": {"kind": "action", "tool": "finish"}},
        {"metadata": {"kind": "plan"}},
    ]
    assert tool_distribution(examples) == {"search": 2, "finish": 1}

    eps = [{"qid": "q1", "gold_answer": "yes"}, {"qid": "q2", "gold_answer": "Delhi"},
           {"qid": "q2", "gold_answer": "Delhi"}]  # q2 duplicated -> counted once
    assert stratum_balance(eps) == {"boolean": 1, "entity": 1}


def test_build_report_shape():
    ep = _episode("q1", "Delhi", "q", [{"feedback": "ok"}])
    report = build_report([ep], [{"metadata": {"kind": "action", "tool": "search"}}])
    assert report["episodes"] == 1
    assert report["leakage_healthcheck"]["clean"] is True
    assert "diversity" in report
