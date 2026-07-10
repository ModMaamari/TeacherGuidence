import math

from scripts.best_answer_score import (
    WEIGHTS,
    best_answer_score,
    score_components,
)


def _episode(teacher=1.0, used=2, f1=0.8, doc_recall=1.0, em=False, cover=True,
             final="Paris", gold="Paris"):
    return {
        "used_steps": used,
        "final_answer": final,
        "gold_answer": gold,
        "final_metrics": {
            "teacher_answer_score": teacher,
            "f1": f1,
            "supporting_doc_recall": doc_recall,
            "exact_match": em,
            "answer_correct": cover,
        },
    }


def test_weights_sum_to_one():
    assert math.isclose(sum(WEIGHTS.values()), 1.0)


def test_teacher_wrong_gates_score_to_zero():
    score, comps = best_answer_score(_episode(teacher=0.0, em=True, f1=1.0))
    assert score == 0.0
    assert comps["f1"] == 1.0  # components still reported for telemetry


def test_score_in_unit_interval_and_perfect_episode_scores_one():
    score, _ = best_answer_score(_episode(teacher=1.0, used=1, f1=1.0,
                                          doc_recall=1.0, em=True, cover=True))
    assert math.isclose(score, 1.0)


def test_fewer_steps_scores_higher():
    fast, _ = best_answer_score(_episode(used=2))
    slow, _ = best_answer_score(_episode(used=7))
    assert fast > slow


def test_step_component_normalized_to_global_budget():
    assert score_components(_episode(used=1))["steps"] == 1.0
    assert score_components(_episode(used=9))["steps"] == 0.0


def test_cover_match_bonus_requires_full_teacher_score():
    partial = score_components(_episode(teacher=0.6, cover=True))
    full = score_components(_episode(teacher=1.0, cover=True))
    assert partial["cover_match"] == 0.0
    assert full["cover_match"] == 1.0


def test_length_similarity():
    same = score_components(_episode(final="Paris", gold="Paris"))["length"]
    longer = score_components(_episode(final="Paris, the capital of France", gold="Paris"))["length"]
    assert same == 1.0
    assert 0 < longer < 1
    assert score_components(_episode(final="", gold="Paris"))["length"] == 0.0


def test_higher_f1_and_recall_score_higher():
    lo, _ = best_answer_score(_episode(f1=0.2, doc_recall=0.5))
    hi, _ = best_answer_score(_episode(f1=0.9, doc_recall=1.0))
    assert hi > lo
