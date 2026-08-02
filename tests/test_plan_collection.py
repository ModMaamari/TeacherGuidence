"""Tests for the model registry and the collection-matrix planner.

The planner decides what a six-figure-episode run will actually do, so the properties that
matter are structural: batches must be the right size, the anchor set must genuinely be
shared by every combination, non-anchor questions must never be silently reused across
combinations, and the whole thing must be reproducible from a seed.
"""

from __future__ import annotations

import json

import pytest

from agentsim.teacher_guidance.model_registry import (
    STUDENTS,
    TEACHERS,
    combination_id,
    get_student,
    get_teacher,
)
from scripts.plan_collection import (
    CONFIGS,
    assign_batches,
    build_plan,
    load_question_pool,
)


def _pool(n, datasets=("hotpotqa", "musique", "2wikimultihopqa")):
    return [
        {"qid": f"q{i}", "dataset": datasets[i % len(datasets)], "file": "f.jsonl"}
        for i in range(n)
    ]


COMBOS = [combination_id(t, s) for t in TEACHERS for s in STUDENTS]


# --- registry ----------------------------------------------------------------------
def test_registry_has_three_teachers_and_four_students():
    assert len(TEACHERS) == 3
    assert len(STUDENTS) == 4
    assert len(COMBOS) == 12


def test_every_teacher_has_a_multi_source_fallback_chain():
    for name, spec in TEACHERS.items():
        assert len(spec.sources) >= 1, name
        assert spec.router[0] == spec.primary
        assert len(set(spec.router)) == len(spec.router), f"{name} has duplicate sources"
        assert spec.license, name
        # a reasoning teacher must be given room to think before the JSON verdict
        assert spec.max_tokens >= 2000 and spec.max_tokens_retry > spec.max_tokens


def test_teacher_model_ids_carry_a_provider_prefix():
    """The router resolves the provider from the id prefix, so a bare id would be
    silently routed to the default provider."""
    for spec in TEACHERS.values():
        for source in spec.sources:
            assert source.model_id.split("/")[0] in {"edenai", "custom", "fau", "ollama", "vllm"}


def test_students_are_served_under_their_real_hf_id():
    for spec in STUDENTS.values():
        assert spec.served_model == f"vllm/{spec.hf_id}"
        assert spec.license


def test_composite_qwen_students_are_flagged_for_merging():
    """vLLM cannot LoRA-serve Qwen3.5's composite architecture; forgetting this silently
    evaluates the base model instead of the trained one."""
    assert get_student("qwen-0.8b").needs_merge_for_vllm is True
    assert get_student("qwen-2b").needs_merge_for_vllm is True
    assert get_student("granite-3b").needs_merge_for_vllm is False


def test_unknown_names_raise_with_a_helpful_message():
    with pytest.raises(KeyError, match="known"):
        get_teacher("nope")
    with pytest.raises(KeyError, match="known"):
        get_student("nope")


# --- batching -----------------------------------------------------------------------
def test_every_combination_gets_the_requested_batch_size():
    batches, anchor = assign_batches(_pool(2000), COMBOS, per_combination=100, anchor=10)
    assert len(batches) == 12
    assert all(len(b) == 100 for b in batches.values())
    assert len(anchor) == 10


def test_anchor_questions_are_shared_by_every_combination():
    """The anchor set is what makes teachers and students comparable on identical items."""
    batches, anchor = assign_batches(_pool(2000), COMBOS, per_combination=100, anchor=10)
    anchor_ids = {q["qid"] for q in anchor}
    for combo, questions in batches.items():
        assert anchor_ids <= {q["qid"] for q in questions}, combo


def test_non_anchor_questions_are_never_reused_across_combinations():
    batches, anchor = assign_batches(_pool(2000), COMBOS, per_combination=100, anchor=10)
    anchor_ids = {q["qid"] for q in anchor}
    seen = set()
    for questions in batches.values():
        unique = {q["qid"] for q in questions} - anchor_ids
        assert not (unique & seen), "a non-anchor question appeared in two batches"
        seen |= unique


def test_batches_are_stratified_across_datasets():
    batches, _ = assign_batches(_pool(2000), COMBOS, per_combination=90, anchor=9)
    for combo, questions in batches.items():
        datasets = {q["dataset"] for q in questions}
        assert len(datasets) == 3, f"{combo} saw only {datasets}"


def test_assignment_is_reproducible_from_the_seed():
    a, _ = assign_batches(_pool(2000), COMBOS, 100, 10, seed=7)
    b, _ = assign_batches(_pool(2000), COMBOS, 100, 10, seed=7)
    c, _ = assign_batches(_pool(2000), COMBOS, 100, 10, seed=8)
    assert {k: [q["qid"] for q in v] for k, v in a.items()} == \
           {k: [q["qid"] for q in v] for k, v in b.items()}
    assert {k: [q["qid"] for q in v] for k, v in a.items()} != \
           {k: [q["qid"] for q in v] for k, v in c.items()}


def test_too_small_a_pool_fails_loudly_rather_than_reusing_questions():
    with pytest.raises(ValueError, match="pool too small"):
        assign_batches(_pool(100), COMBOS, per_combination=100, anchor=10)


def test_anchor_larger_than_batch_is_rejected():
    with pytest.raises(ValueError, match="anchor cannot exceed"):
        assign_batches(_pool(2000), COMBOS, per_combination=10, anchor=50)


# --- plan manifest ------------------------------------------------------------------
def test_plan_produces_one_cell_per_combination_and_config():
    batches, anchor = assign_batches(_pool(2000), COMBOS, 100, 10)
    plan = build_plan(batches, anchor, ["g3_plan", "g0_plan"], 100, seed=13)
    assert len(plan["cells"]) == 12 * 2
    assert plan["assignments"] == 12 * 100
    assert plan["total_episodes"] == 12 * 100 * 2


def test_plan_scales_to_the_target_corpus_size():
    """3 teachers x 4 students x 5000 questions x 2 configs = 120k episodes."""
    batches, anchor = assign_batches(_pool(60_000), COMBOS, per_combination=5000, anchor=500)
    plan = build_plan(batches, anchor, ["g3_plan", "g0_plan"], 5000, seed=13)
    assert plan["assignments"] == 60_000
    assert plan["total_episodes"] == 120_000
    assert plan["unique_questions"] == 500 + 12 * 4500


def test_each_cell_carries_everything_a_runner_needs():
    batches, anchor = assign_batches(_pool(2000), COMBOS, 100, 10)
    plan = build_plan(batches, anchor, ["g3_plan"], 100, seed=13)
    cell = plan["cells"][0]
    for key in ("cell_id", "teacher", "teacher_model", "teacher_router", "student",
                "student_model", "student_hf_id", "config", "guidance_level",
                "plan_review", "budget", "qids"):
        assert key in cell, key
    assert cell["teacher_model"] == cell["teacher_router"][0]
    assert len(cell["qids"]) == cell["num_questions"]


def test_configs_vary_one_axis_at_a_time():
    """g3_plan vs g0_plan must differ ONLY in guidance level, or the dose-response
    comparison confounds supervision richness with planning."""
    a, b = CONFIGS["g3_plan"], CONFIGS["g0_plan"]
    differing = {k for k in a if k != "description" and a[k] != b.get(k)}
    assert differing == {"guidance_level"}


def test_skip_teacher_config_is_a_real_control_arm():
    assert CONFIGS["skip_teacher"]["skip_teacher"] is True


def test_plan_is_json_serializable():
    batches, anchor = assign_batches(_pool(2000), COMBOS, 100, 10)
    plan = build_plan(batches, anchor, ["g3_plan"], 100, seed=13)
    assert json.loads(json.dumps(plan))["total_episodes"] == 1200


def test_load_question_pool_reads_prepared_files(tmp_path):
    rows = [{"id": "q1", "source": "hotpotqa"}, {"id": "q2", "source": "hotpotqa"}]
    (tmp_path / "hotpotqa_train_questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    pool = load_question_pool(tmp_path)
    assert [q["qid"] for q in pool] == ["q1", "q2"]
    assert all(q["dataset"] == "hotpotqa" for q in pool)
