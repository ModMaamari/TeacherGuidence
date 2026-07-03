"""Tests for the experiment-matrix template generator."""

import yaml

from scripts.gen_experiment_matrix_templates import generate, MODELS, SETTINGS


def test_generates_20_templates(tmp_path):
    paths = generate(out_dir=tmp_path)
    assert len(paths) == len(MODELS) * len(SETTINGS) == 20
    assert all(p.exists() for p in paths)


def test_each_setting_has_expected_mode_config(tmp_path):
    generate(out_dir=tmp_path)
    loaded = {}
    for model_slug in MODELS:
        for setting_key in SETTINGS:
            with open(tmp_path / f"exp_matrix_{model_slug}_{setting_key}.yaml", encoding="utf-8") as f:
                loaded[(model_slug, setting_key)] = yaml.safe_load(f)

    # A: planning=5, running=5, default workflow, teacher-guided.
    a = loaded["qwen4b", "A"]
    assert a["mode_config"]["budget"] == 5
    assert a["workflows"] == ["hotpot_teacher_guided_b5_plan_review"]
    assert a["mode_config"]["plan_review"]["planner"] == "student"
    assert a["mode_config"]["skip_teacher"] is False

    # B: planning=10, but still the 5-step running workflow.
    b = loaded["qwen4b", "B"]
    assert b["mode_config"]["budget"] == 10
    assert b["workflows"] == ["hotpot_teacher_guided_b5_plan_review"]

    # C: the hidden-max-20 workflow, planning budget still 5.
    c = loaded["qwen4b", "C"]
    assert c["mode_config"]["budget"] == 5
    assert c["workflows"] == ["hotpot_teacher_guided_bmax20_plan_review"]

    # D: teacher authors the plan.
    d = loaded["qwen4b", "D"]
    assert d["mode_config"]["plan_review"]["planner"] == "teacher"

    # E: skip_teacher ablation.
    e = loaded["qwen4b", "E"]
    assert e["mode_config"]["skip_teacher"] is True


def test_all_templates_reference_the_fixed_30_question_dataset(tmp_path):
    generate(out_dir=tmp_path)
    for model_slug in MODELS:
        for setting_key in SETTINGS:
            with open(tmp_path / f"exp_matrix_{model_slug}_{setting_key}.yaml", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            ds = data["datasets"][0]
            assert ds["num_samples"] == 30
            assert "hotpot_teacher_guidance_exp30" in ds["path"]
            assert data["mode_config"]["teacher_model"] == "custom/z-ai/glm-5.2"


def test_student_model_matches_slug(tmp_path):
    generate(out_dir=tmp_path)
    with open(tmp_path / "exp_matrix_ornith9b_A.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["mode_config"]["student_model"] == MODELS["ornith9b"]
    assert data["id"] == "exp_matrix_ornith9b_A"


def test_filename_matches_template_id(tmp_path):
    # SimulationLoader resolves "<simulation_id>.yaml" directly under templates_dir with
    # no subdirectory search, so the on-disk filename must exactly match the id field.
    paths = generate(out_dir=tmp_path)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert path.name == f"{data['id']}.yaml"
