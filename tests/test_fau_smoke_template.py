"""Tests for the FAU gpt-oss-120b smoke template generator."""

from scripts.gen_fau_smoke_template import build_fau_smoke_template


def test_template_uses_fau_teacher_and_ollama_student():
    t = build_fau_smoke_template()
    assert t["mode_config"]["teacher_model"] == "fau/gpt-oss-120b"
    assert t["mode_config"]["student_model"].startswith("ollama/")


def test_template_gives_reasoning_teacher_a_generous_token_budget():
    mc = build_fau_smoke_template()["mode_config"]
    # A reasoning model spends tokens on chain-of-thought before the JSON verdict.
    assert mc["teacher_max_tokens"] >= 2000
    assert mc["teacher_max_tokens_retry"] > mc["teacher_max_tokens"]


def test_template_defaults_to_eight_samples():
    assert build_fau_smoke_template()["datasets"][0]["num_samples"] == 8


def test_per_worker_overrides_are_respected():
    t = build_fau_smoke_template(
        template_id="fau_smoke_w3", num_samples=1,
        questions_path="/tmp/w3.jsonl", output_dir="/tmp/out/w3",
    )
    assert t["id"] == "fau_smoke_w3"
    assert t["datasets"][0]["num_samples"] == 1
    assert t["datasets"][0]["path"] == "/tmp/w3.jsonl"
    assert t["output_dir"] == "/tmp/out/w3"
