"""
Generate the 4-model x 5-setting experiment-matrix simulation templates.

Settings (teacher is always custom/z-ai/glm-5.2):
    A - planning budget=5, running budget=5
    B - planning budget=10, running budget=5
    C - planning budget=5, running budget hidden-max=20 (never stated in a prompt)
    D - planning budget=5, running budget=5, teacher authors the plan
    E - no teacher guidance at all (student writes the plan, zero teacher LLM calls)

Writes one YAML file per (model, setting) as templates/simulations/exp_matrix_<model_slug>_<setting>.yaml
(flat, alongside the other simulation templates -- SimulationLoader resolves a template
id straight to "<templates_dir>/<id>.yaml" with no subdirectory search), all pointed at
the same fixed 30-question dataset (materialized separately by
prepare_hotpot_teacher_guidance.py --shuffle --seed 2026 --limit 30).

Usage:
    python scripts/gen_experiment_matrix_templates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "templates" / "simulations"
DATASET_DIR = "./data/datasets/hotpot_teacher_guidance_exp30"
TEACHER_MODEL = "custom/z-ai/glm-5.2"
NUM_SAMPLES = 30

MODELS = {
    "qwen4b": "ollama/qwen3.5:4b",
    "qwen2b": "ollama/qwen3.5:2b",
    "qwen0p8b": "ollama/qwen3.5:0.8b",
    "ornith9b": "ollama/hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest",
}

# Each setting: (workflow_id, mode_config overrides beyond the shared defaults, description).
SETTINGS = {
    "A": {
        "workflow": "hotpot_teacher_guided_b5_plan_review",
        "budget": 5,
        "planner": "student",
        "skip_teacher": False,
        "desc": "planning budget=5, running budget=5",
    },
    "B": {
        "workflow": "hotpot_teacher_guided_b5_plan_review",
        "budget": 10,
        "planner": "student",
        "skip_teacher": False,
        "desc": "planning budget=10, running budget=5",
    },
    "C": {
        "workflow": "hotpot_teacher_guided_bmax20_plan_review",
        "budget": 5,
        "planner": "student",
        "skip_teacher": False,
        "desc": "planning budget=5, running budget hidden-max=20 (never stated in a prompt)",
    },
    "D": {
        "workflow": "hotpot_teacher_guided_b5_plan_review",
        "budget": 5,
        "planner": "teacher",
        "skip_teacher": False,
        "desc": "planning budget=5, running budget=5, teacher authors the plan",
    },
    "E": {
        "workflow": "hotpot_teacher_guided_b5_plan_review",
        "budget": 5,
        "planner": "student",
        "skip_teacher": True,
        "desc": "no teacher guidance at all: student writes the plan, zero teacher LLM calls",
    },
}


def build_template(model_slug: str, model_id: str, setting_key: str, setting: dict) -> dict:
    template_id = f"exp_matrix_{model_slug}_{setting_key}"
    mode_config = {
        "budget": setting["budget"],
        "student_model": model_id,
        "teacher_model": TEACHER_MODEL,
        "corpus_path": f"{DATASET_DIR}/hotpot_distractor_validation_corpus.jsonl",
        "retrieval_backend": "hotpot_local",
        "skip_teacher": setting["skip_teacher"],
        "guidance": {
            "level": 3,
            "name": "diagnostic_feedback",
            "score_mode": "continuous",
            "max_feedback_words": 60,
            "expose_next_action_hint": False,
            "expose_tool_hint": False,
            "expose_query_hint": False,
            "expose_doc_title_hint": False,
            "expose_gold_answer_hint": False,
            "leak_policy": "strict",
        },
        "plan_review": {
            "enabled": True,
            "planner": setting["planner"],
            "planning_steps": 1,
            "formal_plan": False,
            "review_guidance_level": 3,
            "max_initial_plan_steps": 6,
            "max_revised_plan_steps": 6,
            "consume_budget": False,
            "include_revised_plan_in_student_context": True,
            "allow_teacher_to_suggest_tools": True,
            "allow_teacher_to_suggest_queries": False,
            "allow_teacher_to_reveal_gold_titles": False,
            "allow_teacher_to_reveal_gold_answer": False,
        },
    }
    return {
        "id": template_id,
        "name": f"Experiment matrix: {model_slug} / setting {setting_key} ({setting['desc']})",
        "mode": "standard",
        "teacher_models": [
            {"name": "student", "model_id": model_id, "role": "teacher", "temperature": 0.2}
        ],
        "consultant_models": [],
        "workflows": [setting["workflow"]],
        "datasets": [
            {
                "name": "hotpot_questions",
                "path": f"{DATASET_DIR}/hotpot_distractor_validation_questions.jsonl",
                "num_samples": NUM_SAMPLES,
                "sample_strategy": "sequential",
            }
        ],
        "max_iterations": 1,
        "similarity_metric": "token_overlap",
        "similarity_threshold": 0.0,
        "verification": {"enabled": False},
        "mode_config": mode_config,
        "output_dir": f"./data/simulation_output/exp_matrix/{model_slug}_{setting_key}",
    }


def generate(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model_slug, model_id in MODELS.items():
        for setting_key, setting in SETTINGS.items():
            template = build_template(model_slug, model_id, setting_key, setting)
            path = out_dir / f"{template['id']}.yaml"
            header = (
                f"# Auto-generated by scripts/gen_experiment_matrix_templates.py -- do not hand-edit.\n"
                f"# Setting {setting_key}: {setting['desc']}\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                yaml.dump(template, f, sort_keys=False, default_flow_style=False, width=100)
            written.append(path)
    return written


if __name__ == "__main__":
    paths = generate()
    print(f"Wrote {len(paths)} templates -> {OUT_DIR}")
