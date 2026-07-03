"""Tests for the pure/parsing helpers in the experiment-matrix orchestrator.

The orchestrator itself drives real subprocesses (ollama serve/pull, nvidia-smi,
`agentsim simulate`) and isn't meaningfully unit-testable beyond its planning and
parsing logic -- covered here.
"""

from scripts.gen_experiment_matrix_templates import MODELS, SETTINGS
from scripts.run_experiment_matrix import plan_runs, select_free_gpus


def test_plan_runs_covers_full_matrix_grouped_by_model():
    runs = plan_runs()
    assert len(runs) == len(MODELS) * len(SETTINGS) == 20

    # Grouped by model: all of one model's settings appear consecutively.
    seen_models = []
    for model_slug, _setting_key, _template_id in runs:
        if not seen_models or seen_models[-1] != model_slug:
            seen_models.append(model_slug)
    assert len(seen_models) == len(MODELS)

    for model_slug, setting_key, template_id in runs:
        assert template_id == f"exp_matrix_{model_slug}_{setting_key}"


def test_select_free_gpus_picks_least_used_ascending():
    csv = "0, 40000\n1, 0\n2, 79000\n3, 100\n4, 0\n5, 60000\n6, 500\n7, 12000\n"
    assert select_free_gpus(csv, n=4) == ["1", "4", "3", "6"]


def test_select_free_gpus_handles_fewer_rows_than_n():
    csv = "0, 100\n1, 50\n"
    assert select_free_gpus(csv, n=4) == ["1", "0"]
