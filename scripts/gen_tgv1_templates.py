"""Generate the tg_v1 collection templates: one per dataset, plus matching smoke variants.

Smoke and production come from the SAME builder and differ only in ``num_samples``, so a
green smoke actually vouches for the configuration the full run will use -- the usual
failure of a smoke test is that it quietly tests something else.

Teacher is fixed to the free FAU DeepSeek-V4-Flash; the student is served by the same free
gateway while no GPU is available. Both are $0, so an 8,000-episode corpus costs nothing.

Usage::

    python scripts/gen_tgv1_templates.py            # writes tgv1_smoke_* and tgv1_*
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_fau_smoke_template import build_fau_smoke_template  # noqa: E402

TEACHER = "fau/deepseek-ai/DeepSeek-V4-Flash"
STUDENT = "fau/ibm-granite/granite-4.1-3b"
DATA_ROOT = "./data/datasets/tg_v1"

#: dataset -> (directory, file stem). The stem is what prepare_dataset.py wrote.
DATASETS = {
    "hotpotqa":        ("hotpotqa", "hotpotqa_train"),
    "2wikimultihopqa": ("2wikimultihopqa", "2wikimultihopqa_train"),
    "musique":         ("musique", "musique_train"),
    "strategyqa":      ("strategyqa", "strategyqa_train"),
}
PROD_SAMPLES = 2000
SMOKE_SAMPLES = 3


def build(dataset: str, num_samples: int, template_id: str, out_root: str) -> dict:
    d, stem = DATASETS[dataset]
    return build_fau_smoke_template(
        template_id=template_id,
        student_model=STUDENT,
        teacher_model=TEACHER,
        teacher_router=[TEACHER],          # single teacher: no silent fallback to a paid one
        num_samples=num_samples,
        questions_path=f"{DATA_ROOT}/{d}/{stem}_questions.jsonl",
        corpus_path=f"{DATA_ROOT}/{d}/{stem}_corpus.jsonl",
        output_dir=f"{out_root}/{template_id}",
        budget=3,
        workflow="hotpot_teacher_guided_b3_plan_review",
        planning_steps=3,
        max_plan_steps=6,
        disclose_budget=False,             # hidden budget: fewer steps, better answers
        student_use_response_schema=False, # the student is API-served, not grammar-constrained
    )


def main() -> int:
    out_dir = REPO_ROOT / "templates" / "simulations"
    written = []
    for dataset in DATASETS:
        for tag, n, root in (("smoke", SMOKE_SAMPLES, "./data/simulation_output/tgv1_smoke"),
                             ("prod", PROD_SAMPLES, "./data/simulation_output/tgv1")):
            tid = f"tgv1_{dataset}" if tag == "prod" else f"tgv1_smoke_{dataset}"
            (out_dir / f"{tid}.yaml").write_text(
                yaml.safe_dump(build(dataset, n, tid, root), sort_keys=False), encoding="utf-8")
            written.append(tid)
    print(f"wrote {len(written)} templates: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
