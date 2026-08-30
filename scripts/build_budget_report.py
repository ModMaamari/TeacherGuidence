"""Turn the measured cost probes into the corpus budget model + report data.

Two things were measured directly (EdenAI billing, g3_plan config): every teacher on
HotpotQA, and every dataset with one teacher. The full teacher x dataset matrix is
*derived* from those two axes -- a dataset's cost scaled by the teacher's HotpotQA cost
ratio -- because measuring all twelve cells buys little: per-episode spend is set by the
protocol (calls per episode), which is what both axes already capture.

Uncertainty is carried as a flat +30% planning contingency, matching the largest observed
per-episode standard deviation (GLM 29%, Qwen 30%). Budget against the planning column.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

#: EUR->USD, open.er-api.com, fetched 2026-08-30.
RATE = 1.160718
#: Planning contingency over the measured mean (see module docstring).
CONTINGENCY = 0.30

DATASETS = {
    "HotpotQA":        {"mean": 0.044751, "sd": 0.013381, "n": 5, "hops": "2",
                        "gold": "sentence", "license": "CC BY-SA 4.0"},
    "2WikiMultihopQA": {"mean": 0.052979, "sd": 0.003837, "n": 3, "hops": "2-4",
                        "gold": "sentence", "license": "Apache-2.0"},
    "MuSiQue":         {"mean": 0.046989, "sd": 0.009665, "n": 3, "hops": "2-4",
                        "gold": "paragraph", "license": "CC BY 4.0"},
    "StrategyQA":      {"mean": 0.042453, "sd": 0.010698, "n": 3, "hops": "2-5",
                        "gold": "paragraph", "license": "MIT"},
}
TEACHERS = {
    "Qwen3.8-2.4T-A95B": {"mean": 0.044751, "sd": 0.013381, "n": 5,
                          "id": "qwen/qwen3.8-2.4t-a95b"},
    "Kimi K3":           {"mean": 0.059866, "sd": 0.014164, "n": 5,
                          "id": "qwen/kimi-k3"},
    "GLM-5.3":           {"mean": 0.079975, "sd": 0.023236, "n": 5,
                          "id": "deepinfra/zai-org/GLM-5.3"},
}
BASE = "Qwen3.8-2.4T-A95B"


def build() -> dict:
    ratio = {t: v["mean"] / TEACHERS[BASE]["mean"] for t, v in TEACHERS.items()}
    matrix = {t: {d: DATASETS[d]["mean"] * ratio[t] for d in DATASETS} for t in TEACHERS}
    per_teacher_avg = {t: sum(row.values()) / len(row) for t, row in matrix.items()}
    scales = [1000, 10000, 120000]
    scenarios = {
        t: {str(n): {"usd": per_teacher_avg[t] * n,
                     "usd_plan": per_teacher_avg[t] * n * (1 + CONTINGENCY)}
            for n in scales}
        for t in TEACHERS
    }
    # The mixed corpus: 1,000 episodes in every teacher x dataset cell.
    mix_cells = {t: {d: c * 1000 for d, c in row.items()} for t, row in matrix.items()}
    mix_total = sum(sum(r.values()) for r in mix_cells.values())
    return {
        "generated_for": "teacher-guidance corpus budget",
        "rate_eur_usd": RATE,
        "contingency": CONTINGENCY,
        "datasets": DATASETS,
        "teachers": TEACHERS,
        "ratio": ratio,
        "matrix": matrix,
        "per_teacher_avg": per_teacher_avg,
        "scenarios": scenarios,
        "mix": {"cells": mix_cells, "total_usd": mix_total,
                "total_eur": mix_total / RATE, "episodes": 12000,
                "total_usd_plan": mix_total * (1 + CONTINGENCY)},
    }


if __name__ == "__main__":
    data = build()
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/budget/budget_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"12,000-episode mixed corpus: ${data['mix']['total_usd']:,.2f} "
          f"(EUR {data['mix']['total_eur']:,.2f}); plan ${data['mix']['total_usd_plan']:,.2f}")
