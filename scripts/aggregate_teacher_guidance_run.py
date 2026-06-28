"""
Summarize a Teacher Guidance run: aggregate metrics across all episodes.

Usage:
    python scripts/aggregate_teacher_guidance_run.py --run-dir data/simulation_output/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_episodes(run_dir: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for path in run_dir.rglob("teacher_guidance_episodes.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    episodes = _load_episodes(Path(args.run_dir))
    if not episodes:
        print("No episodes found.")
        return

    em = [1.0 if e["final_metrics"].get("exact_match") else 0.0 for e in episodes]
    f1 = [float(e["final_metrics"].get("f1", 0.0)) for e in episodes]
    doc_recall = [float(e["final_metrics"].get("supporting_doc_recall", 0.0)) for e in episodes]
    fact_recall = [float(e["final_metrics"].get("supporting_fact_recall", 0.0)) for e in episodes]
    steps = [len(e.get("steps", [])) for e in episodes]

    stop_reasons: Dict[str, int] = {}
    leak_episodes = 0
    invalid_json_steps = 0
    total_steps = 0
    for e in episodes:
        stop_reasons[e.get("stop_reason", "?")] = stop_reasons.get(e.get("stop_reason", "?"), 0) + 1
        leaked = False
        for s in e.get("steps", []):
            total_steps += 1
            if not s.get("metrics", {}).get("json_valid", True):
                invalid_json_steps += 1
            lk = s.get("leakage_check", {}) or {}
            if any(lk.get(k) for k in ("gold_answer_leaked", "hidden_doc_id_leaked", "hidden_title_leaked", "hidden_span_leaked")):
                leaked = True
        if leaked:
            leak_episodes += 1

    print(f"Episodes:                {len(episodes)}")
    print(f"Guidance level:          {episodes[0].get('guidance_level')}")
    print(f"Student / Teacher model: {episodes[0].get('student_model')} / {episodes[0].get('teacher_model')}")
    print("-" * 48)
    print(f"Exact match (mean):      {_mean(em)}")
    print(f"F1 (mean):               {_mean(f1)}")
    print(f"Supporting doc recall:   {_mean(doc_recall)}")
    print(f"Supporting fact recall:  {_mean(fact_recall)}")
    print(f"Avg steps to finish:     {_mean([float(s) for s in steps])}")
    print("-" * 48)
    print(f"Stop reasons:            {stop_reasons}")
    print(f"Invalid-JSON steps:      {invalid_json_steps}/{total_steps}")
    print(f"Episodes with leakage:   {leak_episodes}/{len(episodes)}")


if __name__ == "__main__":
    main()
