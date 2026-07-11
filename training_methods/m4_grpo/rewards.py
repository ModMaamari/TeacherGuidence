"""Episode reward for GRPO/RLAIF training (verifiable, gold-based).

Mirrors the spirit of best_answer_score: correctness first, then efficiency.
Deliberately does NOT use cover_match alone (gameable by verbose answers) --
F1 + EM carry the signal, cover adds a small bonus, and a step-efficiency bonus
is only granted when the answer is actually right (never reward finishing early
with garbage). Invalid actions are penalized so the JSON format stays intact
under RL pressure.
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_WEIGHTS = {
    "f1": 1.0,
    "em": 0.5,
    "cover": 0.25,
    "efficiency": 0.3,   # only when f1 >= efficiency_gate
    "efficiency_gate": 0.5,
    "invalid_step_penalty": 0.1,
    "no_finish_penalty": 0.2,  # forced derive_final_answer fallback
}


def episode_reward(episode: Dict[str, Any], weights: Dict[str, float] | None = None) -> Dict[str, Any]:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    m = episode["final_metrics"]
    budget = max(int(episode.get("budget", 1)), 1)
    used = max(int(episode.get("used_steps", budget)), 1)

    r = w["f1"] * float(m["f1"])
    r += w["em"] * (1.0 if m["exact_match"] else 0.0)
    r += w["cover"] * (1.0 if m["cover_match"] else 0.0)
    if float(m["f1"]) >= w["efficiency_gate"] and budget > 1:
        r += w["efficiency"] * (budget - used) / (budget - 1)
    n_invalid = sum(1 for s in episode["steps"] if not s.get("action_valid"))
    r -= w["invalid_step_penalty"] * n_invalid
    if episode.get("stop_reason") == "budget_forced_finish_no_finish":
        r -= w["no_finish_penalty"]
    return {"reward": round(r, 6), "n_invalid": n_invalid, "used_steps": used}
