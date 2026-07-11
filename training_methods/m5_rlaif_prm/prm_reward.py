"""PRM-based episode reward for RLAIF (plugged into the m4 GRPO trainer).

reward = mean(PRM step scores) - invalid-action penalty. Uses NO gold information:
this is the "internalized teacher as critic" reward. The PRM adapter path and device
come from environment variables so the function can be passed by dotted name to
train_grpo.py --reward-fn:

    PRM_ADAPTER=<path> PRM_DEVICE=cuda:1 CUDA_VISIBLE_DEVICES=0,1 \
      .venv_train/bin/python training_methods/m4_grpo/train_grpo.py \
      --reward-fn training_methods.m5_rlaif_prm.prm_reward.episode_reward ...

(Policy on cuda:0, PRM on cuda:1 -> two of the four A100s.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SCORER = None

INVALID_STEP_PENALTY = 0.1
NO_FINISH_PENALTY = 0.2


def _scorer():
    global _SCORER
    if _SCORER is None:
        from training_methods.m5_rlaif_prm.prm_score import PRMScorer

        adapter = os.environ.get("PRM_ADAPTER")
        if not adapter:
            raise RuntimeError("PRM_ADAPTER env var must point to a trained PRM adapter")
        _SCORER = PRMScorer(adapter, device=os.environ.get("PRM_DEVICE", "cuda:1"))
    return _SCORER


def episode_reward(episode: Dict[str, Any], weights=None) -> Dict[str, Any]:
    scores = _scorer().score_episode_steps(episode)
    n_invalid = sum(1 for s in episode["steps"] if not s.get("action_valid"))
    r = (sum(scores) / len(scores)) if scores else 0.0
    r -= INVALID_STEP_PENALTY * n_invalid
    if episode.get("stop_reason") == "budget_forced_finish_no_finish":
        r -= NO_FINISH_PENALTY
    return {"reward": round(r, 6), "n_invalid": n_invalid,
            "used_steps": episode.get("used_steps"), "prm_step_scores": [round(s, 4) for s in scores]}
