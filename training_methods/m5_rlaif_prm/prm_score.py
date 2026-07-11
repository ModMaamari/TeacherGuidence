"""PRM scoring: continuous step scores from the generative digit judge.

One forward pass per (state, action): softmax over the ten digit tokens at the
first generated position, expected value / 9 -> score in [0, 1].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_methods.m5_rlaif_prm.build_dataset import judge_prompt  # noqa: E402
from agentsim.teacher_guidance.sft_internalize import strip_teacher_guidance_block  # noqa: E402


class PRMScorer:
    def __init__(self, adapter_path: str, model_path: str = "ibm-granite/granite-4.1-3b",
                 device: str = "cuda:0", max_length: int = 8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                     device_map=device)
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        self.max_length = max_length
        self.digit_ids = []
        for d in range(10):
            ids = self.tokenizer.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1, f"digit {d} is not a single token"
            self.digit_ids.append(ids[0])

    def score(self, state_prompt: str, action: Dict) -> float:
        import torch

        msgs = judge_prompt(strip_teacher_guidance_block(state_prompt), action)
        enc = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        if enc["input_ids"].shape[1] > self.max_length:
            return 0.5  # neutral for over-long states (rare; logged by callers)
        enc = enc.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1, :]
        digit_logits = logits[self.digit_ids].float()
        probs = torch.softmax(digit_logits, dim=-1)
        expected = float((probs * torch.arange(10, dtype=torch.float32, device=probs.device)).sum())
        return expected / 9.0

    def score_episode_steps(self, episode: Dict) -> List[float]:
        out = []
        for s in episode.get("steps") or []:
            action = s.get("student_action") or {}
            if not (action.get("action") or {}).get("tool"):
                out.append(0.0)
                continue
            out.append(self.score(s.get("student_prompt") or "", action))
        return out
