"""Offline student policy for smoke tests (``--student mock``): same interface as
``VllmPolicy`` / ``PolicyModel``, answers come from ``agentsim.clients.mock_provider``."""
from __future__ import annotations

from typing import Any, Dict, List

from agentsim.clients.mock_provider import mock_completion


class MockPolicy:
    def __init__(self):
        self.last_stats: List[Dict[str, Any]] = []

    @staticmethod
    def _one(messages: List[Dict[str, str]]):
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
        text = mock_completion(prompt)
        stat = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4,
                "gen_s": 0.0, "batch_size": 1, "backend": "mock"}
        return text, stat

    def generate(self, messages, max_new_tokens: int = 700, temperature: float = 0.0) -> str:
        text, stat = self._one(messages)
        self.last_stats = [stat]
        return text

    def generate_with_stats(self, messages, max_new_tokens: int = 700, temperature: float = 0.0):
        return self._one(messages)

    def generate_batch(self, messages_list, max_new_tokens: int = 700, temperature: float = 0.0) -> List[str]:
        res = [self._one(m) for m in messages_list]
        self.last_stats = [s for _, s in res]
        return [t for t, _ in res]
