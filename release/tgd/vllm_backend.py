"""vLLM-server policy backend for the eval agents.

Drop-in replacement for ``hf_agent_loop.PolicyModel`` that talks to a vLLM
OpenAI-compatible server instead of running HF ``model.generate`` in-process:

  * continuous batching on the server: N concurrent episodes decode together,
    so per-episode latency barely grows with concurrency (bf16, LoRA adapters
    served unmerged);
  * same fairness contract as the HF path -- the server applies the model's own
    chat template, decodes in bf16, honors per-request temperature/top_p and a
    per-request ``seed`` for reproducible sampled reps;
  * same instrumentation contract -- ``last_stats`` carries per-row
    prompt/completion token counts (from the server's ``usage``) and wall time.

Start a server with ``scripts/serve_vllm.sh``; check readiness
with ``wait_ready``. LoRA adapters are addressed by served model name
(``--served-adapter``), the base model by its served name.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT_S = 300.0


class VllmPolicy:
    """PolicyModel-compatible client for a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8300",
        model_name: str = "student",
        seed: Optional[int] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_parallel: int = 32,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.seed = seed
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)
        self._pool = ThreadPoolExecutor(max_workers=max_parallel)
        self.last_stats: List[Dict[str, Any]] = []

    # -- internal ----------------------------------------------------------
    def _one(self, messages: List[Dict[str, str]], max_new_tokens: int,
             temperature: float) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature if temperature and temperature > 0 else 0.0,
        }
        if temperature and temperature > 0:
            body["top_p"] = 0.95
            if self.seed is not None:
                body["seed"] = self.seed
        t0 = time.time()
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                r = self._client.post(f"{self.server_url}/v1/chat/completions", json=body)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                return {
                    "text": (data["choices"][0]["message"]["content"] or "").strip(),
                    "stat": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "gen_s": round(time.time() - t0, 3),
                        "batch_size": 1,
                        "backend": "vllm",
                    },
                }
            except Exception as exc:  # noqa: BLE001 -- transient server hiccups
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"vLLM request failed after 3 attempts: {last_exc}")

    # -- PolicyModel interface ---------------------------------------------
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 700,
                 temperature: float = 0.0) -> str:
        res = self._one(messages, max_new_tokens, temperature)
        self.last_stats = [res["stat"]]
        return res["text"]

    def generate_with_stats(self, messages: List[Dict[str, str]], max_new_tokens: int = 700,
                            temperature: float = 0.0):
        """Race-free variant for concurrent callers: returns (text, stat) directly
        instead of publishing to the shared ``last_stats``."""
        res = self._one(messages, max_new_tokens, temperature)
        return res["text"], res["stat"]

    def generate_batch(self, messages_list: List[List[Dict[str, str]]],
                       max_new_tokens: int = 700, temperature: float = 0.0) -> List[str]:
        """Concurrent requests; the server's continuous batching does the rest."""
        if not messages_list:
            return []
        futures = [self._pool.submit(self._one, m, max_new_tokens, temperature)
                   for m in messages_list]
        results = [f.result() for f in futures]
        self.last_stats = [r["stat"] for r in results]
        return [r["text"] for r in results]


def wait_ready(server_url: str, model_name: Optional[str] = None,
               timeout_s: float = 600.0, interval_s: float = 3.0) -> List[str]:
    """Block until the server answers /v1/models (and serves model_name, if given).
    Returns the served model names."""
    url = server_url.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout_s
    last_err = "no response"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=10)
            if r.status_code == 200:
                names = [m["id"] for m in r.json().get("data", [])]
                if model_name is None or model_name in names:
                    return names
                last_err = f"model {model_name!r} not in {names}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(interval_s)
    raise TimeoutError(f"vLLM server at {server_url} not ready: {last_err}")
