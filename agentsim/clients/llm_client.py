"""Unified LLM client supporting multiple providers"""

import asyncio
import random

import httpx
from typing import Optional, Dict, Any
from loguru import logger
from sentence_transformers import SentenceTransformer

from agentsim.config import config

# HTTP statuses worth retrying: 429 (rate limited / throttled) and 503 (service
# temporarily unavailable) -- both transient on a shared academic gateway.
_RETRYABLE_STATUS = {429, 503}
_BACKOFF_BASE_S = 1.0      # exponential backoff base: base * 2**attempt
_BACKOFF_MAX_S = 30.0      # cap for computed exponential backoff
_BACKOFF_JITTER_S = 0.5    # added uniform(0, jitter) to avoid thundering herd
_RETRY_AFTER_MAX_S = 120.0  # honor an explicit Retry-After header up to this cap


class LLMClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self, default_model: Optional[str] = None):
        self.timeout = config.LLM_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES
        self.default_model = default_model or config.TEACHER_MODELS[0] if config.TEACHER_MODELS else "gpt-4o"
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_model_name: Optional[str] = None
        # Providers that hard-timed-out this process; skipped by the fallback router for
        # the rest of the run so one dead endpoint doesn't cost a full timeout on every
        # subsequent call. Cleared only by restarting the process.
        self._tripped_providers: set[str] = set()
    
    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        return_usage: bool = False,
        return_raw: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> str | Dict[str, Any]:
        """Get completion from any LLM provider

        Args:
            prompt: The prompt text
            model: Model ID (uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            return_usage: If True, returns dict with 'text' and 'usage' keys
            return_raw: If True, also includes the full, unparsed provider response
                body under 'raw_response' (currently only 'custom' and 'ollama'
                support this -- the only two providers this project configures).
            response_schema: A JSON Schema dict (e.g. from a Pydantic model's
                ``.model_json_schema()``) requesting constrained/structured output.
                Ollama grammar-constrains generation to the schema (physically
                prevents invalid JSON or an out-of-vocabulary enum value). OpenRouter/
                custom support for full schema enforcement varies by model, so this
                only requests the broadly-supported looser "valid JSON syntax"
                (response_format: json_object) mode there, not schema conformance.
                Only 'custom' and 'ollama' support this (same two providers as
                return_raw).

        Returns:
            str if return_usage=False and return_raw=False (default), else dict with
            {'text': str, 'usage': dict, ['raw_response': dict]}
        """

        # Use default model if not specified
        model = model or self.default_model

        provider = config.get_provider_from_model_id(model)

        if return_raw and provider not in ("custom", "ollama", "fau", "edenai", "vllm"):
            raise ValueError(f"return_raw is not supported for provider '{provider}'")
        if response_schema and provider not in ("custom", "ollama", "fau", "edenai", "vllm"):
            raise ValueError(f"response_schema is not supported for provider '{provider}'")

        if provider == "openai":
            result = await self._openai_completion(prompt, model, temperature, max_tokens, return_usage)
        elif provider == "anthropic":
            result = await self._anthropic_completion(prompt, model, temperature, max_tokens, return_usage)
        elif provider == "google":
            result = await self._google_completion(prompt, model, temperature, max_tokens, return_usage)
        elif provider == "mistral":
            result = await self._mistral_completion(prompt, model, temperature, max_tokens, return_usage)
        elif provider == "custom":
            result = await self._custom_completion(
                prompt, model, temperature, max_tokens, return_usage or return_raw, return_raw, response_schema
            )
        elif provider == "fau":
            result = await self._fau_completion(
                prompt, model, temperature, max_tokens, return_usage or return_raw, return_raw,
                response_schema, max_retries,
            )
        elif provider == "edenai":
            result = await self._edenai_completion(
                prompt, model, temperature, max_tokens, return_usage or return_raw, return_raw,
                response_schema, max_retries,
            )
        elif provider == "vllm":
            result = await self._vllm_completion(
                prompt, model, temperature, max_tokens, return_usage or return_raw, return_raw, response_schema
            )
        elif provider == "ollama":
            result = await self._ollama_completion(
                prompt, model, temperature, max_tokens, return_usage or return_raw, return_raw, response_schema
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return result
    
    async def _openai_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False) -> str | Dict[str, Any]:
        """OpenAI API completion"""
        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens or config.LLM_MAX_TOKENS
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            
            if return_usage:
                usage = data.get("usage", {})
                return {
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                }
            return text
    
    async def _anthropic_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False) -> str | Dict[str, Any]:
        """Anthropic Claude API completion"""
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens or config.LLM_MAX_TOKENS
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            
            if return_usage:
                usage = data.get("usage", {})
                return {
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    }
                }
            return text
    
    async def _google_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False) -> str | Dict[str, Any]:
        """Google Gemini API completion"""
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured")
        
        # Remove gemini- prefix if present for API
        model_name = model.replace("gemini-", "")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens or config.LLM_MAX_TOKENS
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            if return_usage:
                # Google API includes usageMetadata
                metadata = data.get("usageMetadata", {})
                return {
                    "text": text,
                    "usage": {
                        "prompt_tokens": metadata.get("promptTokenCount", 0),
                        "completion_tokens": metadata.get("candidatesTokenCount", 0),
                        "total_tokens": metadata.get("totalTokenCount", 0)
                    }
                }
            return text
    
    async def _mistral_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False) -> str | Dict[str, Any]:
        """Mistral API completion (OpenAI-compatible)"""
        api_key = config.MISTRAL_API_KEY
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not configured")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens or config.LLM_MAX_TOKENS
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            
            if return_usage:
                usage = data.get("usage", {})
                return {
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                }
            return text
    
    async def _custom_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None) -> str | Dict[str, Any]:
        """Custom endpoint completion (OpenAI-compatible)"""
        endpoint = config.CUSTOM_LLM_ENDPOINT
        api_key = config.CUSTOM_LLM_API_KEY

        if not endpoint:
            raise ValueError("CUSTOM_LLM_ENDPOINT not configured")

        # Remove custom/ prefix
        model_name = model.replace("custom/", "")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
            # OpenRouter-specific extension: asks the response to include real
            # per-call USD cost in 'usage.cost'. This project's CUSTOM_LLM_ENDPOINT
            # is always OpenRouter, so this is safe to send unconditionally.
            "usage": {"include": True},
        }
        if response_schema:
            # Full JSON-Schema enforcement support varies by model on OpenRouter, so
            # we only request the broadly-supported looser "valid JSON syntax"
            # guarantee here rather than relying on schema conformance being honored.
            payload["response_format"] = {"type": "json_object"}

        async def _send() -> Any:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{endpoint.rstrip('/')}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        # Hard wall-clock cap, same rationale as _fau_completion: httpx's read timeout
        # only measures the gap between bytes, so a half-dead connection that trickles
        # keepalive bytes hangs forever (observed in production: an OpenRouter blip left
        # six workers frozen mid-call for 10+ minutes with the API healthy again).
        # asyncio.wait_for turns that into an error the caller's retry/router can handle.
        try:
            data = await asyncio.wait_for(_send(), timeout=config.CUSTOM_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"custom/OpenRouter request exceeded hard timeout of "
                f"{config.CUSTOM_TIMEOUT}s (model={model_name})"
            ) from exc
        text = data["choices"][0]["message"]["content"]

        if return_usage:
            usage = data.get("usage", {})
            result = {
                "text": text,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost": usage.get("cost"),
                }
            }
            if return_raw:
                result["raw_response"] = data
            return result
        return text

    async def _post_json_with_backoff(self, client, url, headers, payload, provider_label="fau", max_retries=None):
        """POST ``payload`` and retry on transient throttling (HTTP 429) or
        unavailability (503) with exponential backoff + jitter, honoring a server
        ``Retry-After`` header when present. Retries up to ``max_retries`` (default
        ``self.max_retries``) times, then returns the final response so the caller can
        ``raise_for_status()`` (i.e. a persistent 429 still surfaces as a clear error
        rather than hanging forever). Pass ``max_retries=0`` to fail fast -- used by the
        provider-fallback router so a rate-limited FAU call falls through immediately
        instead of backing off.
        """
        retries = self.max_retries if max_retries is None else max_retries
        attempt = 0
        while True:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code not in _RETRYABLE_STATUS or attempt >= retries:
                return response

            retry_after = response.headers.get("Retry-After") if response.headers else None
            delay = None
            if retry_after is not None:
                try:  # Retry-After is usually an integer number of seconds.
                    delay = min(float(retry_after), _RETRY_AFTER_MAX_S)
                except (TypeError, ValueError):
                    delay = None  # HTTP-date form or garbage -> fall back to backoff
            if delay is None:
                delay = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_MAX_S)
            delay += random.uniform(0, _BACKOFF_JITTER_S)

            logger.warning(
                f"[{provider_label}] HTTP {response.status_code} (attempt {attempt + 1}/"
                f"{retries}); backing off {delay:.1f}s before retry"
            )
            await asyncio.sleep(delay)
            attempt += 1

    async def get_completion_with_fallback(self, models, *, prompt: str, **kwargs):
        """Try each model in ``models`` in order, returning ``(result, used_model)`` from
        the first that succeeds.

        Robustness contract (checked per call, so every step re-evaluates availability):

        * **Skip unconfigured providers up front.** A model whose provider is missing its
          key/endpoint (``config.provider_available``) is never attempted -- this both
          saves a guaranteed-failing round-trip and avoids the hard crash a provider like
          FAU raises (a plain ``ValueError`` for a missing key, which is not an HTTP
          error). The remaining configured models are the effective chain.
        * **Fall through on ANY failure, not just HTTP errors.** A rate-limit (429),
          connection error, timeout, malformed response, or unexpected exception from one
          model falls through to the next. Only a truly empty/unconfigured chain, or the
          last configured model failing, surfaces an error.
        * Every model but the last configured one is called fail-fast (``max_retries=0``)
          so falling through is quick; the last keeps normal retry behaviour as the final
          fallback.

        Used to route teacher calls FAU -> OpenRouter-free -> OpenRouter-paid: FAU (free
        academic gateway) first, then OpenRouter's free tier, then the paid model last for
        reliability.
        """
        if not models:
            raise ValueError("get_completion_with_fallback requires at least one model")

        available = []
        for m in models:
            if not config.provider_available(m):
                logger.info(
                    f"[router] skipping '{m}' -- provider not configured "
                    "(missing API key/endpoint)"
                )
            elif config.get_provider_from_model_id(m) in self._tripped_providers:
                logger.info(
                    f"[router] skipping '{m}' -- provider tripped earlier this run "
                    "(hard timeout); will not retry until restart"
                )
            else:
                available.append(m)
        if not available:
            raise ValueError(
                f"[router] no configured provider among {list(models)}; set the relevant "
                "API key/endpoint in .env (e.g. FAU_LLM_API_KEY / CUSTOM_LLM_API_KEY)"
            )

        last_exc: Optional[BaseException] = None
        for i, model in enumerate(available):
            is_last = i == len(available) - 1
            call_kwargs = dict(kwargs)
            if not is_last:
                call_kwargs["max_retries"] = 0  # fail fast, fall through on any error
            try:
                result = await self.get_completion(prompt=prompt, model=model, **call_kwargs)
                return result, model
            except Exception as exc:  # noqa: BLE001 -- resilience: fall through on anything
                last_exc = exc
                # A hard timeout means the endpoint is hung, not merely throttled: trip its
                # provider so later calls this run skip it immediately instead of eating the
                # full timeout every time.
                if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                    self._tripped_providers.add(config.get_provider_from_model_id(model))
                logger.warning(
                    f"[router] teacher model '{model}' failed "
                    f"({type(exc).__name__}: {str(exc)[:150]}); "
                    f"{'falling through to next' if not is_last else 'no fallback left'}"
                )
        raise last_exc  # type: ignore[misc]

    async def _fau_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None, max_retries: Optional[int] = None) -> str | Dict[str, Any]:
        """NHR@FAU "LLMs as a Service" gateway completion (OpenAI-compatible).

        Differs from ``_custom_completion`` (OpenRouter) in three ways:
        * the configured base URL already ends in ``/v1``, so we append only
          ``/chat/completions`` (not ``/v1/chat/completions``);
        * it sends no OpenRouter-specific ``usage: {include: true}`` extension, and the
          gateway returns no per-call USD cost (academic service), so ``usage.cost`` is
          ``None``;
        * ``response_schema`` is deliberately NOT turned into a ``response_format:
          json_object`` request. The gateway's JSON mode empirically corrupts the
          output of the reasoning model gpt-oss-120b (e.g. ``{"score": 0.{ ...``),
          whereas leaving it off yields clean JSON. Our prompts already demand
          "Return ONLY a JSON object" and the json_utils repair layer covers the rest,
          so the schema is accepted for API symmetry but not forwarded.

        gpt-oss-120b is a reasoning model: it emits chain-of-thought in
        ``message.reasoning_content`` (which counts against ``max_tokens``) and puts the
        final answer in ``message.content`` -- give it a generous ``max_tokens`` or the
        answer is truncated. If the answer is truncated to empty, ``content`` may be
        ``None``; we coerce to "" so the caller's parse/repair path handles it.
        """
        endpoint = config.FAU_LLM_ENDPOINT
        api_key = config.FAU_LLM_API_KEY

        if not endpoint:
            raise ValueError("FAU_LLM_ENDPOINT not configured")
        if not api_key:
            raise ValueError("FAU_LLM_API_KEY not configured")

        # Remove fau/ prefix
        model_name = model.replace("fau/", "", 1)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        }

        async def _send() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await self._post_json_with_backoff(
                    client, f"{endpoint.rstrip('/')}/chat/completions", headers, payload,
                    provider_label="fau", max_retries=max_retries,
                )
                response.raise_for_status()
                return response.json()

        # Hard wall-clock cap: httpx's read timeout only measures the gap between bytes,
        # so a gateway that keeps the connection alive without ever finishing the response
        # would hang indefinitely. asyncio.wait_for forces a TimeoutError, which the
        # router treats like any other failure and falls through to the next provider.
        try:
            data = await asyncio.wait_for(_send(), timeout=config.FAU_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"FAU request exceeded hard timeout of {config.FAU_TIMEOUT}s "
                f"(model={model_name}); falling through"
            ) from exc
        text = data["choices"][0]["message"].get("content") or ""

        if return_usage:
            usage = data.get("usage", {})
            result = {
                "text": text,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    # Academic gateway: no per-call billing is returned.
                    "cost": usage.get("cost"),
                },
            }
            if return_raw:
                result["raw_response"] = data
            return result
        return text

    async def _edenai_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None, max_retries: Optional[int] = None) -> str | Dict[str, Any]:
        """EdenAI aggregator completion (OpenAI-compatible ``/v2/llm/chat/completions``).

        The configured base URL already ends in ``/v2/llm``, so we append only
        ``/chat/completions``. The model id is passed through verbatim after stripping the
        ``edenai/`` prefix -- EdenAI addresses models as ``<provider>/<model>`` (e.g.
        ``lilac/minimaxai/minimax-m3``), so the remaining slashes are meaningful and MUST
        NOT be collapsed. Retries transient 429/503 with backoff like FAU; a
        ``max_retries=0`` from the fallback router makes it fail fast so the chain falls
        through quickly.

        Like FAU's gpt-oss-120b, EdenAI's MiniMax models are reasoning models: they emit
        chain-of-thought in ``message.reasoning_content`` (counted against ``max_tokens``)
        and the final answer in ``message.content`` -- give a generous ``max_tokens`` or
        the answer truncates to empty (``content`` may be ``None``; coerced to ""). As with
        FAU, ``response_schema`` is accepted for API symmetry but NOT forwarded as a
        ``response_format`` request, to avoid corrupting a reasoning model's JSON output.
        The gateway returns OpenAI-style ``usage`` plus a per-call ``cost`` (USD).
        """
        endpoint = config.EDENAI_LLM_ENDPOINT
        api_key = config.EDENAI_API_KEY

        if not endpoint:
            raise ValueError("EDENAI_LLM_ENDPOINT not configured")
        if not api_key:
            raise ValueError("EDENAI_API_KEY not configured")

        # Strip only the leading edenai/ marker; keep the provider/model path intact.
        model_name = model.replace("edenai/", "", 1)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        }

        async def _send() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await self._post_json_with_backoff(
                    client, f"{endpoint.rstrip('/')}/chat/completions", headers, payload,
                    provider_label="edenai", max_retries=max_retries,
                )
                response.raise_for_status()
                return response.json()

        # Hard wall-clock cap, same rationale as _fau_completion/_custom_completion: httpx's
        # read timeout only measures the gap between bytes, so a half-dead connection hangs
        # forever. asyncio.wait_for turns that into an error the router falls through on.
        try:
            data = await asyncio.wait_for(_send(), timeout=config.EDENAI_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"EdenAI request exceeded hard timeout of {config.EDENAI_TIMEOUT}s "
                f"(model={model_name}); falling through"
            ) from exc
        text = data["choices"][0]["message"].get("content") or ""

        if return_usage:
            usage = data.get("usage", {}) or {}
            result = {
                "text": text,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    # EdenAI reports real per-call USD cost at the top level of the body.
                    "cost": data.get("cost", usage.get("cost")),
                },
            }
            if return_raw:
                result["raw_response"] = data
            return result
        return text

    async def _vllm_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None) -> str | Dict[str, Any]:
        """Local vLLM OpenAI-compatible server completion (student serving).

        The high-throughput alternative to ``_ollama_completion`` for the student: one
        vLLM server per GPU continuously batches many concurrent episodes, so per-episode
        latency barely grows with concurrency. Uses /v1/chat/completions so the server
        applies the model's own chat template (the same contract as Ollama's /api/chat).

        ``response_schema`` is enforced with vLLM's structured output
        (``response_format: json_schema``), which grammar-constrains decoding just like
        Ollama's ``format`` -- the model physically cannot emit invalid JSON. Keeping this
        parity matters: the student templates set ``student_use_response_schema: true``.

        Unlike Ollama, vLLM reports real OpenAI-style token ``usage``, so student token
        counts are no longer zero in the traces.
        """
        endpoint = config.VLLM_ENDPOINT
        if not endpoint:
            raise ValueError("VLLM_ENDPOINT not configured")

        # Remove vllm/ prefix -> the server's --served-model-name (e.g. "student").
        model_name = model.replace("vllm/", "", 1)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        }
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema},
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            # A truncated generation can leave content null -> coerce to "" so the
            # caller's parse/repair path handles it.
            text = data["choices"][0]["message"].get("content") or ""

            if return_usage:
                usage = data.get("usage", {}) or {}
                result = {
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        # Local server: no billing.
                        "cost": None,
                    },
                }
                if return_raw:
                    result["raw_response"] = data
                return result
            return text

    async def _ollama_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None) -> str | Dict[str, Any]:
        """Ollama local completion"""
        endpoint = config.OLLAMA_ENDPOINT
        if not endpoint:
            raise ValueError("OLLAMA_ENDPOINT not configured")

        # Remove ollama/ prefix
        model_name = model.replace("ollama/", "")

        # Use /api/chat (not /api/generate) so Ollama applies the model's chat template.
        # /api/generate feeds the raw prompt unwrapped, which produces coherent output
        # only for models whose Modelfile template happens to no-op; a ChatML/instruct
        # model such as MiniCPM5 (pulled from hf.co) then sees an unformatted prompt and
        # emits pure gibberish. /api/chat wraps the message in the model's template, so it
        # works across models (verified: MiniCPM5-1B produced garbage via /api/generate
        # but coherent task output via /api/chat, with qwen unaffected).
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Disable hybrid "thinking" mode (e.g. Qwen3): this framework asks
            # for JSON-only outputs, so reasoning preambles waste tokens and can
            # leave the response empty if num_predict is exhausted while thinking.
            # Ignored by non-thinking models.
            "think": config.OLLAMA_THINK,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or config.LLM_MAX_TOKENS
            }
        }
        if response_schema:
            # Grammar-constrains generation to the schema -- the model is physically
            # unable to emit invalid JSON or an out-of-vocabulary enum value.
            payload["format"] = response_schema

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            # A truncated 'thinking' turn can leave content empty -> coerce to "".
            text = (data.get("message") or {}).get("content") or ""

            if return_usage:
                # Ollama doesn't provide OpenAI-style token counts; report 0s.
                result = {
                    "text": text,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                if return_raw:
                    result["raw_response"] = data
                return result
            return text

    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> list:
        """Get embedding vector for text using local SentenceTransformer."""
        if not text:
            raise ValueError("Text for embedding must not be empty")
        
        embedding_model = await self._ensure_embedding_model(model)
        
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(
            None,
            lambda: embedding_model.encode(
                text,
                normalize_embeddings=True
            ).tolist()
        )
        return vector
    
    async def _ensure_embedding_model(self, model: Optional[str] = None) -> SentenceTransformer:
        """Load or reuse local embedding model."""
        target_name = model or config.LOCAL_EMBEDDING_MODEL
        if self._embedding_model is None or self._embedding_model_name != target_name:
            logger.info(f"Loading local embedding model: {target_name}")
            loop = asyncio.get_running_loop()
            self._embedding_model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(target_name)
            )
            self._embedding_model_name = target_name
        return self._embedding_model

