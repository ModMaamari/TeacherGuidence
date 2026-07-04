"""Unified LLM client supporting multiple providers"""

import asyncio

import httpx
from typing import Optional, Dict, Any
from loguru import logger
from sentence_transformers import SentenceTransformer

from agentsim.config import config


class LLMClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self, default_model: Optional[str] = None):
        self.timeout = config.LLM_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES
        self.default_model = default_model or config.TEACHER_MODELS[0] if config.TEACHER_MODELS else "gpt-4o"
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_model_name: Optional[str] = None
    
    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        return_usage: bool = False,
        return_raw: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
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

        if return_raw and provider not in ("custom", "ollama", "fau"):
            raise ValueError(f"return_raw is not supported for provider '{provider}'")
        if response_schema and provider not in ("custom", "ollama", "fau"):
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

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
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

    async def _fau_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None) -> str | Dict[str, Any]:
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

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
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

    async def _ollama_completion(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int], return_usage: bool = False, return_raw: bool = False, response_schema: Optional[Dict[str, Any]] = None) -> str | Dict[str, Any]:
        """Ollama local completion"""
        endpoint = config.OLLAMA_ENDPOINT
        if not endpoint:
            raise ValueError("OLLAMA_ENDPOINT not configured")

        # Remove ollama/ prefix
        model_name = model.replace("ollama/", "")

        payload = {
            "model": model_name,
            "prompt": prompt,
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
                f"{endpoint.rstrip('/')}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            text = data["response"]

            if return_usage:
                # Ollama doesn't provide token counts in standard API, return 0
                result = {
                    "text": text,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                if return_raw:
                    # 'context' is Ollama's raw re-tokenized conversation state (only
                    # useful if fed back into a follow-up call for continuation, which
                    # this project never does): a list of hundreds/thousands of token
                    # IDs that can't be decoded back to text without loading each
                    # model's own tokenizer. Drop it and keep a count instead -- the
                    # decoded text is already in 'response' just above it.
                    raw_response = dict(data)
                    context = raw_response.pop("context", None)
                    if context is not None:
                        raw_response["context_length"] = len(context)
                    result["raw_response"] = raw_response
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

