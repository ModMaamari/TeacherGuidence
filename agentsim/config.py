"""Configuration management for AgentSim"""

import os
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Config:
    """Global configuration from environment variables"""
    
    # ============================================
    # LLM Provider API Keys
    # ============================================
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")
    
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_PROJECT_ID: Optional[str] = os.getenv("GOOGLE_PROJECT_ID")
    
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    TOGETHER_API_KEY: Optional[str] = os.getenv("TOGETHER_API_KEY")
    
    # Custom endpoints
    CUSTOM_LLM_ENDPOINT: Optional[str] = os.getenv("CUSTOM_LLM_ENDPOINT")
    CUSTOM_LLM_API_KEY: Optional[str] = os.getenv("CUSTOM_LLM_API_KEY")

    # NHR@FAU "LLMs as a Service" gateway (OpenAI-compatible). Distinct from the
    # OpenRouter `custom` provider: its base URL already ends in /v1 (the client appends
    # only /chat/completions), it needs no OpenRouter-specific usage extension, and it
    # serves reasoning models (e.g. gpt-oss-120b). Key name mirrors the FAU docs'
    # LLMAPI_KEY as a fallback so the same .env works. See docs at /root/DeKIS/hpc_fau.
    FAU_LLM_ENDPOINT: str = os.getenv("FAU_LLM_ENDPOINT", "https://hub.nhr.fau.de/api/llmgw/v1")
    FAU_LLM_API_KEY: Optional[str] = os.getenv("FAU_LLM_API_KEY") or os.getenv("LLMAPI_KEY")

    # EdenAI aggregator. NOTE: this is a *Responses*-style API, NOT OpenAI
    # chat-completions -- the request/response shapes are not interchangeable with the FAU
    # gateway (see /root/DeKIS/hpc_fau/edenai_docs.md). The full endpoint URL is configured
    # here (not a base), because the API exposes the single /v3/responses route. Models are
    # addressed as ``edenai/<provider>/<model>`` (e.g. ``edenai/lilac/minimaxai/minimax-m3``,
    # where ``lilac`` is the upstream provider EdenAI routes to); the ``edenai/`` prefix is
    # stripped before the request. Commercial: every call bills real money and returns a
    # ``cost`` field.
    EDENAI_LLM_ENDPOINT: str = os.getenv("EDENAI_LLM_ENDPOINT", "https://api.edenai.run/v3/responses")
    EDENAI_API_KEY: Optional[str] = os.getenv("EDENAI_API_KEY")

    # Local vLLM OpenAI-compatible server (student serving). Used instead of Ollama when a
    # small student needs high-throughput continuous batching on a big GPU: one server per
    # GPU serves many concurrent episodes. Address models as ``vllm/<served-model-name>``;
    # serve the model under its real HF id (``vllm/ibm-granite/granite-4.1-3b``) rather
    # than a placeholder, so traces record which student actually ran.
    # The endpoint has no /v1 suffix (the client appends /v1/chat/completions); each worker
    # process points at its own GPU's server via the VLLM_ENDPOINT env var, exactly as the
    # Ollama path uses OLLAMA_ENDPOINT.
    VLLM_ENDPOINT: str = os.getenv("VLLM_ENDPOINT", "http://127.0.0.1:8300")

    # Ollama
    OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    # Hybrid "thinking" mode for Ollama models (Qwen3, etc.). Off by default because
    # this framework expects JSON-only outputs.
    OLLAMA_THINK: bool = os.getenv("OLLAMA_THINK", "false").lower() == "true"
    
    # ============================================
    # Default Models
    # ============================================
    TEACHER_MODELS: List[str] = [m.strip() for m in os.getenv("TEACHER_MODELS", "gpt-4o").split(",") if m.strip()]
    TEACHER_TEMPERATURE: float = float(os.getenv("TEACHER_TEMPERATURE", "0.7"))
    
    CONSULTANT_MODELS: List[str] = [m.strip() for m in os.getenv("CONSULTANT_MODELS", "").split(",") if m.strip()]
    CONSULTANT_TEMPERATURE: float = float(os.getenv("CONSULTANT_TEMPERATURE", "0.7"))
    
    VERIFIER_MODEL: str = os.getenv("VERIFIER_MODEL", "gpt-4o-mini")
    VERIFIER_TEMPERATURE: float = float(os.getenv("VERIFIER_TEMPERATURE", "0.1"))
    
    # Embeddings
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # ============================================
    # Retrieval
    # ============================================
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "documents")
    OPENSEARCH_USER: Optional[str] = os.getenv("OPENSEARCH_USER")
    OPENSEARCH_PASSWORD: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    OPENSEARCH_USE_SSL: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
    
    VECTOR_SEARCH_ENABLED: bool = os.getenv("VECTOR_SEARCH_ENABLED", "false").lower() == "true"
    VECTOR_SEARCH_ENDPOINT: Optional[str] = os.getenv("VECTOR_SEARCH_ENDPOINT")
    
    # ChatNoir (primary retrieval)
    CHATNOIR_ENABLED: bool = os.getenv("CHATNOIR_ENABLED", "true").lower() == "true"
    CHATNOIR_API_KEY: Optional[str] = os.getenv("CHATNOIR_API_KEY")
    CHATNOIR_BASE_URL: str = os.getenv("CHATNOIR_BASE_URL", "https://www.chatnoir.eu/api/v1")
    CHATNOIR_DEFAULT_CORPUS: str = os.getenv("CHATNOIR_DEFAULT_CORPUS", "cw12")
    
    # ============================================
    # Simulation
    # ============================================
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    SIMILARITY_METRIC: str = os.getenv("SIMILARITY_METRIC", "embedding_cosine")
    
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./data/simulation_output")
    
    # ============================================
    # Advanced LLM Settings
    # ============================================
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    # Hard wall-clock deadline (seconds) for a single FAU gateway request. httpx's
    # timeout is inter-byte only, so a gateway that holds the connection open trickling
    # keepalive bytes (observed with a stalled gpt-oss-120b backend) never trips it and
    # blocks forever. This asyncio.wait_for cap guarantees a hung FAU call raises so the
    # teacher router falls through to OpenRouter instead of stalling the whole run.
    FAU_TIMEOUT: int = int(os.getenv("FAU_TIMEOUT", "45"))
    # Same wall-clock cap for a single custom/OpenRouter request (observed in
    # production: a momentary OpenRouter blip left workers frozen mid-call for 10+
    # minutes with the API healthy again). Generous default because a reasoning
    # teacher's long generation is legitimate.
    CUSTOM_TIMEOUT: int = int(os.getenv("CUSTOM_TIMEOUT", "180"))
    # Same wall-clock cap for a single EdenAI request (a reasoning teacher such as
    # MiniMax-M3 can legitimately generate for a while), matching CUSTOM_TIMEOUT.
    EDENAI_TIMEOUT: int = int(os.getenv("EDENAI_TIMEOUT", "180"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    @classmethod
    def get_provider_from_model_id(cls, model_id: str) -> str:
        """Detect provider from model ID"""
        model_lower = model_id.lower()
        
        # Check for custom/ollama/fau prefixes first
        if model_id.startswith("custom/"):
            return "custom"
        elif model_id.startswith("ollama/"):
            return "ollama"
        elif model_id.startswith("fau/"):
            return "fau"
        elif model_id.startswith("edenai/"):
            return "edenai"
        elif model_id.startswith("vllm/"):
            return "vllm"
        elif any(x in model_lower for x in ["gpt", "openai", "o1", "davinci", "turbo"]):
            return "openai"
        elif any(x in model_lower for x in ["claude", "anthropic"]):
            return "anthropic"
        elif any(x in model_lower for x in ["gemini", "palm", "bison"]):
            return "google"
        elif "mistral" in model_lower:
            return "mistral"
        elif "cohere" in model_lower or "command" in model_lower:
            return "cohere"
        elif "together" in model_lower:
            return "together"
        else:
            # Default to openai for unknown models
            return "openai"
    
    @classmethod
    def provider_available(cls, model_id: str) -> bool:
        """True if the provider backing this model has the credentials/endpoint it needs
        to be callable right now.

        Used by the router (``LLMClient.get_completion_with_fallback``) to *skip* an
        unconfigured provider instead of attempting a call that can only fail -- e.g. a
        ``fau/`` model when ``FAU_LLM_API_KEY`` is unset. Checking here keeps the failure
        cheap (no wasted HTTP round-trip) and, crucially, avoids a hard crash on providers
        (like FAU) that raise a non-HTTP ``ValueError`` for a missing key.
        """
        provider = cls.get_provider_from_model_id(model_id)
        if provider == "fau":
            return bool(cls.FAU_LLM_ENDPOINT and cls.FAU_LLM_API_KEY)
        if provider == "edenai":
            return bool(cls.EDENAI_LLM_ENDPOINT and cls.EDENAI_API_KEY)
        if provider == "custom":
            return bool(cls.CUSTOM_LLM_ENDPOINT and cls.CUSTOM_LLM_API_KEY)
        if provider == "vllm":
            return bool(cls.VLLM_ENDPOINT)
        if provider == "ollama":
            return bool(cls.OLLAMA_ENDPOINT)
        # Hosted providers (openai/anthropic/google/...): callable iff their key is set.
        return bool(cls.get_api_key_for_provider(provider))

    @classmethod
    def get_api_key_for_provider(cls, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        provider_keys: Dict[str, Optional[str]] = {
            "openai": cls.OPENAI_API_KEY,
            "anthropic": cls.ANTHROPIC_API_KEY,
            "google": cls.GOOGLE_API_KEY,
            "mistral": cls.MISTRAL_API_KEY,
            "cohere": cls.COHERE_API_KEY,
            "together": cls.TOGETHER_API_KEY,
            "custom": cls.CUSTOM_LLM_API_KEY,
            "fau": cls.FAU_LLM_API_KEY,
            "edenai": cls.EDENAI_API_KEY,
            "vllm": None,  # local server, no API key
            "ollama": None,  # Ollama doesn't require API key by default
        }
        return provider_keys.get(provider.lower())
    
    @classmethod
    def get_model_api_key(cls, model_id: str) -> Optional[str]:
        """Get API key for a specific model ID"""
        provider = cls.get_provider_from_model_id(model_id)
        return cls.get_api_key_for_provider(provider)
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, any]:
        """Get additional provider-specific configuration"""
        configs = {
            "openai": {
                "api_key": cls.OPENAI_API_KEY,
                "organization": cls.OPENAI_ORG_ID,
            },
            "google": {
                "api_key": cls.GOOGLE_API_KEY,
                "project_id": cls.GOOGLE_PROJECT_ID,
            },
            "custom": {
                "api_key": cls.CUSTOM_LLM_API_KEY,
                "endpoint": cls.CUSTOM_LLM_ENDPOINT,
            },
            "fau": {
                "api_key": cls.FAU_LLM_API_KEY,
                "endpoint": cls.FAU_LLM_ENDPOINT,
            },
            "edenai": {
                "api_key": cls.EDENAI_API_KEY,
                "endpoint": cls.EDENAI_LLM_ENDPOINT,
            },
            "vllm": {
                "endpoint": cls.VLLM_ENDPOINT,
            },
            "ollama": {
                "endpoint": cls.OLLAMA_ENDPOINT,
                "enabled": cls.OLLAMA_ENABLED,
            },
        }
        return configs.get(provider.lower(), {})


config = Config()

