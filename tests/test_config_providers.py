"""Provider resolution for model-id prefixes, incl. the NHR@FAU gateway."""

from agentsim.config import config


def test_fau_prefix_resolves_to_fau_provider():
    assert config.get_provider_from_model_id("fau/gpt-oss-120b") == "fau"


def test_custom_and_ollama_prefixes_still_resolve():
    assert config.get_provider_from_model_id("custom/z-ai/glm-5.2") == "custom"
    assert config.get_provider_from_model_id("ollama/qwen3.5:4b") == "ollama"


def test_fau_provider_uses_fau_api_key(monkeypatch):
    # get_api_key_for_provider is a classmethod reading cls.FAU_LLM_API_KEY.
    monkeypatch.setattr(type(config), "FAU_LLM_API_KEY", "sk-fau-test")
    assert config.get_api_key_for_provider("fau") == "sk-fau-test"


def test_fau_provider_config_has_endpoint_and_key():
    cfg = config.get_provider_config("fau")
    assert cfg["endpoint"]  # defaults to the FAU gateway URL
    assert "api_key" in cfg


def test_provider_available_reflects_credentials(monkeypatch):
    cls = type(config)
    # FAU: available only when both endpoint and key are set.
    monkeypatch.setattr(cls, "FAU_LLM_ENDPOINT", "https://hub.nhr.fau.de/api/llmgw/v1")
    monkeypatch.setattr(cls, "FAU_LLM_API_KEY", "sk-fau")
    assert config.provider_available("fau/gpt-oss-120b") is True
    monkeypatch.setattr(cls, "FAU_LLM_API_KEY", None)
    assert config.provider_available("fau/gpt-oss-120b") is False

    # custom/OpenRouter: needs endpoint + key.
    monkeypatch.setattr(cls, "CUSTOM_LLM_ENDPOINT", "https://openrouter.ai/api")
    monkeypatch.setattr(cls, "CUSTOM_LLM_API_KEY", "sk-or")
    assert config.provider_available("custom/openai/gpt-oss-120b") is True
    monkeypatch.setattr(cls, "CUSTOM_LLM_API_KEY", None)
    assert config.provider_available("custom/openai/gpt-oss-120b") is False


def test_provider_available_ollama_needs_only_endpoint(monkeypatch):
    monkeypatch.setattr(type(config), "OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    assert config.provider_available("ollama/qwen3.5:0.8b") is True
