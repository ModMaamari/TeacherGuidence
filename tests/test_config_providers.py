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
