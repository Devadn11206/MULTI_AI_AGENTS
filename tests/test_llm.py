import pytest
from core.llm_manager import LLMManager
from core.llm.models import LLMResponse
from core.llm_manager import LLMManager
from core.llm.errors import AuthenticationError, RateLimitError, LLMError
from config.settings import Settings

class FakeProvider(LLMManager):
    def __init__(self, name="fake", fail_auth=False, fail_rate=False):
        self._name = name
        self.fail_auth = fail_auth
        self.fail_rate = fail_rate

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(self, prompt, system_prompt=None, response_schema=None):
        if self.fail_auth:
            raise AuthenticationError("Fake auth error")
        if self.fail_rate:
            raise RateLimitError("Fake rate limit")
        return LLMResponse(content=f"Fake response to: {prompt}", provider=self._name, model="fake-model")

def mock_get_settings_no_keys():
    return Settings(_env_file=None, GEMINI_API_KEY=None, OPENAI_API_KEY=None, GROQ_API_KEY=None, PRIMARY_LLM_PROVIDER="unknown", FALLBACK_LLM_PROVIDERS="")

def test_manager_unknown_provider(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("core.llm.manager.get_settings", mock_get_settings_no_keys)
    manager = LLMManager()
    assert "unknown" not in manager.providers
    with pytest.raises(LLMError, match="All providers failed or none configured"):
        manager.generate("test")

def test_gemini_provider_initializes(monkeypatch):
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    manager = LLMManager()
    assert "gemini" in manager.providers

def test_fallback_mechanism(monkeypatch):
    manager = LLMManager()
    # Replace providers with fakes
    p1 = FakeProvider(name="gemini", fail_rate=True)
    p2 = FakeProvider(name="openai", fail_rate=False)
    
    manager.primary_name = "gemini"
    manager.fallback_names = ["openai"]
    manager.providers = {"gemini": p1, "openai": p2}
    
    from core.llm.manager import ProviderHealth
    manager.health = {"gemini": ProviderHealth("gemini"), "openai": ProviderHealth("openai")}
    
    resp = manager.generate("test")
    assert resp.provider == "openai"
    assert resp.content == "Fake response to: test"

def test_auth_error_no_fallback(monkeypatch):
    manager = LLMManager()
    p1 = FakeProvider(name="gemini", fail_auth=True)
    p2 = FakeProvider(name="openai", fail_rate=False)
    manager.primary_name = "gemini"
    manager.fallback_names = ["openai"]
    manager.providers = {"gemini": p1, "openai": p2}
    from core.llm.manager import ProviderHealth
    manager.health = {"gemini": ProviderHealth("gemini"), "openai": ProviderHealth("openai")}
    
    with pytest.raises(AuthenticationError):
        manager.generate("test")
