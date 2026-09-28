import pytest
from core.llm_manager import LLMManager
from providers.provider_manager import ProviderManager
from providers.base_provider import LLMResponse, ProviderState

class DummyProvider:
    def __init__(self, name="dummy", model="dummy_model"):
        self.name = name
        self.model = model
        self.state = ProviderState(provider_name=name)

    def generate(self, messages, **kwargs):
        return LLMResponse(content="dummy response", provider=self.name, model=self.model, usage={"input_tokens":10, "output_tokens":10})

def test_provider_manager_registration():
    pm = ProviderManager()
    dummy = DummyProvider()
    pm.register_provider("dummy", dummy, is_primary=True)
    assert pm.primary_name == "dummy"
    assert pm.get_provider("dummy") == dummy

def test_llm_response_parsing():
    res = LLMResponse(content='{"key": "value"}', provider="test", model="test")
    assert res.content == '{"key": "value"}'
