import os
import json
import pytest
import time
from core.llm_manager import APIManager, LLMManager
from providers.provider_manager import ProviderManager
from core.retry_manager import RetryManager
from core.limit_manager import LimitManager
from core.checkpoint_manager import CheckpointManager
from core.state.manager import AgentStateManager
from core.errors import TimeoutError, AgentCallLimitExceeded
from api.services.workflow_service import WorkflowService

class MockFailingProvider:
    def __init__(self, name="gemini", fail_times=1):
        from providers.base_provider import ProviderState
        self.name = name
        self.state = ProviderState(provider_name=name)
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("Simulated timeout")
        from providers.base_provider import LLMResponse
        return LLMResponse(content="Success", provider=self.name, model="mock", request_id="123", latency_ms=10)


def test_retry_manager_success_after_fail():
    pm = ProviderManager()
    rm = RetryManager()
    rm.base_backoff = 0.1
    rm.max_retries = 2
    
    mock_prov = MockFailingProvider(fail_times=1)
    
    def _exec():
        return mock_prov.generate()
        
    res = rm.execute_with_retry(_exec)
    assert res.content == "Success"
    assert mock_prov.calls == 2

def test_api_manager_fallback():
    pm = ProviderManager()
    # Replace real providers
    pm.providers["gemini"] = MockFailingProvider("gemini", fail_times=5) # Will exhaust retries
    pm.providers["groq"] = MockFailingProvider("groq", fail_times=0) # Will succeed immediately
    
    rm = RetryManager()
    rm.base_backoff = 0.01
    
    api_manager = APIManager(pm, rm)
    
    res = api_manager.execute_with_fallback([{"role": "user", "content": "hi"}], None, None)
    
    assert res.provider == "groq"
    assert res.fallback_used == True
    assert pm.providers["gemini"].state.failure_count > 0

def test_checkpoint_manager(tmpdir):
    cm = CheckpointManager()
    cm.checkpoints_dir = str(tmpdir)
    
    cm.save_checkpoint("wf_test_123", "data_analyst", "data_analyst", {"status": "good"})
    
    cp = cm.load_latest_checkpoint("wf_test_123")
    assert cp is not None
    assert cp["workflow_id"] == "wf_test_123"
    assert cp["phase"] == "data_analyst"

def test_limit_manager():
    lm = LimitManager()
    lm.max_total_calls = 3
    lm.max_agent_calls = 2
    
    lm.check_and_increment("test_agent")
    lm.check_and_increment("test_agent")
    
    with pytest.raises(AgentCallLimitExceeded):
        lm.check_and_increment("test_agent")
        
    lm.check_and_increment("other_agent")
    
    with pytest.raises(AgentCallLimitExceeded):
        lm.check_and_increment("other_agent")
