import os
import pytest
from config.settings import Settings

def test_settings_default(monkeypatch):
    # Mock empty environment
    monkeypatch.delenv("PRIMARY_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MAX_AGENT_CALLS", raising=False)
    
    # Using _env_file=None to bypass .env file if it exists during test
    settings = Settings(_env_file=None) 
    
    assert settings.PRIMARY_LLM_PROVIDER == "gemini"
    assert settings.MAX_AGENT_CALLS == 20
    assert settings.MAX_AGENT_RETRIES == 3
    assert settings.MAX_WORKFLOW_ITERATIONS == 3
    assert settings.MAX_TOKENS == 4096
    assert settings.GEMINI_API_KEY is None
    assert settings.OPENAI_API_KEY is None

def test_settings_override(monkeypatch):
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MAX_AGENT_CALLS", "50")
    
    settings = Settings(_env_file=None)
    
    assert settings.PRIMARY_LLM_PROVIDER == "openai"
    assert settings.MAX_AGENT_CALLS == 50
