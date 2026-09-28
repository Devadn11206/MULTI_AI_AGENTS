from typing import List, Dict, Any, Optional
import time
import json
from config.settings import get_settings
from core.errors import (
    LLMError, ConfigurationError, QuotaExceededError,
    AgentCallLimitExceeded
)
from providers.provider_manager import ProviderManager
from providers.base_provider import LLMResponse
from core.retry_manager import RetryManager
from core.limit_manager import LimitManager

class APIManager:
    def __init__(self, provider_manager: ProviderManager, retry_manager: RetryManager):
        self.provider_manager = provider_manager
        self.retry_manager = retry_manager

    def call_provider(self, provider_name: str, messages: List[Dict[str, Any]], max_tokens: Optional[int], temperature: Optional[float], response_schema: Optional[Any] = None) -> LLMResponse:
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ConfigurationError(f"Provider {provider_name} not configured.")
            
        def _execute():
            return provider.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_schema=response_schema
            )
            
        try:
            res = self.retry_manager.execute_with_retry(_execute)
            self.provider_manager.record_success(provider_name, res.total_tokens or 0)
            return res
        except Exception as e:
            self.provider_manager.record_failure(provider_name, str(e), cooldown_seconds=get_settings().PROVIDER_COOLDOWN_SECONDS)
            raise e

    def execute_with_fallback(self, messages: List[Dict[str, Any]], max_tokens: Optional[int], temperature: Optional[float], response_schema: Optional[Any] = None, agent_name: str = "unknown", trace_manager=None) -> LLMResponse:
        # Try primary
        primary = self.provider_manager.primary_name
        last_error = None
        
        if self.provider_manager.is_provider_available(primary):
            try:
                req_ev_id = None
                if trace_manager:
                    from config.settings import get_settings
                    req_ev_id = trace_manager.record_llm_request(
                        agent_name, primary, get_settings().GEMINI_MODEL if primary == "gemini" else get_settings().GROQ_MODEL
                    )
                res = self.call_provider(primary, messages, max_tokens, temperature, response_schema)
                if trace_manager:
                    trace_manager.record_llm_response(
                        agent_name, primary,
                        get_settings().GEMINI_MODEL if primary == "gemini" else get_settings().GROQ_MODEL,
                        parent_event_id=req_ev_id,
                        metadata={"total_tokens": res.total_tokens},
                    )
                return res
            except Exception as e:
                last_error = e
                if trace_manager:
                    from config.settings import get_settings
                    trace_manager.record_llm_retry(
                        agent_name, primary, attempt=1, reason=str(e)[:200]
                    )
                print(f"[APIManager] Primary provider {primary} failed: {e}. Falling back...")
        
        # Try fallbacks
        for fallback in self.provider_manager.fallback_names:
            if self.provider_manager.is_provider_available(fallback):
                try:
                    if trace_manager:
                        trace_manager.record_llm_fallback(agent_name, primary, fallback)
                        req_ev_id = trace_manager.record_llm_request(
                            agent_name, fallback, get_settings().GROQ_MODEL if fallback == "groq" else fallback
                        )
                    res = self.call_provider(fallback, messages, max_tokens, temperature, response_schema)
                    res.fallback_used = True
                    if trace_manager:
                        trace_manager.record_llm_response(
                            agent_name, fallback,
                            get_settings().GROQ_MODEL if fallback == "groq" else fallback,
                            parent_event_id=req_ev_id,
                        )
                    return res
                except Exception as e:
                    last_error = e
                    print(f"[APIManager] Fallback provider {fallback} failed: {e}.")
        
        if last_error:
            raise last_error
        raise LLMError("No LLM provider is available or configured.")



class LLMManager:
    def __init__(self):
        self.settings = get_settings()
        self.provider_manager = ProviderManager()
        self.retry_manager = RetryManager()
        self.limit_manager = LimitManager()
        self.api_manager = APIManager(self.provider_manager, self.retry_manager)
        
        self.usage_stats = {
            "gemini": {"calls": 0, "successes": 0, "failures": 0},
            "groq": {"calls": 0, "successes": 0, "failures": 0},
            "fallbacks": 0
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Any] = None, agent_name: str = "unknown", max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> LLMResponse:
        # Check Agent Budget
        self.limit_manager.check_and_increment(agent_name)
        
        # Enforce Max Tokens Context
        max_t = max_tokens or self.settings.MAX_TOKENS
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            res = self.api_manager.execute_with_fallback(
                messages=messages,
                max_tokens=max_t,
                temperature=temperature,
                response_schema=response_schema
            )
            
            # Track Usage
            prov = res.provider
            if prov in self.usage_stats:
                self.usage_stats[prov]["calls"] += 1
                self.usage_stats[prov]["successes"] += 1
            if res.fallback_used:
                self.usage_stats["fallbacks"] += 1
                
            return res
            
        except Exception as e:
            # Note: We can't easily know which provider ultimately failed if both failed,
            # but we can just record a generic failure in tracking.
            if hasattr(e, 'provider') and e.provider in self.usage_stats:
                self.usage_stats[e.provider]["calls"] += 1
                self.usage_stats[e.provider]["failures"] += 1
            raise e
