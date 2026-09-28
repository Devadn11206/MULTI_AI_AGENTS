from typing import Optional, Type, List, Dict
from pydantic import BaseModel
import time

from config.settings import get_settings
from core.llm_manager import LLMManager
from core.llm.models import LLMResponse
from core.llm.errors import (
    AuthenticationError, RateLimitError, QuotaExceededError,
    ProviderUnavailableError, InvalidRequestError, TimeoutError, LLMError
)

class ProviderHealth:
    def __init__(self, provider_name: str):
        self.provider = provider_name
        self.available = True
        self.last_error = None
        self.consecutive_failures = 0
        self.last_success_time = None

class LLMManager:
    def __init__(self):
        self.settings = get_settings()
        self.primary_name = self.settings.PRIMARY_LLM_PROVIDER.lower()
        
        fallback_str = self.settings.FALLBACK_LLM_PROVIDERS or ""
        self.fallback_names = [name.strip().lower() for name in fallback_str.split(",") if name.strip()]
        
        self.max_retries = self.settings.LLM_MAX_RETRIES
        self.providers: Dict[str, LLMManager] = {}
        self.health: Dict[str, ProviderHealth] = {}
        
        self._init_provider(self.primary_name)
        for name in self.fallback_names:
            self._init_provider(name)
            
    def _init_provider(self, name: str):
        if name in self.providers:
            return
        try:
            if name == "gemini":
                from core.llm.gemini import GeminiProvider
                self.providers[name] = GeminiProvider()
            elif name == "openai":
                from core.llm.openai_provider import OpenAIProvider
                self.providers[name] = OpenAIProvider()
            elif name == "groq":
                from core.llm.groq_provider import GroqProvider
                self.providers[name] = GroqProvider()
            else:
                print(f"[LLM] Warning: Unknown provider {name}")
                return
            self.health[name] = ProviderHealth(name)
        except Exception as e:
            print(f"[LLM] Warning: Could not initialize provider {name}: {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None, provider: Optional[str] = None) -> LLMResponse:
        names_to_try = []
        if provider:
            names_to_try = [provider.lower()]
        else:
            names_to_try = [self.primary_name] + self.fallback_names

        last_error = None
        
        for name in names_to_try:
            if name not in self.providers:
                continue
                
            prov = self.providers[name]
            h = self.health[name]
            
            if not h.available:
                print(f"[LLM] Skipping {name}, marked as unavailable.")
                continue

            retries = 0
            while retries <= self.max_retries:
                try:
                    print(f"[LLM] Provider: {name}")
                    print(f"[LLM] Request started")
                    resp = prov.generate(prompt=prompt, system_prompt=system_prompt, response_schema=response_schema)
                    
                    tokens_str = ""
                    if resp.usage:
                        tokens_str = f" | Tokens: {resp.usage.get('input_tokens')} -> {resp.usage.get('output_tokens')}"
                    print(f"[LLM] {name} request completed{tokens_str}")
                    
                    h.available = True
                    h.consecutive_failures = 0
                    h.last_success_time = time.time()
                    return resp
                    
                except AuthenticationError as e:
                    print(f"[LLM] {name} failed: Auth error - {e}")
                    h.available = False
                    raise e
                    
                except InvalidRequestError as e:
                    print(f"[LLM] {name} failed: Invalid request - {e}")
                    raise e
                    
                except (RateLimitError, QuotaExceededError, ProviderUnavailableError, TimeoutError) as e:
                    retries += 1
                    last_error = e
                    print(f"[LLM] {name} failed: {type(e).__name__} - {e}")
                    h.consecutive_failures += 1
                    if retries <= self.max_retries:
                        print(f"[LLM] Retrying {name} ({retries}/{self.max_retries})...")
                    else:
                        print(f"[LLM] Falling back to next provider...")
                        
                except Exception as e:
                    retries += 1
                    last_error = e
                    print(f"[LLM] {name} unexpected failure: {e}")
                    h.consecutive_failures += 1
                    if retries > self.max_retries:
                        print(f"[LLM] Falling back to next provider...")
                    
        raise last_error or LLMError("All providers failed or none configured.")
