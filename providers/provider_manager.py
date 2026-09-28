from typing import List, Optional
import time
from config.settings import get_settings
from providers.base_provider import BaseProvider
from providers.gemini_provider import GeminiProvider
from providers.groq_provider import GroqProvider

class ProviderManager:
    def __init__(self):
        self.settings = get_settings()
        self.providers: dict[str, BaseProvider] = {}
        
        self.primary_name = self.settings.PRIMARY_LLM_PROVIDER
        self.fallback_names = [p.strip() for p in self.settings.FALLBACK_LLM_PROVIDERS.split(",") if p.strip()]
        
        self._initialize_providers()

    def _initialize_providers(self):
        if "gemini" in [self.primary_name] + self.fallback_names:
            self.providers["gemini"] = GeminiProvider()
        if "groq" in [self.primary_name] + self.fallback_names:
            self.providers["groq"] = GroqProvider()
            
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        return self.providers.get(name)

    def get_primary(self) -> Optional[BaseProvider]:
        return self.providers.get(self.primary_name)

    def get_fallbacks(self) -> List[BaseProvider]:
        return [self.providers[name] for name in self.fallback_names if name in self.providers]

    def record_success(self, name: str, tokens: int = 0):
        if name in self.providers:
            provider = self.providers[name]
            provider.state.status = "AVAILABLE"
            provider.state.last_success = time.time()
            provider.state.success_count += 1
            provider.state.failure_count = 0
            provider.state.total_requests += 1
            provider.state.total_tokens += tokens

    def record_failure(self, name: str, error_msg: str, status: str = "DEGRADED", cooldown_seconds: int = 0):
        if name in self.providers:
            provider = self.providers[name]
            provider.state.status = status
            provider.state.last_error = error_msg
            provider.state.failure_count += 1
            provider.state.total_requests += 1
            if cooldown_seconds > 0:
                provider.state.cooldown_until = time.time() + cooldown_seconds

    def is_provider_available(self, name: str) -> bool:
        if name not in self.providers:
            return False
        provider = self.providers[name]
        if provider.state.status == "CONFIGURATION_ERROR":
            return False
        if time.time() < provider.state.cooldown_until:
            return False
        return True
