class LLMError(Exception):
    """Base class for all LLM errors."""
    def __init__(self, message: str, provider: str = "unknown"):
        self.message = message
        self.provider = provider
        super().__init__(f"[{provider}] {message}")

class ProviderError(LLMError): pass
class RateLimitError(LLMError): pass
class TimeoutError(LLMError): pass
class AuthenticationError(LLMError): pass
class NetworkError(LLMError): pass
class InvalidResponseError(LLMError): pass
class TokenLimitError(LLMError): pass
class QuotaExceededError(LLMError): pass
class ServiceUnavailableError(LLMError): pass
class ConfigurationError(LLMError): pass
class AgentCallLimitExceeded(LLMError): pass
