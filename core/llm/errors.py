class LLMError(Exception):
    """Base class for all LLM-related errors."""
    pass

class AuthenticationError(LLMError):
    pass

class RateLimitError(LLMError):
    pass

class QuotaExceededError(LLMError):
    pass

class ProviderUnavailableError(LLMError):
    pass

class InvalidRequestError(LLMError):
    pass

class TimeoutError(LLMError):
    pass

class UnknownProviderError(LLMError):
    pass
