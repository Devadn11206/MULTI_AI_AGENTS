import time
import json
from typing import List, Dict, Any, Optional
import groq

from config.settings import get_settings
from core.errors import (
    AuthenticationError, RateLimitError, ServiceUnavailableError,
    TimeoutError, LLMError, ConfigurationError
)
from providers.base_provider import BaseProvider, LLMResponse

class GroqProvider(BaseProvider):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="groq", model=settings.GROQ_MODEL)
        api_key = settings.GROQ_API_KEY
        if not api_key:
            self.state.status = "CONFIGURATION_ERROR"
            self.state.last_error = "GROQ_API_KEY is not set."
            self.client = None
        else:
            self.client = groq.Groq(api_key=api_key)

    def _map_error(self, e: Exception):
        if isinstance(e, groq.AuthenticationError):
            raise AuthenticationError(f"Groq Auth Error: {str(e)}", provider="groq") from e
        elif isinstance(e, groq.RateLimitError):
            raise RateLimitError(f"Groq Rate Limit: {str(e)}", provider="groq") from e
        elif isinstance(e, (groq.InternalServerError, groq.APIConnectionError)):
            raise ServiceUnavailableError(f"Groq Unavailable: {str(e)}", provider="groq") from e
        elif isinstance(e, groq.BadRequestError):
            raise LLMError(f"Groq Invalid Request: {str(e)}", provider="groq") from e
        elif isinstance(e, groq.APITimeoutError):
            raise TimeoutError(f"Groq Timeout: {str(e)}", provider="groq") from e
        raise LLMError(f"Groq Unknown Error: {str(e)}", provider="groq") from e

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, response_schema: Optional[Any] = None) -> LLMResponse:
        if not self.client:
            raise ConfigurationError("Groq client is not initialized due to missing API key.", provider="groq")
            
        start_time = time.time()
        try:
            if response_schema:
                schema_str = json.dumps(response_schema.model_json_schema())
                messages.append({"role": "user", "content": f"You must respond in JSON format matching this schema:\n{schema_str}"})
                
            kwargs = {
                "model": self.model,
                "messages": messages
            }
            if max_tokens:
                kwargs["max_tokens"] = min(max_tokens, 2048)
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_schema:
                kwargs["response_format"] = {"type": "json_object"}
                
            response = self.client.chat.completions.create(**kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            input_tokens = None
            output_tokens = None
            total_tokens = None
            if response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', None)
                output_tokens = getattr(response.usage, 'completion_tokens', None)
                total_tokens = getattr(response.usage, 'total_tokens', None)
                
            content = response.choices[0].message.content if response.choices else ""
            
            return LLMResponse(
                content=content or "",
                provider=self.name,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            self._map_error(e)
