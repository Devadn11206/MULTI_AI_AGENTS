import json
from typing import Optional, Type
from pydantic import BaseModel
import groq

from config.settings import get_settings
from core.llm_manager import LLMManager
from core.llm.models import LLMResponse
from core.llm.errors import (
    AuthenticationError, RateLimitError, QuotaExceededError,
    ProviderUnavailableError, InvalidRequestError, TimeoutError, LLMError
)

class GroqProvider(LLMManager):
    def __init__(self):
        settings = get_settings()
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise AuthenticationError("GROQ_API_KEY is not set.")
        self.client = groq.Groq(api_key=api_key)
        self.model_name = settings.GROQ_MODEL or "llama-3.1-8b-instant"
        self.provider_name_val = "groq"

    @property
    def provider_name(self) -> str:
        return self.provider_name_val

    def _map_error(self, e: Exception):
        if isinstance(e, groq.AuthenticationError):
            raise AuthenticationError(f"Groq Auth Error: {str(e)}") from e
        elif isinstance(e, groq.RateLimitError):
            raise RateLimitError(f"Groq Rate Limit: {str(e)}") from e
        elif isinstance(e, (groq.InternalServerError, groq.APIConnectionError)):
            raise ProviderUnavailableError(f"Groq Unavailable: {str(e)}") from e
        elif isinstance(e, groq.BadRequestError):
            raise InvalidRequestError(f"Groq Invalid Request: {str(e)}") from e
        elif isinstance(e, groq.APITimeoutError):
            raise TimeoutError(f"Groq Timeout: {str(e)}") from e
        raise LLMError(f"Groq Unknown Error: {str(e)}") from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            p = prompt
            if response_schema:
                schema_str = json.dumps(response_schema.model_json_schema())
                p += f"\n\nYou must respond in JSON format matching this schema:\n{schema_str}"
            
            messages.append({"role": "user", "content": p})
            
            kwargs = {
                "model": self.model_name,
                "messages": messages
            }
            if response_schema:
                kwargs["response_format"] = {"type": "json_object"}
                
            response = self.client.chat.completions.create(**kwargs)
            
            usage = None
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
            finish_reason = response.choices[0].finish_reason if response.choices else None
            content = response.choices[0].message.content if response.choices else ""
            
            return LLMResponse(
                content=content or "",
                provider=self.provider_name_val,
                model=self.model_name,
                usage=usage,
                finish_reason=finish_reason
            )
        except Exception as e:
            self._map_error(e)
