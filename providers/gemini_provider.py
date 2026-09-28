import time
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import get_settings
from core.errors import (
    AuthenticationError, RateLimitError, QuotaExceededError,
    ServiceUnavailableError, InvalidResponseError, TimeoutError, LLMError, ConfigurationError
)
from providers.base_provider import BaseProvider, LLMResponse

class GeminiProvider(BaseProvider):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="gemini", model=settings.GEMINI_MODEL)
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            self.state.status = "CONFIGURATION_ERROR"
            self.state.last_error = "GEMINI_API_KEY is not set."
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def _map_error(self, e: Exception):
        if isinstance(e, APIError):
            code = getattr(e, 'code', None)
            msg = getattr(e, 'message', str(e)).lower()
            if code in (401, 403):
                raise AuthenticationError(f"Gemini Auth Error: {msg}", provider="gemini") from e
            elif code == 429:
                if "quota" in msg:
                    raise QuotaExceededError(f"Gemini Quota Exceeded: {msg}", provider="gemini") from e
                raise RateLimitError(f"Gemini Rate Limit: {msg}", provider="gemini") from e
            elif code in (500, 502, 503, 504):
                raise ServiceUnavailableError(f"Gemini Unavailable: {msg}", provider="gemini") from e
            elif code == 400:
                raise LLMError(f"Gemini Invalid Request: {msg}", provider="gemini") from e
        raise LLMError(f"Gemini Unknown Error: {str(e)}", provider="gemini") from e

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, response_schema: Optional[Any] = None) -> LLMResponse:
        if not self.client:
            raise ConfigurationError("Gemini client is not initialized due to missing API key.", provider="gemini")
            
        start_time = time.time()
        try:
            # Build prompt from messages
            prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
            
            config_dict = {}
            if max_tokens:
                config_dict["max_output_tokens"] = max_tokens
            if temperature is not None:
                config_dict["temperature"] = temperature
                
            if response_schema:
                config_dict["response_mime_type"] = "application/json"
                config_dict["response_schema"] = response_schema

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict) if config_dict else None
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            input_tokens = None
            output_tokens = None
            total_tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(response.usage_metadata, 'total_token_count', None)
                
            return LLMResponse(
                content=response.text or "",
                provider=self.name,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms
            )
            
        except AuthenticationError:
            raise
        except RateLimitError:
            raise
        except QuotaExceededError:
            raise
        except ServiceUnavailableError:
            raise
        except LLMError:
            raise
        except Exception as e:
            self._map_error(e)
