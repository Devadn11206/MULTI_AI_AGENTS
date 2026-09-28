import json
from typing import Optional, Type
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import get_settings
from core.llm_manager import LLMManager
from core.llm.models import LLMResponse
from core.llm.errors import (
    AuthenticationError, RateLimitError, QuotaExceededError,
    ProviderUnavailableError, InvalidRequestError, TimeoutError, LLMError
)

class GeminiProvider(LLMManager):
    def __init__(self):
        settings = get_settings()
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise AuthenticationError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = settings.GEMINI_MODEL or "gemini-3.8-flash"
        self.provider_name_val = "gemini"

    @property
    def provider_name(self) -> str:
        return self.provider_name_val

    def _map_error(self, e: Exception):
        if isinstance(e, APIError):
            code = e.code
            msg = e.message.lower() if e.message else ""
            if code == 401 or code == 403:
                raise AuthenticationError(f"Gemini Auth Error: {e.message}") from e
            elif code == 429:
                if "quota" in msg:
                    raise QuotaExceededError(f"Gemini Quota Exceeded: {e.message}") from e
                raise RateLimitError(f"Gemini Rate Limit: {e.message}") from e
            elif code in (500, 502, 503, 504):
                raise ProviderUnavailableError(f"Gemini Unavailable: {e.message}") from e
            elif code == 400:
                raise InvalidRequestError(f"Gemini Invalid Request: {e.message}") from e
        raise LLMError(f"Gemini Unknown Error: {str(e)}") from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            config_dict = {}
            if system_prompt:
                config_dict["system_instruction"] = system_prompt
                
            if response_schema:
                schema_dict = response_schema.model_json_schema()
                schema_dict.pop("$defs", None)
                if "properties" in schema_dict:
                    for prop in schema_dict["properties"].values():
                        prop.pop("title", None)
                        prop.pop("default", None)
                        if "anyOf" in prop:
                            types_list = [t.get("type") for t in prop["anyOf"] if t.get("type") != "null"]
                            if types_list:
                                prop["type"] = types_list[0]
                            prop.pop("anyOf")
                config_dict["response_mime_type"] = "application/json"
                config_dict["response_schema"] = schema_dict

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict) if config_dict else None
            )
            
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
                
            finish_reason = None
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = str(response.candidates[0].finish_reason)
                
            return LLMResponse(
                content=response.text or "",
                provider=self.provider_name_val,
                model=self.model_name,
                usage=usage,
                finish_reason=finish_reason
            )
            
        except AuthenticationError:
            raise
        except RateLimitError:
            raise
        except QuotaExceededError:
            raise
        except ProviderUnavailableError:
            raise
        except InvalidRequestError:
            raise
        except LLMError:
            raise
        except Exception as e:
            self._map_error(e)
