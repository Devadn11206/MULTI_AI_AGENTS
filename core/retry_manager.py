import time
from typing import Callable, Any
from config.settings import get_settings
from core.errors import (
    RateLimitError, TimeoutError, ServiceUnavailableError, NetworkError
)

class RetryManager:
    def __init__(self):
        settings = get_settings()
        self.max_retries = settings.MAX_RETRIES
        self.base_backoff = settings.BASE_BACKOFF_SECONDS
        self.max_backoff = settings.MAX_BACKOFF_SECONDS

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        attempts = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempts += 1
                if not self.is_retryable(e):
                    raise
                if attempts > self.max_retries:
                    raise

                # Calculate exponential backoff
                backoff = min(self.max_backoff, self.base_backoff * (2 ** (attempts - 1)))
                print(f"[RetryManager] Attempt {attempts} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)

    def is_retryable(self, e: Exception) -> bool:
        if isinstance(e, (RateLimitError, TimeoutError, ServiceUnavailableError, NetworkError)):
            return True
        return False
