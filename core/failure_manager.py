from typing import Dict, Any, Optional
from core.errors import LLMError

class FailureManager:
    @staticmethod
    def format_failure(e: Exception, agent_name: str, context: Optional[str] = None) -> Dict[str, Any]:
        error_type = e.__class__.__name__
        provider = getattr(e, 'provider', 'unknown')
        
        # Don't expose internal messages to UI directly if they contain secrets, 
        # but our API mapping already sanitized them.
        message = str(e)
        
        return {
            "success": False,
            "error_type": error_type,
            "provider": provider,
            "agent": agent_name,
            "context": context,
            "retryable": FailureManager.is_retryable(e),
            "message": f"LLM operation failed in {agent_name} [{error_type}]: {message}"
        }

    @staticmethod
    def is_retryable(e: Exception) -> bool:
        from core.retry_manager import RetryManager
        return RetryManager().is_retryable(e)
