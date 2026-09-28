from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    service: str

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "service": "mlforge"}

@router.get("/health/providers")
def provider_status():
    from core.llm_manager import LLMManager
    manager = LLMManager()
    
    pm = manager.provider_manager
    gemini_prov = pm.get_provider("gemini")
    groq_prov = pm.get_provider("groq")
    
    return {
        "gemini": {
            "configured": bool(gemini_prov and gemini_prov.state.status != "CONFIGURATION_ERROR"),
            "status": gemini_prov.state.status if gemini_prov else "UNAVAILABLE",
            "error": gemini_prov.state.last_error if gemini_prov else None
        },
        "groq": {
            "configured": bool(groq_prov and groq_prov.state.status != "CONFIGURATION_ERROR"),
            "status": groq_prov.state.status if groq_prov else "UNAVAILABLE",
            "error": groq_prov.state.last_error if groq_prov else None
        },
        "primary": pm.primary_name,
        "fallback": pm.fallback_names
    }
