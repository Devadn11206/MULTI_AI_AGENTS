from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # LLM Settings
    PRIMARY_LLM_PROVIDER: str = "gemini"
    FALLBACK_LLM_PROVIDERS: str = "groq"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.6-flash"
    
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    
    LLM_TIMEOUT_SECONDS: int = 60
    MAX_RETRIES: int = 2
    BASE_BACKOFF_SECONDS: int = 1
    MAX_BACKOFF_SECONDS: int = 10
    PROVIDER_COOLDOWN_SECONDS: int = 60
    
    # Workflow / Agent Settings
    MAX_AGENT_CALLS: int = 50
    MAX_CALLS_PER_AGENT: int = 10
    MAX_WORKFLOW_ITERATIONS: int = 3
    MAX_TOKENS: int = 4096
    WORKSPACE_DIR: str = "."
    MAX_UPLOAD_SIZE_MB: int = 100
    
    # Application / Production Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "http://localhost:8501,http://localhost:3000"
    API_RATE_LIMIT: str = "30/minute"
    
    # Database
    DATABASE_URL: str = "sqlite:///./mlforge.db" # Default for dev fallback, prod will override
    
    # Auth
    JWT_SECRET_KEY: str = "change_me_in_production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # MLflow
    MLFLOW_ENABLED: bool = False
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    # Observability (Phase 15)
    TRACE_RETENTION_DAYS: int = 30
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def get_settings() -> Settings:
    return Settings()
