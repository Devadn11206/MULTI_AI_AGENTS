import sys
import os
import requests
from config.settings import get_settings

def check_env():
    print("Checking Python environment...")
    if sys.version_info < (3, 9):
        print("❌ Python version must be >= 3.9")
        sys.exit(1)
    print("✅ Python environment is valid.")

def check_backend():
    print("Checking FastAPI Backend...")
    try:
        res = requests.get("http://localhost:8000/health")
        if res.status_code == 200:
            print("✅ FastAPI backend is reachable.")
        else:
            print("❌ FastAPI backend returned unexpected status.")
    except Exception as e:
        print("❌ FastAPI backend is unreachable. Is uvicorn running?")

def check_frontend():
    print("Checking Streamlit Frontend...")
    try:
        res = requests.get("http://localhost:8501/healthz")
        if res.status_code in [200, 404]: # Streamlit might return 404 for healthz but connection succeeds
            print("✅ Streamlit frontend is reachable.")
    except Exception as e:
        print("❌ Streamlit frontend is unreachable. Is it running?")

def check_db():
    print("Checking Database...")
    try:
        from core.database import engine
        with engine.connect() as conn:
            print("✅ Database connection successful.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

def check_providers():
    print("Checking LLM Providers...")
    settings = get_settings()
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "placeholder":
        print("✅ Gemini API Key configured.")
    else:
        print("⚠️ Gemini API Key not found.")
        
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "placeholder":
        print("✅ Groq API Key configured.")
    else:
        print("⚠️ Groq API Key not found.")

if __name__ == "__main__":
    print("="*40)
    print("MLForge Health Check")
    print("="*40)
    check_env()
    check_db()
    check_providers()
    check_backend()
    check_frontend()
    print("="*40)
