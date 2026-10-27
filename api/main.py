import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, datasets, workflows, results, artifacts, auth, trace, evaluation
from api.middleware.request_id import RequestIDMiddleware
from api.middleware.errors import global_exception_handler
from config.settings import get_settings
from core.logging import setup_logging

setup_logging()
settings = get_settings()

app = FastAPI(
    title="MLForge API",
    version="1.0.0",
    description="Multi-agent ML engineering platform"
)

# Configure CORS
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]

@app.on_event("startup")
def startup_event():
    from core.database import Base, engine
    import models.domain
    import observability.db_model
    import evaluation.db_models
    Base.metadata.create_all(bind=engine)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestIDMiddleware)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(auth.router)
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(workflows.router)
app.include_router(results.router)
app.include_router(artifacts.router)
app.include_router(trace.router)
app.include_router(evaluation.router)
