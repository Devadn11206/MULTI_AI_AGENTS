# MLForge Project Audit

## 1. Architecture
MLForge is a Multi-Agent Machine Learning Engineering Platform.
- **Request Layer**: FastAPI backend providing endpoints for workflows, evaluation, and traces.
- **Frontend Layer**: Streamlit dashboard (UI) serving as the main interface.
- **Agent Layer**: Agents (Orchestrator, Data Analyst, Feature Engineering, Model Engineering, Training, Evaluation, Debugger, Reviewer) implemented as tool-using sub-agents.
- **Shared State**: In-memory and persisted `AgentState` object allowing stateful communication between agents.
- **LLM Layer**: Configurable multi-provider setup (Gemini, Groq) with robust retry mechanisms, rate-limiting handlers, and fallback capabilities.
- **Persistence Layer**: PostgreSQL database holding user accounts, dataset info, workflows, trace events, and evaluation metrics.
- **Observability Layer**: `TraceManager` tracking LLM, agent, tool, and system events for tracing.

## 2. Major Components
- `core/`: Core settings, database connection, LLM and trace managers.
- `api/`: FastAPI application, endpoints (workflows, evaluation, results).
- `ui/`: Streamlit dashboard, trace dashboard, evaluation dashboard.
- `agents/`: Implementation of the 8 distinct AI agents.
- `tools/`: Implementations for python execution, dataset extraction, ML workflows.
- `evaluation/`: The programmatic testing and CLI runner.
- `debug/`: The Self-Correction Controller for executing execution loops.
- `models/` & `repositories/`: SQLAlchemy models and DB repository patterns.

## 3. Important Files
- `api/main.py`: Main FastAPI application root.
- `ui/app.py`: Main Streamlit frontend application.
- `docker-compose.yml` / `Dockerfile`: Dockerized execution environments.
- `core/orchestration/orchestrator.py`: Entry point for workflow planning.
- `debug/self_correction_controller.py`: The evaluation & feedback loop execution engine.
- `evaluation/runner.py`: The evaluation benchmark CLI tool.

## 4. Dependencies
- **Core backend**: `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg2-binary`.
- **Frontend**: `streamlit`.
- **AI/LLM**: `google-genai`, `groq`, `openai` (for base classes).
- **ML/DS**: `scikit-learn`, `pandas`, `mlflow`.
- **Misc**: `pydantic`, `pyjwt`, `bcrypt`.

## 5. Configuration
- `.env` controls secrets and thresholds.
- `config/settings.py` loads and validates configurations.
- `LLMManager` handles dynamic fallback configurations.

## 6. Known Issues / Limitations
- Agent planning heavily depends on LLM context windows (max token limits).
- Tool execution assumes valid python environments for synthetic tests (sandbox execution via subprocess instead of secure containers).
- LLM hallucinations can still break rigid formatting schemas if self-correction exhaustion occurs (MAX_ITERATIONS).
