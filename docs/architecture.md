# MLForge Architecture

## 1. Request Layer
The entry point is a FastAPI application serving REST endpoints, which are consumed by the Streamlit frontend. The backend handles job queuing, state persistence, and authentication securely via JWT.

## 2. Orchestrator
The `Orchestrator` receives the user's ML objective and constructs a deterministic multi-agent workflow. It parses intents and delegates tasks down to the specialized Agents.

## 3. Agent Layer
A series of highly specialized LLM-backed AI agents execute specific tasks:
- **Data Analyst:** Analyzes dataset features and columns.
- **Feature Agent:** Decides on necessary preprocessing (imputation, encoding).
- **Model Agent:** Formulates an experiment plan of candidate algorithms.
- **Training Agent:** Writes and executes training scripts via Tools.
- **Evaluation Agent:** Benchmarks trained models against metrics.
- **Debugger:** Intervenes when execution fails or metrics are unacceptable.
- **Reviewer:** Final validation gatekeeper before user delivery.

## 4. Shared State
Agents do not communicate through a messy chat transcript. Instead, they read from and mutate a strictly typed, central `AgentState` object (powered by Pydantic). This enforces predictable knowledge transfer and mitigates hallucination drift.

## 5. Tool Layer
Agents use a strictly sandboxed set of tools. Examples include the `PythonExecutor` and `ModelComparator`, which allow the models to actually run experiments instead of just theorizing them.

## 6. LLM Layer
The `LLMManager` interfaces with multiple LLM providers (Gemini, Groq) using a unified `BaseProvider` architecture. It supports schema validation and JSON mode natively.

## 7. Reliability Layer
Wrapped around the LLM API is the `RetryManager`, featuring exponential backoff, rate limit handling, and automatic provider fallbacks (e.g., if Gemini experiences a 503, Groq automatically assumes the load).

## 8. Persistence
All artifacts, traces, evaluation runs, and workflow results are securely persisted inside PostgreSQL and MLflow (for models).

## 9. Observability
The `TraceManager` streams low-level reasoning, tool use, and system events to the database and UI, allowing for transparent introspection of AI decisions without exposing raw system prompts.

## 10. Evaluation
An automated benchmarking harness evaluates the entire architecture against synthetic workloads programmatically, ensuring no regressions break the complex multi-agent handoffs.
