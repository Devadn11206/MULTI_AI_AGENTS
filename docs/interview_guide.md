# MLForge Interview Guide

**"What is MLForge?"**
MLForge is a multi-agent AI-powered machine learning engineering platform. It autonomously orchestrates ML tasks like dataset analysis, feature engineering, model training, evaluation, and independent review.

**"Why multi-agent? Why not one agent?"**
Single-agent models struggle with massive, compounding context windows and suffer from "hallucination drift" when attempting complex workflows. A multi-agent system enforces modularity, compartmentalizes responsibilities, and heavily reduces token bloat per step.

**"How do agents communicate?"**
Unlike naive conversational agents, MLForge agents do not converse via a long chat transcript. They communicate by reading and mutating a strict, Pydantic-validated `AgentState` object. 

**"How does tool calling work?"**
Agents are supplied with specialized tools (like `PythonExecutor`) allowing them to write scripts, read datasets, and train models locally. Tools are heavily sandboxed and their outputs are fed back into the agent context.

**"How does self-correction work?"**
If an evaluation fails to meet user requirements (e.g., target accuracy not reached) or if python execution crashes, the Debugger agent steps in. It analyzes the error and mutates the experiment plan, recursively passing it back to the Training agent up to `MAX_ITERATIONS`.

**"How do you prevent infinite loops?"**
The `SelfCorrectionController` explicitly bounds retry loops. If `MAX_ITERATIONS` is reached without meeting requirements, it gracefully fails the workflow and surfaces the best model found so far.

**"How do Gemini and Groq work together?"**
Gemini serves as the primary intelligence. If Gemini experiences a 503 error, rate limit, or crashes, the `LLMManager` seamlessly falls back to Groq (using Llama 3) to prevent the workflow from dying midway.

**"How is observability implemented?"**
Every event (LLM prompt/response, tool execution, state mutation) is intercepted by a `TraceManager` and persisted into a PostgreSQL table. This allows users to view an "Agent Trace" on the frontend.

**"How did you test reliability?"**
We built a dedicated Evaluation framework (Phase 16) that programmatically loops synthetic datasets through the multi-agent pipeline in a mock environment to verify tool and state reliability without excessive API costs.

**"What are the limitations?"**
MLForge relies on the availability of underlying LLM APIs and currently focuses heavily on tabular ML workflows. Complex domain-specific tasks (e.g. computer vision architectures) are not fully supported yet.
