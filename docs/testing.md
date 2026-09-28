# Testing MLForge

MLForge is thoroughly tested across multiple layers: Unit, Integration, E2E, and Automated Evaluation Benchmarking.

## 1. Running the Test Suite
The project uses `pytest` for all unit and integration tests.

To run the standard test suite:
```bash
pytest tests/
```

### Test Categories:
- **Unit Tests (`tests/unit/`)**: Validates core components like `LLMManager`, `AgentState`, and `ProviderManager` without hitting external APIs.
- **Integration Tests (`tests/integration/`)**: Validates the agent orchestration and interactions.
- **Legacy Tests (`tests/legacy/`)**: Snapshot tests from previous iterative development phases.

## 2. Automated Evaluation Framework
MLForge includes an advanced, built-in Evaluation Runner designed to benchmark the multi-agent system on objective ML workloads.

### How to Run Benchmarks
You can run the CLI evaluation runner directly:
```bash
python -m evaluation.runner --mode mock
```

- `--mode mock`: (Default) Forces deterministic failures and simulates LLM generation to rapidly test workflow recovery, error catching, and the evaluation harnesses without consuming real LLM API credits.
- `--mode live`: Uses actual configured LLMs to execute the workflows end-to-end on synthetic datasets and records performance metrics.

### Test Datasets
Synthetic datasets are automatically generated during benchmarking (Simple Classification, Medium Regression, Messy Classification, Imbalanced Classification).

### Quality Gates & Reports
Evaluation metrics (Execution Time, LLM Calls, Retries, Fallbacks, Success Rates) are stored securely in PostgreSQL and visualized on the Streamlit Evaluation Dashboard.
