# Contributing to MLForge

Thank you for your interest in contributing to MLForge! 

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MLForge.git
   cd MLForge
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   Copy `.env.example` to `.env` and fill in your Gemini and Groq API keys.

4. **Run the Database & Infrastructure:**
   ```bash
   docker compose up -d postgres mlflow
   ```

## Code Style
- Use standard Python PEP-8 formatting.
- Type hints should be used extensively throughout the repository to maintain clarity in agent state definitions and schemas.
- Ensure all API endpoints are well-documented via Pydantic descriptions (exposed to FastAPI OpenAPI).

## Testing
Always run the test suite before submitting a Pull Request:
```bash
pytest tests/
```

For major logic changes in the orchestrator or agents, run the local evaluation benchmark:
```bash
python -m evaluation.runner --mode mock
```

## Pull Requests
- Branch out from `main` using descriptive names (e.g. `feature/time-series-agent`).
- Provide clear PR descriptions mapping to an existing issue if possible.
- Ensure no secrets or API keys are accidentally committed.

## Commit Expectations
- Write concise, descriptive commit messages.
- If introducing breaking changes to the state schema, update `migrations/` appropriately.
