# Configuration

MLForge uses a centralized configuration system powered by Pydantic's `BaseSettings`. All configuration can be controlled via environment variables or a `.env` file at the root of the project.

## Environment Variables

### Core APIs
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Yes* | None |
| `GROQ_API_KEY` | Your Groq API key (fallback/fast inference) | Yes* | None |
*Note: At least one provider must be configured.*

### Database
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string | No | `postgresql://postgres:postgres@localhost:5432/mlforge` |

### Security
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `JWT_SECRET` | Secret used for signing JWTs | Yes | None |
| `JWT_ALGORITHM` | Algorithm used for JWT | No | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration time | No | `1440` |

### MLflow & Observability
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking endpoint | No | `http://localhost:5000` |

### Provider Reliability
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PROVIDER_COOLDOWN_SECONDS` | Time before retrying a failed provider | No | `300` |
| `MAX_RETRIES` | Max attempts for an API call | No | `3` |

Never commit your `.env` file. Use `.env.example` as a template.
