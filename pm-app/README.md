# AEGIS Predictive Maintenance Backend v2

Production-ready Flask backend for predictive maintenance with integrated Gemini 2.5 Pro copilot.

## Architecture Overview

```
pm-app/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Environment-based configuration
│   ├── db.py                # SQLAlchemy database management
│   ├── models.py            # Database models (Telemetry, ModelArtifact, etc.)
│   ├── routes/              # API endpoints
│   │   ├── classification.py   # POST /predict/classification
│   │   ├── forecast.py         # POST /predict/forecast
│   │   ├── optimizer.py        # POST /optimizer/schedule
│   │   ├── models_mgmt.py      # GET /models, POST /retrain/run
│   │   ├── maintenance.py      # GET /maintenance/schedule
│   │   ├── health.py           # GET /health
│   │   └── copilot_tools.py    # POST /copilot/chat
│   └── services/            # Business logic layer
│       ├── models_loader.py         # Model caching (file:// + gs://)
│       ├── classification_service.py # XGBoost failure prediction
│       ├── forecast_service.py      # RUL forecasting
│       ├── optimizer_service.py     # Maintenance scheduling
│       ├── retrain_service.py       # Incremental retraining
│       └── copilot_service.py       # Gemini 2.5 Pro integration
├── tests/
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_classification_api.py
│   ├── test_forecast_api.py
│   └── test_optimizer_api.py
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # App + PostgreSQL services
├── requirements.txt          # Python dependencies
├── run.py                    # Development server entry point
└── .env.example              # Environment variable template
```

## Key Design Features

### 1. Clean Architecture
- **Routes**: HTTP request/response handling, validation
- **Services**: Business logic, model inference, database operations
- **Models**: SQLAlchemy ORM with JSONB support for PostgreSQL

### 2. Standardized JSON Responses
All endpoints return:
```json
{
  "data": { ... },
  "error": null
}
```
or on error:
```json
{
  "data": null,
  "error": "Error message"
}
```

### 3. Timezone-Aware Timestamps
- All `TIMESTAMPTZ` columns use UTC
- Timestamp parsing accepts ISO 8601 with fallback to UTC
- Server-side defaults via `func.now()`

### 4. PostgreSQL/SQLite Compatibility
- JSONB for PostgreSQL, JSON fallback for SQLite
- Connection pooling for PostgreSQL
- StaticPool for SQLite tests

### 5. URI-Based Model Storage
- `file://` for local filesystem
- `gs://` for Google Cloud Storage (future)
- Abstraction layer in `models_loader.py`

## Quick Start

### Local Development

1. Install dependencies:
```bash
cd pm-app
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run development server:
```bash
python run.py
```

Server runs on `http://localhost:5000`

### Docker Deployment

1. Set environment variables:
```bash
export GEMINI_API_KEY="your_key_here"
export SECRET_KEY="production_secret"
```

2. Start services:
```bash
docker-compose up -d
```

3. Check health:
```bash
curl http://localhost:5000/health
```

### Database Setup

For PostgreSQL:
```bash
psql -U pm_user -d pm_database -f ../sql/schema.sql
```

For SQLite (tests):
```bash
sqlite3 pm_app.db < ../sql/schema.sql
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Failure Prediction
```bash
POST /predict/classification
{
  "model_name": "xgb_classifier",
  "product_id": "PROD_001",
  "unit_id": "UNIT_001",
  "timestamp_before": "2024-01-15T10:00:00Z"  # optional
}
```

### RUL Forecasting
```bash
POST /predict/forecast
{
  "model_name": "xgb_regressor",
  "product_id": "PROD_001",
  "unit_id": "UNIT_001",
  "horizon_steps": 10
}
```

### Maintenance Optimization
```bash
POST /optimizer/schedule
{
  "unit_list": [
    {"product_id": "PROD_001", "unit_id": "UNIT_001"},
    {"product_id": "PROD_001", "unit_id": "UNIT_002"}
  ],
  "risk_threshold": 0.7,
  "rul_threshold": 24.0,
  "horizon_days": 7,
  "teams_available": 2
}
```

### Model Management
```bash
GET /models
POST /models/reload
POST /retrain/run
{
  "model_type": "classification",
  "incremental": true
}
```

### Copilot Chat
```bash
POST /copilot/chat
{
  "messages": [
    {"role": "user", "content": "What's the failure risk for UNIT_001?"}
  ],
  "context": {"product_id": "PROD_001"}
}
```

## Gemini 2.5 Pro Integration

### Tool Definitions
The copilot exposes three tools to Gemini:
1. `predict_failure` - Equipment failure probability
2. `predict_rul` - Remaining useful life forecast
3. `optimize_schedule` - Maintenance schedule generation

### Tool Calling Flow
```
User → POST /copilot/chat
  ↓
Gemini 2.5 Pro (with tool definitions)
  ↓ (returns tool calls)
Backend executes tools (predict_failure, etc.)
  ↓ (tool results)
Gemini 2.5 Pro (with tool results)
  ↓ (final response)
User ← Natural language answer + tool data
```

### System Prompt
The copilot enforces:
- No hallucinated predictions (always use tools)
- Ask for missing parameters
- Explain prediction confidence
- Recommend manual inspection on errors

### Configuration
```bash
# Required for copilot functionality
GEMINI_API_KEY=your_google_ai_api_key
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_classification_api.py -v
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///pm_app.db` | Database connection string |
| `MODEL_STORAGE_BACKEND` | `local` | `local` or `gcs` |
| `MODEL_STORAGE_PATH` | `./artifacts` | Local model storage path |
| `GCS_MODEL_BUCKET` | - | GCS bucket name (if backend=gcs) |
| `GEMINI_API_KEY` | - | Google AI API key |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash-exp` | Gemini model version |
| `FLASK_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `text` | `text` or `json` |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated origins |

## Model Artifacts

Expected artifacts in `MODEL_STORAGE_PATH`:
```
artifacts/
├── xgb_classifier.joblib       # Failure classification model
├── xgb_regressor.joblib        # RUL prediction model
├── scaler.joblib               # StandardScaler for features
└── encoder_engine_type.joblib  # LabelEncoder for engine_type
```

Models are registered in `model_artifact` table with:
- `path`: URI (e.g., `file:///app/artifacts/xgb_classifier_20250101_120000.joblib`)
- `version`: Timestamp-based versioning
- `metadata`: Training parameters
- `metrics`: Performance metrics (accuracy, RMSE, etc.)

## Production Considerations

### Performance
- Connection pooling configured for PostgreSQL
- Model lazy loading with in-memory cache
- Gunicorn with 4 workers, 120s timeout

### Security
- API key authentication (add middleware)
- CORS configured for specific origins
- SECRET_KEY for Flask session security

### Monitoring
- Health endpoint checks DB + models
- Structured JSON logging available
- Prometheus metrics (TODO)

### Scalability
- Stateless design (horizontal scaling)
- Redis for distributed model cache (TODO)
- Background job queue for retraining (TODO)

## Known Limitations

1. **LSTM models disabled**: XGBoost outperforms LSTM (98.75% vs 87.52% accuracy)
2. **Synchronous tool calls**: Copilot waits for tool execution (no async)
3. **No authentication**: Add API key or JWT middleware
4. **No rate limiting**: Add flask-limiter for production
5. **GCS not fully tested**: GCS storage implemented but not validated

## Future Enhancements

- [ ] Add API key authentication
- [ ] Implement rate limiting
- [ ] Add Prometheus metrics
- [ ] Use Redis for distributed model cache
- [ ] Async copilot tool execution
- [ ] LSTM model improvements
- [ ] Real-time telemetry ingestion via WebSocket
- [ ] Alembic migrations for schema evolution

## License

MIT

## Contact

Project: AEGIS Predictive Maintenance Copilot
Team: A25-CS046
