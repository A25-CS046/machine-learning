# End-to-End Validation Plan
## Predictive Maintenance Copilot Backend

**Version:** 2.0  
**Date:** November 19, 2025  
**Team:** A25-CS046

---

## Table of Contents

1. [High-Level Test Strategy](#1-high-level-test-strategy)
2. [Detailed Test Cases](#2-detailed-test-cases)
3. [Cross-Flow E2E Scenarios](#3-cross-flow-e2e-scenarios)
4. [Automation Guidance (pytest)](#4-automation-guidance-pytest)

---

## 1. High-Level Test Strategy

### 1.1 Testing Pyramid

```
                    ┌─────────────────┐
                    │  E2E Scenarios  │  (3 comprehensive journeys)
                    └─────────────────┘
                  ┌───────────────────────┐
                  │  Integration Tests    │  (API + Service + DB)
                  └───────────────────────┘
              ┌─────────────────────────────────┐
              │        Unit Tests               │  (Service logic, utils)
              └─────────────────────────────────┘
```

### 1.2 Test Coverage Areas

| Area | Coverage | Priority |
|------|----------|----------|
| **Health & Connectivity** | `/health` endpoint, DB connectivity, model loader status | P0 |
| **Prediction Endpoints** | Classification with telemetry aggregation, explicit features | P0 |
| **Forecasting Endpoints** | RUL prediction, horizon-based forecasts | P0 |
| **Optimizer & Persistence** | Schedule generation, DB writes to `maintenance_schedule` | P0 |
| **Model Management** | Listing artifacts, cache reload, version ordering | P1 |
| **Retraining** | Incremental training, artifact creation, pointer updates | P1 |
| **Copilot Tool-Calling** | Individual tool endpoints, parameter validation | P0 |
| **Copilot Conversational** | Multi-turn chats, tool execution, fallback handling | P0 |
| **Error Paths** | Missing fields, invalid timestamps, no telemetry, tool failures | P0 |
| **Timestamp Handling** | UTC parsing, timezone fallback, TIMESTAMPTZ queries | P1 |
| **Model Reload Workflow** | Cache invalidation, lazy reloading, hot-swap | P1 |

### 1.3 Test Environment Setup

**Requirements:**
- SQLite in-memory database for isolation
- Mock Gemini 2.5 Pro HTTP client
- Pre-seeded telemetry data (10+ timesteps per unit)
- Pre-registered model artifacts in DB
- Sample XGBoost models (or mocks)

**Test Data Strategy:**
- Use `conftest.py` fixtures for repeatable test data
- Generate synthetic telemetry with known failure patterns
- Create deterministic model artifacts with fixed versions

### 1.4 Success Criteria

- ✅ All P0 tests pass with 100% success rate
- ✅ All endpoints return standardized `{data, error}` envelope
- ✅ No database leaks between tests (proper rollback)
- ✅ Gemini tool-calls execute correctly (with mocks)
- ✅ Timezone-aware timestamps parsed correctly
- ✅ Model cache invalidates and reloads correctly

---

## 2. Detailed Test Cases

### 2.1 Health & Connectivity

#### Test Case 1.1: Health Check - Healthy System
- **Goal:** Verify health endpoint returns OK when DB and models are accessible
- **Pre-conditions:** Database initialized, model artifacts exist
- **HTTP Request:**
  ```http
  GET /health
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "status": "healthy",
      "database": "ok",
      "models": "ok",
      "cached_models_count": <integer>
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.status == "healthy"`
  - `data.database == "ok"`
  - `data.cached_models_count >= 0`

#### Test Case 1.2: Health Check - Database Unavailable
- **Goal:** Verify health endpoint returns 503 when DB is unreachable
- **Pre-conditions:** Mock DB connection failure
- **HTTP Request:**
  ```http
  GET /health
  ```
- **Expected Status:** `503 Service Unavailable`
- **Expected JSON:**
  ```json
  {
    "data": {
      "status": "unhealthy",
      "database": "error: <connection_error_message>",
      "models": "ok" | "error: <message>",
      "cached_models_count": <integer>
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.status == "unhealthy"`
  - `status_code == 503`

---

### 2.2 Prediction Endpoints (Classification)

#### Test Case 2.1: Classification - Happy Path with Telemetry
- **Goal:** Predict failure using historical telemetry aggregation
- **Pre-conditions:** 
  - Telemetry data exists for `PROD_001/UNIT_001` (10+ rows)
  - Model artifact `xgb_classifier` registered in DB
  - Scaler and encoder artifacts available
- **HTTP Request:**
  ```http
  POST /predict/classification
  Content-Type: application/json
  
  {
    "model_name": "xgb_classifier",
    "product_id": "PROD_001",
    "unit_id": "UNIT_001"
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "prediction": 0 | 1,
      "failure_type": null | "<failure_type_string>",
      "probabilities": {
        "no_failure": <float_0_to_1>,
        "failure": <float_0_to_1>
      },
      "model_name": "xgb_classifier",
      "model_version": "<version_string>",
      "inputs_used": {
        "source": "telemetry",
        "product_id": "PROD_001",
        "unit_id": "UNIT_001",
        "timestamp_before": "<ISO8601_timestamp>",
        "rows_used": <integer>
      },
      "fallback_recommendation": "<string>"
    },
    "error": null
  }
  ```
- **Side-effects:** None (read-only)
- **Assertions:**
  - `data.prediction in [0, 1]`
  - `data.probabilities.no_failure + data.probabilities.failure ≈ 1.0`
  - `data.inputs_used.rows_used > 0`
  - `data.model_version` matches latest artifact version

#### Test Case 2.2: Classification - Explicit Features
- **Goal:** Predict failure using explicit feature dictionary
- **Pre-conditions:** Model artifacts exist
- **HTTP Request:**
  ```http
  POST /predict/classification
  Content-Type: application/json
  
  {
    "model_name": "xgb_classifier",
    "product_id": "PROD_001",
    "unit_id": "UNIT_001",
    "features": {
      "air_temperature_K_mean": 300.0,
      "air_temperature_K_std": 5.0,
      "air_temperature_K_min": 290.0,
      "air_temperature_K_max": 310.0,
      "air_temperature_K_last": 305.0,
      "air_temperature_K_trend": 0.5,
      "...": "...(31 features total)",
      "engine_type_encoded": 1
    }
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:** (Same structure as 2.1)
- **Side-effects:** None
- **Assertions:**
  - `data.inputs_used.source == "explicit_features"`
  - Prediction succeeds without telemetry query

#### Test Case 2.3: Classification - Missing Required Fields
- **Goal:** Verify validation rejects incomplete requests
- **Pre-conditions:** None
- **HTTP Request:**
  ```http
  POST /predict/classification
  Content-Type: application/json
  
  {
    "model_name": "xgb_classifier"
  }
  ```
- **Expected Status:** `400 Bad Request`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "Missing required fields: model_name, product_id, unit_id"
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `error` contains substring "required fields"

#### Test Case 2.4: Classification - Invalid Timestamp Format
- **Goal:** Verify timestamp parsing rejects malformed dates
- **Pre-conditions:** Telemetry exists
- **HTTP Request:**
  ```http
  POST /predict/classification
  Content-Type: application/json
  
  {
    "model_name": "xgb_classifier",
    "product_id": "PROD_001",
    "unit_id": "UNIT_001",
    "timestamp_before": "not-a-valid-timestamp"
  }
  ```
- **Expected Status:** `400 Bad Request` or `404 Not Found`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "<error_message_about_timestamp>"
  }
  ```
- **Side-effects:** None

#### Test Case 2.5: Classification - No Telemetry Available
- **Goal:** Verify graceful error when unit has no data
- **Pre-conditions:** Database empty for specified unit
- **HTTP Request:**
  ```http
  POST /predict/classification
  Content-Type: application/json
  
  {
    "model_name": "xgb_classifier",
    "product_id": "NONEXISTENT",
    "unit_id": "NONEXISTENT"
  }
  ```
- **Expected Status:** `404 Not Found`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "No telemetry found for product_id=NONEXISTENT, unit_id=NONEXISTENT"
  }
  ```
- **Side-effects:** None

#### Test Case 2.6: Classification - Timestamp with UTC Timezone
- **Goal:** Verify ISO 8601 timestamp with explicit UTC timezone
- **Pre-conditions:** Telemetry exists before specified timestamp
- **HTTP Request:**
  ```http
  POST /predict/classification
  Content-Type: application/json
  
  {
    "model_name": "xgb_classifier",
    "product_id": "PROD_001",
    "unit_id": "UNIT_001",
    "timestamp_before": "2024-01-15T10:00:00Z"
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:** (Same structure as 2.1)
- **Side-effects:** None
- **Assertions:**
  - Timestamp parsed correctly as UTC
  - Only telemetry before `2024-01-15T10:00:00Z` used

---

### 2.3 Forecasting Endpoints (RUL Prediction)

#### Test Case 3.1: Forecast - Happy Path
- **Goal:** Predict RUL with horizon-based forecast
- **Pre-conditions:** 
  - Telemetry with `synthetic_RUL` exists for unit
  - Model artifact `xgb_regressor` registered
- **HTTP Request:**
  ```http
  POST /predict/forecast
  Content-Type: application/json
  
  {
    "model_name": "xgb_regressor",
    "product_id": "PROD_001",
    "unit_id": "UNIT_001",
    "horizon_steps": 10
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "product_id": "PROD_001",
      "unit_id": "UNIT_001",
      "current_rul_hours": <float>,
      "forecast_horizon_steps": 10,
      "forecast": [
        {
          "step": 1,
          "timestamp": "<ISO8601>",
          "predicted_rul_hours": <float>,
          "confidence": "medium"
        },
        ...
      ],
      "model_name": "xgb_regressor",
      "model_version": "<version>",
      "baseline_timestamp": "<ISO8601>",
      "telemetry_rows_used": <integer>
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.forecast.length == 10`
  - `data.forecast[0].step == 1`
  - `data.current_rul_hours >= 0`
  - RUL decreases monotonically in forecast

#### Test Case 3.2: Forecast - Missing Required Fields
- **Goal:** Verify validation for missing product_id/unit_id
- **Pre-conditions:** None
- **HTTP Request:**
  ```http
  POST /predict/forecast
  Content-Type: application/json
  
  {
    "horizon_steps": 5
  }
  ```
- **Expected Status:** `400 Bad Request`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "Missing required fields: product_id, unit_id"
  }
  ```
- **Side-effects:** None

#### Test Case 3.3: Forecast - Insufficient RUL History
- **Goal:** Verify error when no RUL telemetry exists
- **Pre-conditions:** Telemetry exists but `synthetic_RUL` is NULL
- **HTTP Request:**
  ```http
  POST /predict/forecast
  Content-Type: application/json
  
  {
    "product_id": "PROD_002",
    "unit_id": "UNIT_NO_RUL"
  }
  ```
- **Expected Status:** `404 Not Found`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "No RUL telemetry found for product_id=PROD_002, unit_id=UNIT_NO_RUL"
  }
  ```
- **Side-effects:** None

---

### 2.4 Optimizer & Persistence

#### Test Case 4.1: Optimizer - Single Unit Scheduling
- **Goal:** Generate maintenance schedule for one high-risk unit
- **Pre-conditions:** 
  - Telemetry exists for unit with high failure probability
  - RUL < 24 hours
- **HTTP Request:**
  ```http
  POST /optimizer/schedule
  Content-Type: application/json
  
  {
    "unit_list": [
      {"product_id": "PROD_001", "unit_id": "UNIT_001"}
    ],
    "risk_threshold": 0.7,
    "rul_threshold": 24.0,
    "horizon_days": 7,
    "teams_available": 2
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "schedule_id": "SCH_<12_hex_chars>",
      "total_units_evaluated": 1,
      "high_risk_units_found": <0_or_1>,
      "maintenance_scheduled": <0_or_1>,
      "recommendations": [
        {
          "product_id": "PROD_001",
          "unit_id": "UNIT_001",
          "recommended_start": "<ISO8601>",
          "recommended_end": "<ISO8601>",
          "risk_score": <float>,
          "reason": "<string>",
          "actions": ["<action_string>", ...]
        }
      ],
      "constraints_applied": {
        "risk_threshold": 0.7,
        "rul_threshold": 24.0,
        "horizon_days": 7,
        "teams_available": 2,
        "hours_per_day": 8,
        "earliest_allowed": "<ISO8601>",
        "latest_allowed": "<ISO8601>"
      }
    },
    "error": null
  }
  ```
- **Side-effects:** 
  - Inserts rows into `maintenance_schedule` table
  - All entries have same `schedule_id`
  - `status` set to `'PENDING'`
- **Assertions:**
  - `data.schedule_id` starts with `"SCH_"`
  - If unit is high-risk: `data.maintenance_scheduled == 1`
  - DB contains matching schedule entries

#### Test Case 4.2: Optimizer - Multi-Unit Scheduling
- **Goal:** Generate optimal schedule for multiple units with team constraints
- **Pre-conditions:** Multiple units with varying risk levels
- **HTTP Request:**
  ```http
  POST /optimizer/schedule
  Content-Type: application/json
  
  {
    "unit_list": [
      {"product_id": "PROD_001", "unit_id": "UNIT_001"},
      {"product_id": "PROD_001", "unit_id": "UNIT_002"},
      {"product_id": "PROD_002", "unit_id": "UNIT_003"}
    ],
    "risk_threshold": 0.5,
    "teams_available": 2,
    "hours_per_day": 8
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:** (Same structure as 4.1 with multiple recommendations)
- **Side-effects:** 
  - Multiple rows inserted into `maintenance_schedule`
  - Recommendations sorted by `risk_score` descending
- **Assertions:**
  - `data.total_units_evaluated == 3`
  - `data.maintenance_scheduled <= data.high_risk_units_found`
  - No more than `teams_available` schedules per day

#### Test Case 4.3: Optimizer - No High-Risk Units
- **Goal:** Verify empty schedule when all units are healthy
- **Pre-conditions:** All units have low failure probability and high RUL
- **HTTP Request:**
  ```http
  POST /optimizer/schedule
  Content-Type: application/json
  
  {
    "unit_list": [
      {"product_id": "PROD_HEALTHY", "unit_id": "UNIT_HEALTHY"}
    ],
    "risk_threshold": 0.9,
    "rul_threshold": 10.0
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "schedule_id": "SCH_<12_hex_chars>",
      "total_units_evaluated": 1,
      "high_risk_units_found": 0,
      "maintenance_scheduled": 0,
      "recommendations": [],
      ...
    },
    "error": null
  }
  ```
- **Side-effects:** Empty schedule_id generated but no maintenance entries
- **Assertions:**
  - `data.recommendations == []`

#### Test Case 4.4: Optimizer - Missing unit_list
- **Goal:** Verify validation rejects request without unit_list
- **Pre-conditions:** None
- **HTTP Request:**
  ```http
  POST /optimizer/schedule
  Content-Type: application/json
  
  {
    "risk_threshold": 0.7
  }
  ```
- **Expected Status:** `400 Bad Request`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "unit_list is required and must be a non-empty list"
  }
  ```
- **Side-effects:** None

#### Test Case 4.5: Optimizer - Invalid Unit Format
- **Goal:** Verify validation checks unit structure
- **Pre-conditions:** None
- **HTTP Request:**
  ```http
  POST /optimizer/schedule
  Content-Type: application/json
  
  {
    "unit_list": [
      {"product_id": "PROD_001"}
    ]
  }
  ```
- **Expected Status:** `400 Bad Request`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "Each unit must have product_id and unit_id"
  }
  ```
- **Side-effects:** None

---

### 2.5 Maintenance Schedule Queries

#### Test Case 5.1: Query Schedules - All
- **Goal:** Retrieve all maintenance schedules without filters
- **Pre-conditions:** `maintenance_schedule` table has entries
- **HTTP Request:**
  ```http
  GET /maintenance/schedule
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": [
      {
        "id": <bigint>,
        "schedule_id": "<SCH_id>",
        "product_id": "<string>",
        "unit_id": "<string>",
        "recommended_start": "<ISO8601>",
        "recommended_end": "<ISO8601>",
        "reason": "<string>",
        "risk_score": <float>,
        "model_version": "<string>",
        "actions": ["<action>", ...],
        "constraints_applied": {...},
        "created_at": "<ISO8601>",
        "status": "PENDING" | "COMPLETED" | "CANCELLED"
      },
      ...
    ],
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - Returns array (may be empty)
  - Ordered by `recommended_start` ascending

#### Test Case 5.2: Query Schedules - Filter by product_id and unit_id
- **Goal:** Filter schedules for specific unit
- **Pre-conditions:** Schedules exist for multiple units
- **HTTP Request:**
  ```http
  GET /maintenance/schedule?product_id=PROD_001&unit_id=UNIT_001
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:** (Array with matching entries only)
- **Side-effects:** None
- **Assertions:**
  - All returned entries have `product_id == "PROD_001"`
  - All returned entries have `unit_id == "UNIT_001"`

#### Test Case 5.3: Query Schedules - Filter by Status
- **Goal:** Retrieve only PENDING schedules
- **Pre-conditions:** Mix of PENDING/COMPLETED schedules exist
- **HTTP Request:**
  ```http
  GET /maintenance/schedule?status=PENDING
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:** (Array with PENDING entries only)
- **Side-effects:** None
- **Assertions:**
  - All returned entries have `status == "PENDING"`

#### Test Case 5.4: Query Schedules - Date Range Filter
- **Goal:** Filter schedules within date range
- **Pre-conditions:** Schedules with various dates exist
- **HTTP Request:**
  ```http
  GET /maintenance/schedule?start_date=2024-01-01T00:00:00Z&end_date=2024-12-31T23:59:59Z
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:** (Array with entries in range)
- **Side-effects:** None
- **Assertions:**
  - All `recommended_start >= start_date`
  - All `recommended_end <= end_date`

---

### 2.6 Model Management

#### Test Case 6.1: List Models
- **Goal:** Retrieve all registered model artifacts
- **Pre-conditions:** Multiple model artifacts in DB with different versions
- **HTTP Request:**
  ```http
  GET /models
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": [
      {
        "model_name": "xgb_classifier",
        "version": "<timestamp_version>",
        "path": "file:///path/to/model.joblib",
        "metadata": {...},
        "metrics": {...},
        "promoted_at": "<ISO8601>"
      },
      ...
    ],
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - Returns latest version for each `model_name`
  - Ordered by `model_name` ascending
  - Each entry has valid `path` URI

#### Test Case 6.2: Reload Model Cache
- **Goal:** Clear model cache and force reload
- **Pre-conditions:** Models loaded in cache
- **HTTP Request:**
  ```http
  POST /models/reload
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "message": "Model cache cleared",
      "cached_models_before": ["<uri1>", "<uri2>", ...],
      "cache_size_after": 0
    },
    "error": null
  }
  ```
- **Side-effects:** 
  - Model loader cache cleared
  - Next inference will reload models from disk/DB
- **Assertions:**
  - `data.cache_size_after == 0`
  - `len(data.cached_models_before) >= 0`

---

### 2.7 Retraining

#### Test Case 7.1: Retrain Classification Model - Incremental
- **Goal:** Trigger incremental retraining for classifier
- **Pre-conditions:** 
  - New telemetry exists after last retrain timestamp
  - `retrain_pointer` table has entry for `xgb_classifier`
- **HTTP Request:**
  ```http
  POST /retrain/run
  Content-Type: application/json
  
  {
    "model_type": "classification",
    "incremental": true
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "model_name": "xgb_classifier",
      "version": "<timestamp_YYYYMMDD_HHMMSS>",
      "path": "file:///<path>/xgb_classifier_<version>.joblib",
      "metrics": {
        "accuracy": <float>,
        "recall": <float>,
        "samples": <integer>
      },
      "training_samples": <integer>,
      "incremental": true
    },
    "error": null
  }
  ```
- **Side-effects:**
  - New `.joblib` file created in `MODEL_STORAGE_PATH`
  - New row inserted into `model_artifact` table
  - `retrain_pointer.last_retrain_ts` updated to latest telemetry timestamp
  - `retrain_pointer.last_retrain_id` updated to latest telemetry ID
- **Assertions:**
  - `data.version` format: `YYYYMMDD_HHMMSS`
  - `data.path` contains version in filename
  - `data.training_samples > 0`
  - DB has new `model_artifact` entry

#### Test Case 7.2: Retrain Forecast Model - Full
- **Goal:** Trigger full (non-incremental) retraining for regressor
- **Pre-conditions:** Telemetry with `synthetic_RUL` exists
- **HTTP Request:**
  ```http
  POST /retrain/run
  Content-Type: application/json
  
  {
    "model_type": "forecast",
    "incremental": false
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "model_name": "xgb_regressor",
      "version": "<timestamp>",
      "path": "file:///<path>/xgb_regressor_<version>.joblib",
      "metrics": {
        "mae": <float>,
        "rmse": <float>,
        "r2": <float>,
        "samples": <integer>
      },
      "training_samples": <integer>,
      "incremental": false
    },
    "error": null
  }
  ```
- **Side-effects:** (Same as 7.1)
- **Assertions:**
  - Uses all telemetry (not just new data)
  - Metrics include MAE, RMSE, R²

#### Test Case 7.3: Retrain - No New Data
- **Goal:** Verify error when no telemetry available for incremental retrain
- **Pre-conditions:** No telemetry after last retrain timestamp
- **HTTP Request:**
  ```http
  POST /retrain/run
  Content-Type: application/json
  
  {
    "model_type": "classification",
    "incremental": true
  }
  ```
- **Expected Status:** `400 Bad Request`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "No new telemetry data available for retraining"
  }
  ```
- **Side-effects:** None

#### Test Case 7.4: Retrain - Invalid Model Type
- **Goal:** Verify validation rejects unknown model types
- **Pre-conditions:** None
- **HTTP Request:**
  ```http
  POST /retrain/run
  Content-Type: application/json
  
  {
    "model_type": "unknown_type"
  }
  ```
- **Expected Status:** `400 Bad Request`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "model_type must be \"classification\" or \"forecast\""
  }
  ```
- **Side-effects:** None

---

### 2.8 Copilot Tool Endpoints

#### Test Case 8.1: Tool - Predict Failure
- **Goal:** Simplified failure prediction for LLM integration
- **Pre-conditions:** Telemetry exists for unit
- **HTTP Request:**
  ```http
  POST /copilot/tools/predict_failure
  Content-Type: application/json
  
  {
    "product_id": "PROD_001",
    "unit_id": "UNIT_001"
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "product_id": "PROD_001",
      "unit_id": "UNIT_001",
      "failure_probability": <float_0_to_1>,
      "will_fail": true | false,
      "failure_type": null | "<string>",
      "recommendation": "<string>"
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.failure_probability` in range [0, 1]
  - `data.will_fail` is boolean
  - If `will_fail == true`, then `failure_type` is not null

#### Test Case 8.2: Tool - Predict RUL
- **Goal:** Simplified RUL prediction for LLM integration
- **Pre-conditions:** Telemetry with RUL exists
- **HTTP Request:**
  ```http
  POST /copilot/tools/predict_rul
  Content-Type: application/json
  
  {
    "product_id": "PROD_001",
    "unit_id": "UNIT_001",
    "horizon_steps": 5
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "product_id": "PROD_001",
      "unit_id": "UNIT_001",
      "current_rul_hours": <float>,
      "forecast_horizon_steps": 5,
      "critical": true | false,
      "recommendation": "<string>"
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.current_rul_hours >= 0`
  - `data.critical == true` if RUL < 24 hours

#### Test Case 8.3: Tool - Optimize Schedule
- **Goal:** Simplified optimizer for LLM integration
- **Pre-conditions:** Multiple units exist
- **HTTP Request:**
  ```http
  POST /copilot/tools/optimize_schedule
  Content-Type: application/json
  
  {
    "unit_list": [
      {"product_id": "PROD_001", "unit_id": "UNIT_001"}
    ]
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "schedule_id": "SCH_<12_hex>",
      "units_scheduled": <integer>,
      "high_risk_units": <integer>,
      "recommendations": [...]
    },
    "error": null
  }
  ```
- **Side-effects:** Inserts into `maintenance_schedule`
- **Assertions:**
  - `data.units_scheduled <= len(unit_list)`

---

### 2.9 Copilot Chat Endpoint

#### Test Case 9.1: Chat - Simple Query with Tool Call
- **Goal:** Verify copilot executes tool and returns natural language response
- **Pre-conditions:** 
  - Gemini API key configured
  - Mock Gemini response with tool call
- **HTTP Request:**
  ```http
  POST /copilot/chat
  Content-Type: application/json
  
  {
    "messages": [
      {"role": "user", "content": "What's the failure risk for UNIT_001?"}
    ],
    "context": {"product_id": "PROD_001"}
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "reply": "<natural_language_response_from_gemini>",
      "tool_calls": [
        {
          "name": "predict_failure",
          "arguments": {
            "product_id": "PROD_001",
            "unit_id": "UNIT_001"
          }
        }
      ],
      "tool_results": [
        {
          "tool_name": "predict_failure",
          "success": true,
          "result": {...}
        }
      ],
      "raw_model_response": {...}
    },
    "error": null
  }
  ```
- **Side-effects:** None (tools are read-only)
- **Assertions:**
  - `data.tool_calls.length > 0`
  - `data.tool_results[0].success == true`
  - `data.reply` is non-empty string

#### Test Case 9.2: Chat - Missing Parameters → Follow-up
- **Goal:** Verify copilot asks for missing unit_id
- **Pre-conditions:** Mock Gemini to respond with clarifying question
- **HTTP Request:**
  ```http
  POST /copilot/chat
  Content-Type: application/json
  
  {
    "messages": [
      {"role": "user", "content": "Check the failure risk"}
    ]
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "reply": "<question_asking_for_product_id_and_unit_id>",
      "tool_calls": [],
      "tool_results": [],
      "raw_model_response": {...}
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.tool_calls == []` (no tools executed)
  - `data.reply` contains request for missing parameters

#### Test Case 9.3: Chat - Tool Execution Failure → Safe Fallback
- **Goal:** Verify copilot handles tool errors gracefully
- **Pre-conditions:** 
  - Mock tool to raise exception
  - Mock Gemini to provide fallback recommendation
- **HTTP Request:**
  ```http
  POST /copilot/chat
  Content-Type: application/json
  
  {
    "messages": [
      {"role": "user", "content": "Predict failure for UNIT_BROKEN"}
    ],
    "context": {"product_id": "PROD_001"}
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "reply": "<explanation_of_error_and_manual_inspection_recommendation>",
      "tool_calls": [
        {
          "name": "predict_failure",
          "arguments": {"product_id": "PROD_001", "unit_id": "UNIT_BROKEN"}
        }
      ],
      "tool_results": [
        {
          "tool_name": "predict_failure",
          "success": false,
          "error": "<error_message>"
        }
      ],
      "raw_model_response": {...}
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - `data.tool_results[0].success == false`
  - `data.reply` mentions error and recommends manual action

#### Test Case 9.4: Chat - Multi-Turn Conversation
- **Goal:** Verify copilot maintains conversation context
- **Pre-conditions:** Previous messages in history
- **HTTP Request:**
  ```http
  POST /copilot/chat
  Content-Type: application/json
  
  {
    "messages": [
      {"role": "user", "content": "Check UNIT_001"},
      {"role": "assistant", "content": "UNIT_001 has 95% failure risk."},
      {"role": "user", "content": "What's the RUL?"}
    ],
    "context": {"product_id": "PROD_001"}
  }
  ```
- **Expected Status:** `200 OK`
- **Expected JSON:**
  ```json
  {
    "data": {
      "reply": "<rul_information>",
      "tool_calls": [
        {
          "name": "predict_rul",
          "arguments": {"product_id": "PROD_001", "unit_id": "UNIT_001"}
        }
      ],
      "tool_results": [...],
      "raw_model_response": {...}
    },
    "error": null
  }
  ```
- **Side-effects:** None
- **Assertions:**
  - Copilot infers `unit_id` from conversation context
  - Executes `predict_rul` tool

#### Test Case 9.5: Chat - Missing Gemini API Key
- **Goal:** Verify graceful error when API key not configured
- **Pre-conditions:** `GEMINI_API_KEY` not set
- **HTTP Request:**
  ```http
  POST /copilot/chat
  Content-Type: application/json
  
  {
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }
  ```
- **Expected Status:** `503 Service Unavailable`
- **Expected JSON:**
  ```json
  {
    "data": null,
    "error": "GEMINI_API_KEY not configured"
  }
  ```
- **Side-effects:** None

---

## 3. Cross-Flow E2E Scenarios

### 3.1 Scenario A: Detect → Forecast → Schedule → Query

**User Journey:** Maintenance engineer checks equipment, forecasts RUL, generates schedule, then queries results.

**Steps:**

1. **Detect Failure Risk**
   ```http
   POST /predict/classification
   {
     "model_name": "xgb_classifier",
     "product_id": "PROD_ALPHA",
     "unit_id": "UNIT_101"
   }
   ```
   - **Expected:** `failure_probability > 0.8` indicates high risk
   - **Assertion:** Response has `data.prediction == 1`

2. **Forecast RUL**
   ```http
   POST /predict/forecast
   {
     "model_name": "xgb_regressor",
     "product_id": "PROD_ALPHA",
     "unit_id": "UNIT_101",
     "horizon_steps": 24
   }
   ```
   - **Expected:** `current_rul_hours < 48` indicates critical state
   - **Assertion:** Forecast shows declining RUL over 24 steps

3. **Generate Maintenance Schedule**
   ```http
   POST /optimizer/schedule
   {
     "unit_list": [
       {"product_id": "PROD_ALPHA", "unit_id": "UNIT_101"}
     ],
     "risk_threshold": 0.7,
     "rul_threshold": 48.0
   }
   ```
   - **Expected:** Schedule created with `status: "PENDING"`
   - **Assertion:** `data.maintenance_scheduled == 1`
   - **Side-effect:** Row inserted into `maintenance_schedule` table

4. **Query Generated Schedule**
   ```http
   GET /maintenance/schedule?product_id=PROD_ALPHA&unit_id=UNIT_101&status=PENDING
   ```
   - **Expected:** Returns array with schedule entry from step 3
   - **Assertion:** 
     - `data[0].schedule_id` matches from step 3
     - `data[0].status == "PENDING"`
     - `data[0].risk_score` is high

**End-to-End Validation:**
- All 4 requests succeed (200 OK)
- Data flows correctly: classification → forecast → optimizer → query
- Database has persistent schedule entry
- Schedule entry contains correct timestamps, risk_score, and actions

---

### 3.2 Scenario B: Retrain → Promote → Reload → Re-infer

**User Journey:** ML engineer retrains model, promotes new version, reloads cache, and validates predictions.

**Steps:**

1. **Trigger Incremental Retrain**
   ```http
   POST /retrain/run
   {
     "model_type": "classification",
     "incremental": true
   }
   ```
   - **Expected:** New model artifact created
   - **Assertion:** 
     - Response contains new `version` (e.g., `20241119_153000`)
     - `data.path` points to new `.joblib` file
   - **Side-effect:** 
     - New file: `xgb_classifier_20241119_153000.joblib`
     - New row in `model_artifact` table
     - `retrain_pointer` updated

2. **Verify New Model Artifact**
   ```http
   GET /models
   ```
   - **Expected:** Latest model listed first for `xgb_classifier`
   - **Assertion:** 
     - `data[0].model_name == "xgb_classifier"`
     - `data[0].version == "20241119_153000"`
     - `data[0].promoted_at` is recent timestamp

3. **Reload Model Cache**
   ```http
   POST /models/reload
   ```
   - **Expected:** Cache cleared
   - **Assertion:** `data.cache_size_after == 0`
   - **Side-effect:** Next inference will load new model from disk

4. **Re-run Classification**
   ```http
   POST /predict/classification
   {
     "model_name": "xgb_classifier",
     "product_id": "PROD_ALPHA",
     "unit_id": "UNIT_101"
   }
   ```
   - **Expected:** Uses retrained model
   - **Assertion:** 
     - `data.model_version == "20241119_153000"` (new version)
     - Prediction completes successfully

**End-to-End Validation:**
- Retrain creates new artifact file on disk
- Database records new model version
- Cache reload triggers lazy reloading
- Next inference uses new model version
- Model version in response matches retrained version

---

### 3.3 Scenario C: Multi-Turn Copilot Conversation

**User Journey:** User has conversational interaction with copilot, asking follow-up questions.

**Steps:**

1. **Initial Question**
   ```http
   POST /copilot/chat
   {
     "messages": [
       {"role": "user", "content": "I'm worried about UNIT_202. Can you check it?"}
     ],
     "context": {"product_id": "PROD_BETA"}
   }
   ```
   - **Expected:** Copilot executes `predict_failure` tool
   - **Assertion:**
     - `data.tool_calls[0].name == "predict_failure"`
     - `data.tool_calls[0].arguments.unit_id == "UNIT_202"`
     - `data.reply` contains natural language summary of failure risk

2. **Follow-Up: RUL Question**
   ```http
   POST /copilot/chat
   {
     "messages": [
       {"role": "user", "content": "I'm worried about UNIT_202. Can you check it?"},
       {"role": "assistant", "content": "<previous_response>"},
       {"role": "user", "content": "How much time do we have?"}
     ],
     "context": {"product_id": "PROD_BETA"}
   }
   ```
   - **Expected:** Copilot infers context and executes `predict_rul` tool
   - **Assertion:**
     - `data.tool_calls[0].name == "predict_rul"`
     - `data.tool_calls[0].arguments.unit_id == "UNIT_202"` (inferred from context)
     - `data.reply` mentions RUL in hours

3. **Follow-Up: Schedule Request**
   ```http
   POST /copilot/chat
   {
     "messages": [
       {"role": "user", "content": "I'm worried about UNIT_202. Can you check it?"},
       {"role": "assistant", "content": "<response_1>"},
       {"role": "user", "content": "How much time do we have?"},
       {"role": "assistant", "content": "<response_2>"},
       {"role": "user", "content": "Schedule maintenance for it"}
     ],
     "context": {"product_id": "PROD_BETA"}
   }
   ```
   - **Expected:** Copilot executes `optimize_schedule` tool
   - **Assertion:**
     - `data.tool_calls[0].name == "optimize_schedule"`
     - `data.tool_calls[0].arguments.unit_list` contains `UNIT_202`
     - `data.reply` confirms schedule creation
   - **Side-effect:** Maintenance schedule created in DB

4. **Follow-Up: Confirmation**
   ```http
   POST /copilot/chat
   {
     "messages": [...all_previous_messages...],
     {"role": "user", "content": "Thanks! Show me the schedule details."}
   }
   ```
   - **Expected:** Copilot provides schedule information from previous tool result
   - **Assertion:**
     - Response includes schedule_id and recommended times
     - No new tool calls needed (uses cached result)

**End-to-End Validation:**
- Conversation context maintained across 4 turns
- Copilot correctly infers `unit_id` from previous messages
- Tools execute in logical sequence: detect → forecast → schedule
- Natural language responses are coherent and informative
- Final state: schedule exists in DB for UNIT_202

---

## 4. Automation Guidance (pytest)

### 4.1 Test File Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── test_health_api.py               # Health endpoint tests
├── test_classification_api.py       # Classification tests
├── test_forecast_api.py             # Forecast tests
├── test_optimizer_api.py            # Optimizer tests
├── test_maintenance_api.py          # Schedule query tests
├── test_models_mgmt_api.py          # Model management tests
├── test_retrain_api.py              # Retraining tests
├── test_copilot_tools_api.py        # Copilot tool endpoints
├── test_copilot_chat_api.py         # Copilot chat endpoint
└── test_e2e_scenarios.py            # Cross-flow scenarios
```

### 4.2 Using Existing Fixtures

**From `conftest.py`:**

```python
import pytest

def test_classification_with_fixtures(client, db_session, sample_telemetry, sample_model_artifact):
    """
    client: Flask test client
    db_session: SQLAlchemy session with automatic rollback
    sample_telemetry: 10 pre-seeded telemetry rows for PROD_001/UNIT_001
    sample_model_artifact: Pre-registered test_model artifact
    """
    
    # Use client to make HTTP requests
    response = client.post('/predict/classification', json={
        'model_name': 'test_model',
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001'
    })
    
    # Assert response
    assert response.status_code in [200, 404]
    data = response.json
    assert 'data' in data
    assert 'error' in data
    
    # Query DB to verify side-effects
    from app.models import Telemetry
    telemetry_count = db_session.query(Telemetry).filter(
        Telemetry.product_id == 'PROD_001'
    ).count()
    assert telemetry_count == 10
```

### 4.3 Mocking Gemini 2.5 Pro HTTP Client

**Strategy:** Use `pytest-mock` to mock `requests.post` in `copilot_service.py`.

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_gemini_response():
    """Return mock Gemini API response with tool call."""
    return {
        'candidates': [{
            'content': {
                'parts': [
                    {
                        'functionCall': {
                            'name': 'predict_failure',
                            'args': {
                                'product_id': 'PROD_001',
                                'unit_id': 'UNIT_001'
                            }
                        }
                    }
                ]
            }
        }]
    }

def test_copilot_chat_with_tool_call(client, mocker, mock_gemini_response, sample_telemetry):
    """Test copilot executes tool and returns response."""
    
    # Mock requests.post to return predefined response
    mock_post = mocker.patch('app.services.copilot_service.requests.post')
    mock_response = Mock()
    mock_response.json.return_value = mock_gemini_response
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    # Make request
    response = client.post('/copilot/chat', json={
        'messages': [
            {'role': 'user', 'content': 'Check UNIT_001'}
        ],
        'context': {'product_id': 'PROD_001'}
    })
    
    # Assertions
    assert response.status_code == 200
    data = response.json['data']
    assert len(data['tool_calls']) == 1
    assert data['tool_calls'][0]['name'] == 'predict_failure'
    assert data['tool_results'][0]['success'] in [True, False]
```

**Alternative: Mock at Service Layer**

```python
def test_copilot_service_unit(mocker):
    """Test copilot service independently."""
    from app.services.copilot_service import GeminiCopilotClient
    
    client = GeminiCopilotClient()
    
    # Mock HTTP call
    mocker.patch.object(client, 'chat', return_value={
        'candidates': [{'content': {'parts': [{'text': 'Test response'}]}}]
    })
    
    response = client.chat([{'role': 'user', 'content': 'Test'}])
    assert 'candidates' in response
```

### 4.4 Integration Test Pattern

```python
def test_integration_classification_to_optimizer(client, db_session, sample_telemetry):
    """E2E test: Classification → Forecast → Optimizer."""
    
    # Step 1: Classify
    response1 = client.post('/predict/classification', json={
        'model_name': 'xgb_classifier',
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001'
    })
    assert response1.status_code in [200, 404]
    
    # Step 2: Forecast
    response2 = client.post('/predict/forecast', json={
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001',
        'horizon_steps': 10
    })
    assert response2.status_code in [200, 404]
    
    # Step 3: Optimize
    response3 = client.post('/optimizer/schedule', json={
        'unit_list': [
            {'product_id': 'PROD_001', 'unit_id': 'UNIT_001'}
        ]
    })
    assert response3.status_code in [200, 400]
    
    # Verify DB side-effects
    from app.models import MaintenanceSchedule
    schedules = db_session.query(MaintenanceSchedule).filter(
        MaintenanceSchedule.product_id == 'PROD_001'
    ).all()
    
    # May or may not create schedule depending on risk
    assert len(schedules) >= 0
```

### 4.5 Parameterized Testing

```python
@pytest.mark.parametrize("product_id,unit_id,expected_status", [
    ("PROD_001", "UNIT_001", [200, 404]),
    ("PROD_002", "UNIT_002", [200, 404]),
    ("NONEXISTENT", "NONEXISTENT", [404]),
])
def test_classification_multiple_units(client, sample_telemetry, product_id, unit_id, expected_status):
    """Test classification for multiple units."""
    response = client.post('/predict/classification', json={
        'model_name': 'xgb_classifier',
        'product_id': product_id,
        'unit_id': unit_id
    })
    
    assert response.status_code in expected_status
```

### 4.6 Database Isolation

**Ensure each test gets clean DB state:**

```python
@pytest.fixture(scope='function')
def db_session(test_db):
    """Create database session for tests with automatic rollback."""
    Session = sessionmaker(bind=test_db)
    session = Session()
    
    yield session
    
    # Rollback all changes after test
    session.rollback()
    session.close()
```

**Key principle:** Use `scope='function'` for `db_session` to ensure isolation.

### 4.7 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_classification_api.py -v

# Run specific test
pytest tests/test_classification_api.py::test_predict_classification_success -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run only E2E scenarios
pytest tests/test_e2e_scenarios.py -v

# Run with markers
pytest -m "integration" tests/

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

### 4.8 Continuous Integration Setup

**Example GitHub Actions workflow:**

```yaml
name: Backend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd pm-app
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock
    
    - name: Run tests
      env:
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        cd pm-app
        pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./pm-app/coverage.xml
```

### 4.9 Test Data Management

**Create reusable test data builders:**

```python
# tests/builders.py
from datetime import datetime, timezone
from app.models import Telemetry

class TelemetryBuilder:
    def __init__(self):
        self.data = {
            'product_id': 'PROD_001',
            'unit_id': 'UNIT_001',
            'timestamp_ts': datetime.now(timezone.utc),
            'air_temperature_K': 300.0,
            'process_temperature_K': 310.0,
            'rotational_speed_rpm': 1500.0,
            'torque_Nm': 40.0,
            'tool_wear_min': 100.0,
            'is_failure': 0,
            'synthetic_RUL': 100.0
        }
    
    def with_product(self, product_id):
        self.data['product_id'] = product_id
        return self
    
    def with_failure(self):
        self.data['is_failure'] = 1
        self.data['failure_type'] = 'Heat Dissipation Failure'
        return self
    
    def build(self):
        return Telemetry(**self.data)

# Usage in tests
def test_with_builder(db_session):
    telemetry = TelemetryBuilder().with_product('PROD_002').with_failure().build()
    db_session.add(telemetry)
    db_session.commit()
```

---

## Appendix: Test Coverage Checklist

### Endpoints Coverage

- [ ] `GET /health` - Healthy system
- [ ] `GET /health` - Unhealthy DB
- [ ] `POST /predict/classification` - Happy path
- [ ] `POST /predict/classification` - Explicit features
- [ ] `POST /predict/classification` - Missing fields
- [ ] `POST /predict/classification` - Invalid timestamp
- [ ] `POST /predict/classification` - No telemetry
- [ ] `POST /predict/forecast` - Happy path
- [ ] `POST /predict/forecast` - Missing fields
- [ ] `POST /predict/forecast` - Insufficient RUL history
- [ ] `POST /optimizer/schedule` - Single unit
- [ ] `POST /optimizer/schedule` - Multi-unit
- [ ] `POST /optimizer/schedule` - No risk units
- [ ] `POST /optimizer/schedule` - Missing unit_list
- [ ] `POST /optimizer/schedule` - Invalid unit format
- [ ] `GET /maintenance/schedule` - All schedules
- [ ] `GET /maintenance/schedule` - Filter by product/unit
- [ ] `GET /maintenance/schedule` - Filter by status
- [ ] `GET /maintenance/schedule` - Filter by date range
- [ ] `GET /models` - List artifacts
- [ ] `POST /models/reload` - Clear cache
- [ ] `POST /retrain/run` - Incremental classification
- [ ] `POST /retrain/run` - Full forecast
- [ ] `POST /retrain/run` - No new data
- [ ] `POST /retrain/run` - Invalid model type
- [ ] `POST /copilot/tools/predict_failure` - Success
- [ ] `POST /copilot/tools/predict_rul` - Success
- [ ] `POST /copilot/tools/optimize_schedule` - Success
- [ ] `POST /copilot/chat` - Simple query with tool
- [ ] `POST /copilot/chat` - Missing parameters
- [ ] `POST /copilot/chat` - Tool failure fallback
- [ ] `POST /copilot/chat` - Multi-turn conversation
- [ ] `POST /copilot/chat` - Missing API key

### E2E Scenarios Coverage

- [ ] Scenario A: Detect → Forecast → Schedule → Query
- [ ] Scenario B: Retrain → Promote → Reload → Re-infer
- [ ] Scenario C: Multi-turn copilot conversation

---

**End of E2E Validation Plan**
