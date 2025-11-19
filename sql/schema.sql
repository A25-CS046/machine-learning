-- This schema is kept in sync with app/models.py (v2 architecture).

DROP TABLE IF EXISTS maintenance_schedule CASCADE;
DROP TABLE IF EXISTS model_artifact CASCADE;
DROP TABLE IF EXISTS retrain_pointer CASCADE;
DROP TABLE IF EXISTS telemetry CASCADE;

CREATE TABLE telemetry (
  id BIGSERIAL PRIMARY KEY,
  product_id TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  timestamp_ts TIMESTAMPTZ NOT NULL,
  step_index INTEGER,
  engine_type TEXT,
  air_temperature_K DOUBLE PRECISION,
  process_temperature_K DOUBLE PRECISION,
  rotational_speed_rpm DOUBLE PRECISION,
  torque_Nm DOUBLE PRECISION,
  tool_wear_min DOUBLE PRECISION,
  is_failure INTEGER,
  failure_type TEXT,
  synthetic_RUL DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS telemetry_unit_time_idx 
  ON telemetry (product_id, unit_id, timestamp_ts DESC);
CREATE INDEX IF NOT EXISTS telemetry_failure_idx 
  ON telemetry (is_failure);

CREATE TABLE retrain_pointer (
  model_name TEXT PRIMARY KEY,
  last_retrain_ts TIMESTAMPTZ DEFAULT '1970-01-01T00:00:00Z',
  last_retrain_id BIGINT DEFAULT 0
);

CREATE TABLE model_artifact (
  id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  path TEXT NOT NULL,
  metadata JSONB,
  metrics JSONB,
  promoted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS model_artifact_idx 
  ON model_artifact(model_name, promoted_at DESC);

CREATE TABLE maintenance_schedule (
  id BIGSERIAL PRIMARY KEY,
  schedule_id TEXT NOT NULL,
  product_id TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  recommended_start TIMESTAMPTZ NOT NULL,
  recommended_end TIMESTAMPTZ NOT NULL,
  reason TEXT NOT NULL,
  risk_score DOUBLE PRECISION,
  model_version TEXT,
  actions JSONB,
  constraints_applied JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  status TEXT DEFAULT 'PENDING'
);

CREATE INDEX IF NOT EXISTS maintenance_schedule_unit_idx 
  ON maintenance_schedule (product_id, unit_id, recommended_start);