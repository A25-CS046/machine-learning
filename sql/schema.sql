DROP TABLE IF EXISTS telemetry CASCADE;
DROP TABLE IF EXISTS retrain_pointer;
DROP TABLE IF EXISTS model_artifact;

CREATE TABLE telemetry (
  product_id TEXT,
  unit_id TEXT,
  timestamp TEXT,
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

CREATE INDEX IF NOT EXISTS telemetry_unit_idx ON telemetry(unit_id);
CREATE INDEX IF NOT EXISTS telemetry_failure_idx ON telemetry(is_failure);

CREATE TABLE retrain_pointer (
  model_name TEXT PRIMARY KEY,
  last_retrain_ts TIMESTAMPTZ DEFAULT '1970-01-01T00:00:00Z',
  last_retrain_id BIGINT DEFAULT 0
);

CREATE TABLE model_artifact (
  id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  metadata JSONB,
  promoted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS model_artifact_idx 
  ON model_artifact(model_name, promoted_at DESC);