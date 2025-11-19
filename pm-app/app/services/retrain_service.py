import logging
import os
from datetime import datetime, timezone
from typing import Literal
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sqlalchemy import and_
from app.db import get_db
from app.models import Telemetry, RetrainPointer, ModelArtifact
from app.config import load_config
from app.services.classification_service import FEATURE_COLUMNS, aggregate_features_from_telemetry

logger = logging.getLogger(__name__)


def retrain_model(
    model_type: Literal['classification', 'forecast'],
    incremental: bool = True,
    sample_limit: int | None = None
) -> dict:
    """
    Retrain XGBoost model using telemetry data.
    Supports incremental training from last retrain checkpoint.
    """
    config = load_config()
    model_name = 'xgb_classifier' if model_type == 'classification' else 'xgb_regressor'
    
    with get_db() as session:
        pointer = session.query(RetrainPointer).filter(
            RetrainPointer.model_name == model_name
        ).first()
        
        if pointer is None:
            pointer = RetrainPointer(
                model_name=model_name,
                last_retrain_ts=datetime(1970, 1, 1, tzinfo=timezone.utc),
                last_retrain_id=0
            )
            session.add(pointer)
            session.commit()
        
        last_retrain_ts = pointer.last_retrain_ts
        last_retrain_id = pointer.last_retrain_id
        
        if incremental:
            query = session.query(Telemetry).filter(
                and_(
                    Telemetry.timestamp_ts > last_retrain_ts,
                    Telemetry.id > last_retrain_id
                )
            )
        else:
            query = session.query(Telemetry)
        
        if model_type == 'forecast':
            query = query.filter(Telemetry.synthetic_RUL.isnot(None))
        
        query = query.order_by(Telemetry.timestamp_ts)
        
        if sample_limit:
            query = query.limit(sample_limit)
        
        telemetry_data = query.all()
        
        if not telemetry_data:
            raise ValueError("No new telemetry data available for retraining")
        
        logger.info(f"Found {len(telemetry_data)} rows for retraining {model_name}")
        
        units = {}
        for row in telemetry_data:
            key = (row.product_id, row.unit_id)
            if key not in units:
                units[key] = []
            units[key].append(row)
        
        try:
            scaler = joblib.load(os.path.join(config.model_storage.local_path, 'scaler.joblib'))
            encoder = joblib.load(os.path.join(config.model_storage.local_path, 'encoder_engine_type.joblib'))
        except FileNotFoundError as e:
            logger.error(f"Preprocessing artifacts not found: {e}")
            raise ValueError("Scaler or encoder not found. Cannot retrain.")
        
        X_list = []
        y_list = []
        
        for unit_rows in units.values():
            if len(unit_rows) < 5:
                continue
            
            feature_vector = aggregate_features_from_telemetry(unit_rows, encoder)
            
            if model_type == 'classification':
                label = 1 if any(r.is_failure == 1 for r in unit_rows) else 0
            else:
                label = unit_rows[-1].synthetic_RUL if unit_rows[-1].synthetic_RUL else 0.0
            
            X_list.append(feature_vector[0])
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Training dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        X_scaled = scaler.transform(X)
        
        if model_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        model.fit(X_scaled, y)
        
        version = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        
        if config.model_storage.backend == 'local':
            model_filename = f"{model_name}_{version}.joblib"
            model_path = os.path.join(config.model_storage.local_path, model_filename)
            joblib.dump(model, model_path)
            model_uri = f"file://{model_path}"
        else:
            raise NotImplementedError("GCS storage not yet implemented for retraining")
        
        if model_type == 'classification':
            from sklearn.metrics import accuracy_score, recall_score
            y_pred = model.predict(X_scaled)
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'samples': int(len(y))
            }
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_pred = model.predict(X_scaled)
            metrics = {
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred)),
                'samples': int(len(y))
            }
        
        artifact = ModelArtifact(
            model_name=model_name,
            version=version,
            path=model_uri,
            model_metadata={
                'model_type': model_type,
                'incremental': incremental,
                'training_samples': len(y)
            },
            metrics=metrics
        )
        session.add(artifact)
        
        last_row = telemetry_data[-1]
        pointer.last_retrain_ts = last_row.timestamp_ts
        pointer.last_retrain_id = last_row.id
        
        session.commit()
        
        logger.info(f"Model {model_name} v{version} trained and promoted")
        
        return {
            'model_name': model_name,
            'version': version,
            'path': model_uri,
            'metrics': metrics,
            'training_samples': len(y),
            'incremental': incremental
        }
