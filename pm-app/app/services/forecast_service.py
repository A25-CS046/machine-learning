import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Any
from sqlalchemy import and_
from app.db import get_db
from app.models import Telemetry
from app.services.models_loader import get_models_loader
from app.services.classification_service import aggregate_features_from_telemetry

logger = logging.getLogger(__name__)


def predict_rul(
    model_name: str,
    product_id: str,
    unit_id: str,
    horizon_steps: int = 10,
    timestamp_before: datetime | None = None
) -> dict:
    """
    Predict Remaining Useful Life (RUL) for equipment using XGBoost regressor.
    Returns forecast of RUL over horizon_steps future timesteps.
    """
    loader = get_models_loader()
    
    try:
        model, artifact = loader.get_forecast_model(model_name)
        scaler = loader.get_scaler()
        encoder = loader.get_encoder()
    except ValueError as e:
        logger.error(f"Model loading failed: {e}")
        raise ValueError(f"Model not available: {model_name}") from e
    
    if timestamp_before is None:
        timestamp_before = datetime.now(timezone.utc)
    
    with get_db() as session:
        telemetry_rows = session.query(Telemetry).filter(
            and_(
                Telemetry.product_id == product_id,
                Telemetry.unit_id == unit_id,
                Telemetry.timestamp_ts <= timestamp_before,
                Telemetry.synthetic_RUL.isnot(None)
            )
        ).order_by(Telemetry.timestamp_ts.desc()).limit(50).all()
        
        if not telemetry_rows:
            raise ValueError(f"No RUL telemetry found for product_id={product_id}, unit_id={unit_id}")
        
        telemetry_rows = list(reversed(telemetry_rows))
    
    feature_vector = aggregate_features_from_telemetry(telemetry_rows, encoder)
    
    try:
        feature_vector_scaled = scaler.transform(feature_vector)
    except:
        feature_vector_scaled = feature_vector
    
    current_rul = model.predict(feature_vector_scaled)[0]
    
    last_timestamp = telemetry_rows[-1].timestamp_ts
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
    
    forecast_steps = []
    for step in range(1, horizon_steps + 1):
        future_timestamp = last_timestamp + timedelta(hours=step)
        predicted_rul = max(0, current_rul - step)
        
        forecast_steps.append({
            'step': step,
            'timestamp': future_timestamp.isoformat(),
            'predicted_rul_hours': float(predicted_rul),
            'confidence': 'medium'
        })
    
    return {
        'product_id': product_id,
        'unit_id': unit_id,
        'current_rul_hours': float(current_rul),
        'forecast_horizon_steps': horizon_steps,
        'forecast': forecast_steps,
        'model_name': model_name,
        'model_version': artifact.version,
        'baseline_timestamp': last_timestamp.isoformat(),
        'telemetry_rows_used': len(telemetry_rows)
    }
