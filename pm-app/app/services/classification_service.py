import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import and_
from app.db import get_db
from app.models import Telemetry
from app.services.models_loader import get_models_loader

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    'air_temperature_K', 'process_temperature_K', 'rotational_speed_rpm',
    'torque_Nm', 'tool_wear_min'
]


def aggregate_features_from_telemetry(telemetry_rows: list[Telemetry], encoder: Any) -> np.ndarray:
    """
    Aggregate per-machine features from telemetry history.
    Returns a feature vector matching the XGBoost model's expected input.
    """
    if not telemetry_rows:
        raise ValueError("No telemetry data available for feature aggregation")
    
    df = pd.DataFrame([{
        'air_temperature_K': t.air_temperature_K,
        'process_temperature_K': t.process_temperature_K,
        'rotational_speed_rpm': t.rotational_speed_rpm,
        'torque_Nm': t.torque_Nm,
        'tool_wear_min': t.tool_wear_min,
        'engine_type': t.engine_type,
        'timestamp': t.timestamp,
    } for t in telemetry_rows])
    
    feature_dict = {}
    
    for col in FEATURE_COLUMNS:
        feature_dict[f'{col}_mean'] = df[col].mean()
        feature_dict[f'{col}_std'] = df[col].std()
        feature_dict[f'{col}_min'] = df[col].min()
        feature_dict[f'{col}_max'] = df[col].max()
        feature_dict[f'{col}_last'] = df[col].iloc[-1]
        
        if len(df) > 1:
            x = np.arange(len(df))
            y = df[col].values
            trend = np.polyfit(x, y, 1)[0]
        else:
            trend = 0.0
        feature_dict[f'{col}_trend'] = trend
    
    if telemetry_rows[-1].engine_type:
        try:
            engine_encoded = encoder.transform([telemetry_rows[-1].engine_type])[0]
        except:
            engine_encoded = 0
    else:
        engine_encoded = 0
    
    feature_dict['engine_type_encoded'] = engine_encoded
    
    features_ordered = []
    for col in FEATURE_COLUMNS:
        features_ordered.extend([
            feature_dict[f'{col}_mean'],
            feature_dict[f'{col}_std'],
            feature_dict[f'{col}_min'],
            feature_dict[f'{col}_max'],
            feature_dict[f'{col}_last'],
            feature_dict[f'{col}_trend'],
        ])
    features_ordered.append(feature_dict['engine_type_encoded'])
    
    return np.array(features_ordered).reshape(1, -1)


def predict_failure(
    model_name: str,
    product_id: str,
    unit_id: str,
    timestamp_before: datetime | None = None,
    features: dict | None = None
) -> dict:
    """
    Predict equipment failure probability using XGBoost classifier.
    
    If features is None, fetches latest telemetry history for (product_id, unit_id)
    and aggregates features. Otherwise uses provided features directly.
    """
    loader = get_models_loader()
    
    try:
        model, artifact = loader.get_classification_model(model_name)
        scaler = loader.get_scaler()
        encoder = loader.get_encoder()
    except ValueError as e:
        logger.error(f"Model loading failed: {e}")
        raise ValueError(f"Model not available: {model_name}") from e
    
    if features is None:
        if timestamp_before is None:
            timestamp_before = datetime.now(timezone.utc)
        
        timestamp_before_str = timestamp_before.isoformat()
        
        with get_db() as session:
            from sqlalchemy import text
            
            query = session.query(Telemetry).filter(
                and_(
                    Telemetry.product_id == product_id,
                    Telemetry.unit_id == unit_id,
                    text("timestamp::timestamptz <= :ts_before")
                )
            ).params(ts_before=timestamp_before_str).order_by(
                text("timestamp::timestamptz DESC")
            ).limit(50)
            
            telemetry_rows = query.all()
            
            if not telemetry_rows:
                raise ValueError(f"No telemetry found for product_id={product_id}, unit_id={unit_id}")
            
            telemetry_rows = list(reversed(telemetry_rows))
        
        feature_vector = aggregate_features_from_telemetry(telemetry_rows, encoder)
        inputs_used = {
            'source': 'telemetry',
            'product_id': product_id,
            'unit_id': unit_id,
            'timestamp_before': timestamp_before.isoformat(),
            'rows_used': len(telemetry_rows)
        }
    else:
        feature_list = []
        for col in FEATURE_COLUMNS:
            feature_list.extend([
                features.get(f'{col}_mean', 0),
                features.get(f'{col}_std', 0),
                features.get(f'{col}_min', 0),
                features.get(f'{col}_max', 0),
                features.get(f'{col}_last', 0),
                features.get(f'{col}_trend', 0),
            ])
        feature_list.append(features.get('engine_type_encoded', 0))
        feature_vector = np.array(feature_list).reshape(1, -1)
        inputs_used = {'source': 'explicit_features', 'features': features}
    
    try:
        feature_vector_scaled = scaler.transform(feature_vector)
    except:
        feature_vector_scaled = feature_vector
    
    prediction = model.predict(feature_vector_scaled)[0]
    probabilities = model.predict_proba(feature_vector_scaled)[0]
    
    failure_prob = float(probabilities[1])
    predicted_failure = int(prediction)
    
    if predicted_failure == 1:
        failure_type = "Equipment Failure Predicted"
        recommendation = "Schedule immediate inspection and maintenance"
    else:
        failure_type = None
        recommendation = "Equipment operating normally"
    
    return {
        'prediction': predicted_failure,
        'failure_type': failure_type,
        'probabilities': {
            'no_failure': float(probabilities[0]),
            'failure': failure_prob
        },
        'model_name': model_name,
        'model_version': artifact.version,
        'inputs_used': inputs_used,
        'fallback_recommendation': recommendation
    }
