import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from app.services.classification_service import predict_failure

logger = logging.getLogger(__name__)

classification_bp = Blueprint('classification', __name__)


def parse_timestamp(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {ts_str}")


@classification_bp.route('/predict/classification', methods=['POST'])
def predict_classification():
    """
    POST /predict/classification
    
    Body:
        model_name (str): Model identifier
        product_id (str): Product ID
        unit_id (str): Unit ID
        timestamp_before (str, optional): ISO timestamp
        features (dict, optional): Explicit feature values
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        model_name = data.get('model_name')
        product_id = data.get('product_id')
        unit_id = data.get('unit_id')
        
        if not all([model_name, product_id, unit_id]):
            return jsonify({
                'data': None,
                'error': 'Missing required fields: model_name, product_id, unit_id'
            }), 400
        
        timestamp_before = parse_timestamp(data.get('timestamp_before'))
        features = data.get('features')
        
        result = predict_failure(
            model_name=model_name,
            product_id=product_id,
            unit_id=unit_id,
            timestamp_before=timestamp_before,
            features=features
        )
        
        return jsonify({'data': result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Classification request failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 404
    except FileNotFoundError as e:
        logger.warning(f"Model file not found: {e}")
        return jsonify({'data': None, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500