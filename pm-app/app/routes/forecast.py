import logging
from flask import Blueprint, request, jsonify
from app.routes.classification import parse_timestamp
from app.services.forecast_service import predict_rul

logger = logging.getLogger(__name__)

forecast_bp = Blueprint('forecast', __name__)


@forecast_bp.route('/predict/forecast', methods=['POST'])
def predict_forecast():
    """
    POST /predict/forecast
    
    Body:
        model_name (str): Model identifier
        product_id (str): Product ID
        unit_id (str): Unit ID
        horizon_steps (int, optional): Number of future steps to forecast
        timestamp_before (str, optional): ISO timestamp
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        model_name = data.get('model_name', 'xgb_regressor')
        product_id = data.get('product_id')
        unit_id = data.get('unit_id')
        
        if not all([product_id, unit_id]):
            return jsonify({
                'data': None,
                'error': 'Missing required fields: product_id, unit_id'
            }), 400
        
        horizon_steps = data.get('horizon_steps', 10)
        timestamp_before = parse_timestamp(data.get('timestamp_before'))
        
        result = predict_rul(
            model_name=model_name,
            product_id=product_id,
            unit_id=unit_id,
            horizon_steps=horizon_steps,
            timestamp_before=timestamp_before
        )
        
        return jsonify({'data': result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Forecast request failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500
