import logging
from flask import Blueprint, request, jsonify
from app.routes.classification import parse_timestamp
from app.services.optimizer_service import optimize_maintenance_schedule

logger = logging.getLogger(__name__)

optimizer_bp = Blueprint('optimizer', __name__)


@optimizer_bp.route('/optimizer/schedule', methods=['POST'])
def create_optimized_schedule():
    """
    POST /optimizer/schedule
    
    Body:
        unit_list (list): List of {product_id, unit_id} dicts
        risk_threshold (float, optional): Failure probability threshold (default: 0.7)
        rul_threshold (float, optional): RUL hours threshold (default: 24.0)
        horizon_days (int, optional): Planning horizon in days (default: 7)
        teams_available (int, optional): Number of maintenance teams (default: 2)
        hours_per_day (int, optional): Work hours per team per day (default: 8)
        earliest_allowed (str, optional): ISO timestamp for earliest maintenance start
        latest_allowed (str, optional): ISO timestamp for latest maintenance end
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        unit_list = data.get('unit_list', [])
        
        if not unit_list or not isinstance(unit_list, list):
            return jsonify({
                'data': None,
                'error': 'unit_list is required and must be a non-empty list'
            }), 400
        
        for unit in unit_list:
            if 'product_id' not in unit or 'unit_id' not in unit:
                return jsonify({
                    'data': None,
                    'error': 'Each unit must have product_id and unit_id'
                }), 400
        
        risk_threshold = data.get('risk_threshold', 0.7)
        rul_threshold = data.get('rul_threshold', 24.0)
        horizon_days = data.get('horizon_days', 7)
        teams_available = data.get('teams_available', 2)
        hours_per_day = data.get('hours_per_day', 8)
        
        earliest_allowed = parse_timestamp(data.get('earliest_allowed'))
        latest_allowed = parse_timestamp(data.get('latest_allowed'))
        
        result = optimize_maintenance_schedule(
            unit_list=unit_list,
            risk_threshold=risk_threshold,
            rul_threshold=rul_threshold,
            horizon_days=horizon_days,
            teams_available=teams_available,
            hours_per_day=hours_per_day,
            earliest_allowed=earliest_allowed,
            latest_allowed=latest_allowed
        )
        
        return jsonify({'data': result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Optimizer request failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Optimizer error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500
