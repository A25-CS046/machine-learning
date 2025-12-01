"""Copilot API endpoints for conversational maintenance queries."""

import logging
from flask import Blueprint, request, jsonify
from app.services.langchain_agent_service import get_agent_service
from app.services.classification_service import predict_failure
from app.services.forecast_service import predict_rul
from app.services.optimizer_service import optimize_maintenance_schedule

logger = logging.getLogger(__name__)
copilot_bp = Blueprint('copilot', __name__)


@copilot_bp.route('/copilot/tools/predict_failure', methods=['POST'])
def tool_predict_failure():
    """Predict failure probability. Body: {product_id, unit_id, horizon_hours?}"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        product_id = data.get('product_id')
        unit_id = data.get('unit_id')
        
        if not all([product_id, unit_id]):
            return jsonify({
                'data': None,
                'error': 'product_id and unit_id are required'
            }), 400
        
        result = predict_failure(
            model_name='xgb_classifier',
            product_id=product_id,
            unit_id=unit_id
        )
        
        simplified_result = {
            'product_id': product_id,
            'unit_id': unit_id,
            'failure_probability': result['probabilities']['failure'],
            'will_fail': result['prediction'] == 1,
            'failure_type': result['failure_type'],
            'recommendation': result['fallback_recommendation']
        }
        
        return jsonify({'data': simplified_result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Tool predict_failure failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Tool predict_failure error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500


@copilot_bp.route('/copilot/tools/predict_rul', methods=['POST'])
def tool_predict_rul():
    """Predict remaining useful life. Body: {product_id, unit_id, horizon_steps?}"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        product_id = data.get('product_id')
        unit_id = data.get('unit_id')
        horizon_steps = data.get('horizon_steps', 10)
        
        if not all([product_id, unit_id]):
            return jsonify({
                'data': None,
                'error': 'product_id and unit_id are required'
            }), 400
        
        result = predict_rul(
            model_name='xgb_regressor',
            product_id=product_id,
            unit_id=unit_id,
            horizon_steps=horizon_steps
        )
        
        simplified_result = {
            'product_id': product_id,
            'unit_id': unit_id,
            'current_rul_hours': result['current_rul_hours'],
            'forecast_horizon_steps': result['forecast_horizon_steps'],
            'critical': result['current_rul_hours'] < 24,
            'recommendation': 'Schedule immediate maintenance' if result['current_rul_hours'] < 24 else 'Monitor equipment'
        }
        
        return jsonify({'data': simplified_result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Tool predict_rul failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Tool predict_rul error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500


@copilot_bp.route('/copilot/tools/optimize_schedule', methods=['POST'])
def tool_optimize_schedule():
    """Generate optimized maintenance schedule. Body: {unit_list, risk_threshold?, rul_threshold?, horizon_days?}"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        unit_list = data.get('unit_list', [])
        
        if not unit_list:
            return jsonify({
                'data': None,
                'error': 'unit_list is required and must be non-empty'
            }), 400
        
        result = optimize_maintenance_schedule(
            unit_list=unit_list,
            risk_threshold=data.get('risk_threshold', 0.7),
            rul_threshold=data.get('rul_threshold', 24.0),
            horizon_days=data.get('horizon_days', 7)
        )
        
        simplified_result = {
            'schedule_id': result['schedule_id'],
            'units_scheduled': result['maintenance_scheduled'],
            'high_risk_units': result['high_risk_units_found'],
            'recommendations': result['recommendations']
        }
        
        return jsonify({'data': simplified_result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Tool optimize_schedule failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Tool optimize_schedule error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500


@copilot_bp.route('/copilot/chat', methods=['POST'])
def copilot_chat():
    """Main copilot endpoint. Body: {messages: [{role, content}], session_id?, context?}"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        messages = data.get('messages', [])
        context = data.get('context', {})
        session_id = data.get('session_id') or request.headers.get('X-Session-ID', 'default')
        
        if not messages:
            return jsonify({
                'data': None,
                'error': 'messages list is required'
            }), 400
        
        # Extract last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content')
                break
        
        if not user_message:
            return jsonify({
                'data': None,
                'error': 'No user message found in messages list'
            }), 400
        
        # Get agent service and invoke
        agent_service = get_agent_service()
        result = agent_service.invoke_agent(
            session_id=session_id,
            user_message=user_message,
            context=context
        )
        
        if result['success']:
            # Format successful response
            response_data = {
                'reply': result['reply'],
                'session_id': result['session_id'],
                'execution_time_seconds': result['execution_time_seconds'],
                'timestamp': result['timestamp'],
                'intermediate_steps': [
                    {
                        'tool': step[0].tool if hasattr(step[0], 'tool') else 'unknown',
                        'tool_input': step[0].tool_input if hasattr(step[0], 'tool_input') else {},
                        'output': str(step[1])[:500]  # Truncate long outputs
                    }
                    for step in result.get('intermediate_steps', [])
                ]
            }
            
            return jsonify({'data': response_data, 'error': None}), 200
        else:
            # Format error response
            return jsonify({
                'data': None,
                'error': result.get('error', 'Agent execution failed'),
                'error_type': result.get('error_type', 'unknown')
            }), 500
        
    except Exception as e:
        logger.error(f"Copilot chat error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500


@copilot_bp.route('/copilot/session/<session_id>', methods=['DELETE'])
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        agent_service = get_agent_service()
        success = agent_service.clear_session(session_id)
        
        if success:
            return jsonify({
                'data': {'session_id': session_id, 'cleared': True},
                'error': None
            }), 200
        else:
            return jsonify({
                'data': None,
                'error': 'Failed to clear session'
            }), 500
    
    except Exception as e:
        logger.error(f"Clear session error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': str(e)}), 500
