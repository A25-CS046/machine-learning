import logging
from flask import Blueprint, request, jsonify
from app.services.copilot_service import get_gemini_client
from app.services.classification_service import predict_failure
from app.services.forecast_service import predict_rul
from app.services.optimizer_service import optimize_maintenance_schedule

logger = logging.getLogger(__name__)

copilot_bp = Blueprint('copilot', __name__)


@copilot_bp.route('/copilot/tools/predict_failure', methods=['POST'])
def tool_predict_failure():
    """
    POST /copilot/tools/predict_failure
    
    Simplified tool endpoint for LLM copilot integration.
    
    Body:
        product_id (str): Product ID
        unit_id (str): Unit ID
        horizon_hours (int, optional): Prediction horizon (default: 24)
    """
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
    """
    POST /copilot/tools/predict_rul
    
    Simplified tool endpoint for RUL prediction.
    
    Body:
        product_id (str): Product ID
        unit_id (str): Unit ID
        horizon_steps (int, optional): Forecast steps (default: 10)
    """
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
    """
    POST /copilot/tools/optimize_schedule
    
    Simplified tool endpoint for maintenance optimization.
    
    Body:
        unit_list (list): List of {product_id, unit_id} dicts
        risk_threshold (float, optional): Default 0.7
        rul_threshold (float, optional): Default 24.0
        horizon_days (int, optional): Default 7
    """
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
    """
    POST /copilot/chat
    
    Main copilot conversational endpoint with Gemini 2.5 Pro integration.
    
    Body:
        messages (list): Conversation history [{role: str, content: str}, ...]
        context (dict, optional): Additional context (product_id, unit_id, etc.)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        messages = data.get('messages', [])
        context = data.get('context', {})
        
        if not messages:
            return jsonify({
                'data': None,
                'error': 'messages list is required'
            }), 400
        
        client = get_gemini_client()
        
        gemini_response = client.chat(messages, enable_tools=True)
        
        tool_calls = client.extract_tool_calls(gemini_response)
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                if tool_name == 'predict_failure':
                    tool_result = predict_failure(
                        model_name='xgb_classifier',
                        product_id=tool_args['product_id'],
                        unit_id=tool_args['unit_id']
                    )
                elif tool_name == 'predict_rul':
                    tool_result = predict_rul(
                        model_name='xgb_regressor',
                        product_id=tool_args['product_id'],
                        unit_id=tool_args['unit_id'],
                        horizon_steps=tool_args.get('horizon_steps', 10)
                    )
                elif tool_name == 'optimize_schedule':
                    tool_result = optimize_maintenance_schedule(
                        unit_list=tool_args['unit_list'],
                        risk_threshold=tool_args.get('risk_threshold', 0.7),
                        rul_threshold=tool_args.get('rul_threshold', 24.0),
                        horizon_days=tool_args.get('horizon_days', 7)
                    )
                else:
                    tool_result = {'error': f'Unknown tool: {tool_name}'}
                
                tool_results.append({
                    'tool_name': tool_name,
                    'success': True,
                    'result': tool_result
                })
            
            except Exception as e:
                logger.error(f"Tool execution failed: {tool_name} - {e}")
                tool_results.append({
                    'tool_name': tool_name,
                    'success': False,
                    'error': str(e)
                })
        
        text_reply = client.extract_text_response(gemini_response)
        
        result = {
            'reply': text_reply or "I've processed your request using the available tools.",
            'tool_calls': tool_calls,
            'tool_results': tool_results,
            'raw_model_response': gemini_response
        }
        
        return jsonify({'data': result, 'error': None}), 200
        
    except RuntimeError as e:
        logger.error(f"Gemini API error: {e}")
        return jsonify({'data': None, 'error': str(e)}), 503
    except Exception as e:
        logger.error(f"Copilot chat error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500
