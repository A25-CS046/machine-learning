import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import desc
from app.db import get_db
from app.models import ModelArtifact
from app.services.models_loader import get_models_loader
from app.services.retrain_service import retrain_model

logger = logging.getLogger(__name__)

models_mgmt_bp = Blueprint('models_mgmt', __name__)


@models_mgmt_bp.route('/models', methods=['GET'])
def list_models():
    """
    GET /models
    
    Returns latest model artifact for each model_name.
    """
    try:
        with get_db() as session:
            subquery = session.query(
                ModelArtifact.model_name,
                desc(ModelArtifact.promoted_at).label('max_promoted_at')
            ).group_by(ModelArtifact.model_name).subquery()
            
            models = session.query(ModelArtifact).join(
                subquery,
                (ModelArtifact.model_name == subquery.c.model_name)
            ).order_by(ModelArtifact.model_name).all()
            
            result = [{
                'model_name': m.model_name,
                'version': m.version,
                'path': m.path,
                'metadata': m.model_metadata,
                'metrics': m.metrics,
                'promoted_at': m.promoted_at.isoformat() if m.promoted_at else None
            } for m in models]
        
        return jsonify({'data': result, 'error': None}), 200
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500


@models_mgmt_bp.route('/models/reload', methods=['POST'])
def reload_models():
    """
    POST /models/reload
    
    Clears model cache and forces reload from database.
    """
    try:
        loader = get_models_loader()
        cached_before = loader.get_cached_models()
        loader.reload_all()
        
        result = {
            'message': 'Model cache cleared',
            'cached_models_before': cached_before,
            'cache_size_after': len(loader.get_cached_models())
        }
        
        return jsonify({'data': result, 'error': None}), 200
        
    except Exception as e:
        logger.error(f"Failed to reload models: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500


@models_mgmt_bp.route('/retrain/run', methods=['POST'])
def trigger_retrain():
    """
    POST /retrain/run
    
    Body:
        model_type (str): 'classification' or 'forecast'
        incremental (bool, optional): Use incremental training (default: True)
        sample_limit (int, optional): Limit training samples
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'data': None, 'error': 'Request body required'}), 400
        
        model_type = data.get('model_type')
        
        if model_type not in ['classification', 'forecast']:
            return jsonify({
                'data': None,
                'error': 'model_type must be "classification" or "forecast"'
            }), 400
        
        incremental = data.get('incremental', True)
        sample_limit = data.get('sample_limit')
        
        result = retrain_model(
            model_type=model_type,
            incremental=incremental,
            sample_limit=sample_limit
        )
        
        return jsonify({'data': result, 'error': None}), 200
        
    except ValueError as e:
        logger.warning(f"Retrain request failed: {e}")
        return jsonify({'data': None, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Retrain error: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500
