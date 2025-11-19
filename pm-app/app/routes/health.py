import logging
from flask import Blueprint, jsonify
from app.db import get_engine
from app.services.models_loader import get_models_loader

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    GET /health
    
    Performs basic health checks for database and model availability.
    """
    try:
        engine = get_engine()
        
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            db_status = 'ok'
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            db_status = f'error: {str(e)}'
        
        try:
            loader = get_models_loader()
            cached_models = loader.get_cached_models()
            models_status = 'ok'
            models_count = len(cached_models)
        except Exception as e:
            logger.error(f"Models check failed: {e}")
            models_status = f'error: {str(e)}'
            models_count = 0
        
        overall_healthy = db_status == 'ok'
        
        result = {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'database': db_status,
            'models': models_status,
            'cached_models_count': models_count
        }
        
        status_code = 200 if overall_healthy else 503
        
        return jsonify({'data': result, 'error': None}), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            'data': {'status': 'unhealthy'},
            'error': 'Health check failed'
        }), 503
