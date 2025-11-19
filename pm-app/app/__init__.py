import logging
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from app.config import load_config
from app.db import init_db


def setup_logging(log_level: str, log_format: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    if log_format == 'json':
        import json
        
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                }
                if record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_data)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    logging.root.setLevel(level)
    logging.root.addHandler(handler)


def create_app() -> Flask:
    config = load_config()
    
    setup_logging(config.log_level, config.log_format)
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.secret_key
    
    CORS(app, origins=config.cors_origins)
    
    init_db()
    
    from app.routes.health import health_bp
    from app.routes.classification import classification_bp
    from app.routes.forecast import forecast_bp
    from app.routes.optimizer import optimizer_bp
    from app.routes.models_mgmt import models_mgmt_bp
    from app.routes.maintenance import maintenance_bp
    from app.routes.copilot_tools import copilot_bp
    
    app.register_blueprint(health_bp)
    app.register_blueprint(classification_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(optimizer_bp)
    app.register_blueprint(models_mgmt_bp)
    app.register_blueprint(maintenance_bp)
    app.register_blueprint(copilot_bp)
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'data': None, 'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'data': None, 'error': 'Internal server error'}), 500
    
    return app
