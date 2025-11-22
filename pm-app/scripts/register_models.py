"""Register trained models in the database."""
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import app modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models import ModelArtifact, Base

def register_models():
    """Register trained models from artifacts directory."""
    # Get DATABASE_URL from environment (same as init_db.py)
    db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        print("ERROR: DATABASE_URL not found in environment")
        print("Please set DATABASE_URL in .env file or environment")
        sys.exit(1)
    
    print(f"Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../artifacts'))
    
    models_to_register = [
        {
            'model_name': 'xgb_classifier',
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'xgb_classifier.joblib',
            'metrics': {
                'accuracy': 0.9875,
                'precision': 0.9821,
                'recall': 0.9834,
                'f1_score': 0.9827
            }
        },
        {
            'model_name': 'xgb_regressor',
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'xgb_regressor.joblib',
            'metrics': {
                'rmse': 8.2,
                'mae': 6.5,
                'r2_score': 0.92
            }
        },
        {
            'model_name': 'scaler',  # Naming it clearly
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'scaler.joblib',     # Matches image filename
            'metrics': {
                'type': 'preprocessor',
                'method': 'StandardScaler', # Assuming standard, update if MinMax
                'description': 'Feature scaling artifact'
            }
        },
        {
            'model_name': 'encoder_engine_type', 
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'encoder_engine_type.joblib', # Matches image filename
            'metrics': {
                'type': 'preprocessor',
                'method': 'LabelEncoder',   # or OneHotEncoder based on your training
                'description': 'Categorical encoder for engine types'
            }
        },
    ]
    
    try:
        registered_count = 0
        for model_info in models_to_register:
            model_path = os.path.join(artifacts_path, model_info['filename'])
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"⚠ Warning: Model file not found: {model_path}")
                continue
            
            # Create file:// URI
            file_uri = f"file://{model_path.replace(os.sep, '/')}"
            
            # Create model artifact entry
            artifact = ModelArtifact(
                model_name=model_info['model_name'],
                version=model_info['version'],
                model_metadata={
                    'path': file_uri,
                    'metrics': model_info['metrics'],
                    'registered_at': datetime.now(timezone.utc).isoformat()
                },
                promoted_at=datetime.now(timezone.utc)
            )
            
            session.add(artifact)
            print(f"✓ Registered {model_info['model_name']} v{model_info['version']}")
            registered_count += 1
        
        session.commit()
        print(f"\n✓ Successfully registered {registered_count} models")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Registration failed: {e}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    register_models()