import os
import pytest
import tempfile
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import create_app
from app.models import Base, Telemetry, ModelArtifact
from app.db import reset_db


@pytest.fixture(scope='session')
def test_db():
    """Create temporary test database."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    db_url = f'sqlite:///{db_path}'
    
    os.environ['DATABASE_URL'] = db_url
    
    engine = create_engine(
        db_url,
        connect_args={'check_same_thread': False}
    )
    
    # Drop all existing tables and recreate fresh
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    reset_db(engine)
    
    yield engine
    
    # Properly dispose of all connections before deleting file
    engine.dispose()
    os.close(fd)
    
    # Small delay to ensure file handles are released
    import time
    time.sleep(0.1)
    
    try:
        os.unlink(db_path)
    except PermissionError:
        pass  # File will be cleaned up by OS eventually


@pytest.fixture(scope='function')
def app(test_db):
    """Create Flask test app."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture(scope='function')
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture(scope='function')
def db_session(test_db):
    """Create database session for tests."""
    Session = sessionmaker(bind=test_db)
    session = Session()
    
    yield session
    
    session.rollback()
    session.close()
    
    # Clean up all tables after each test
    for table in reversed(Base.metadata.sorted_tables):
        session = Session()
        session.execute(table.delete())
        session.commit()
        session.close()


@pytest.fixture(scope='function')
def sample_telemetry(db_session):
    """Create sample telemetry data."""
    records = []
    
    for i in range(10):
        record = Telemetry(
            product_id='PROD_001',
            unit_id='UNIT_001',
            timestamp=datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc).isoformat(),
            step_index=i,
            engine_type='L',
            air_temperature_K=298.0 + i,
            process_temperature_K=308.0 + i,
            rotational_speed_rpm=1500.0 + i * 10,
            torque_Nm=40.0 + i,
            tool_wear_min=100.0 + i * 5,
            is_failure=0 if i < 8 else 1,
            failure_type='No Failure' if i < 8 else 'Heat Dissipation Failure',
            synthetic_RUL=100.0 - i * 10
        )
        records.append(record)
        db_session.add(record)
    
    db_session.commit()
    return records


@pytest.fixture(scope='function')
def sample_model_artifact(db_session):
    """Create sample model artifact."""
    artifact = ModelArtifact(
        model_name='test_model',
        version='20240101_120000',
        model_metadata={
            'path': 'file:///tmp/test_model_20240101_120000.joblib',
            'metrics': {'accuracy': 0.95},
            'test': True
        },
        promoted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    db_session.add(artifact)
    db_session.commit()
    
    # Force load all attributes before returning to avoid lazy loading issues
    _ = artifact.id
    _ = artifact.model_name
    _ = artifact.version
    _ = artifact.model_metadata
    _ = artifact.promoted_at
    
    db_session.refresh(artifact)  # Refresh to get the auto-generated id
    return artifact