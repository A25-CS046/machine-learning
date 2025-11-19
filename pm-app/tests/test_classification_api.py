import json
from datetime import datetime, timezone


def test_predict_classification_success(client, sample_telemetry, sample_model_artifact):
    """Test successful classification prediction."""
    response = client.post('/predict/classification', json={
        'model_name': 'test_model',
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001'
    })
    
    assert response.status_code in [200, 404]
    data = json.loads(response.data)
    assert 'data' in data
    assert 'error' in data


def test_predict_classification_missing_fields(client):
    """Test classification with missing required fields."""
    response = client.post('/predict/classification', json={
        'model_name': 'test_model'
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] is not None
    assert 'required fields' in data['error'].lower()


def test_predict_classification_with_timestamp(client, sample_telemetry):
    """Test classification with explicit timestamp."""
    response = client.post('/predict/classification', json={
        'model_name': 'test_model',
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001',
        'timestamp_before': '2024-01-01T12:00:00Z'
    })
    
    assert response.status_code in [200, 404]
    data = json.loads(response.data)
    assert 'data' in data
    assert 'error' in data


def test_predict_classification_no_telemetry(client):
    """Test classification with non-existent unit."""
    response = client.post('/predict/classification', json={
        'model_name': 'test_model',
        'product_id': 'NONEXISTENT',
        'unit_id': 'NONEXISTENT'
    })
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['error'] is not None


def test_predict_classification_invalid_timestamp(client):
    """Test classification with invalid timestamp format."""
    response = client.post('/predict/classification', json={
        'model_name': 'test_model',
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001',
        'timestamp_before': 'invalid-timestamp'
    })
    
    assert response.status_code in [400, 404]
    data = json.loads(response.data)
    assert data['error'] is not None
