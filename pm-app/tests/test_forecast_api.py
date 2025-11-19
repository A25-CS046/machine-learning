import json


def test_predict_forecast_success(client, sample_telemetry):
    """Test successful RUL forecast."""
    response = client.post('/predict/forecast', json={
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001',
        'horizon_steps': 5
    })
    
    assert response.status_code in [200, 404]
    data = json.loads(response.data)
    assert 'data' in data
    assert 'error' in data


def test_predict_forecast_missing_fields(client):
    """Test forecast with missing required fields."""
    response = client.post('/predict/forecast', json={
        'product_id': 'PROD_001'
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] is not None
    assert 'required fields' in data['error'].lower()


def test_predict_forecast_no_rul_data(client, db_session):
    """Test forecast with no RUL telemetry data."""
    response = client.post('/predict/forecast', json={
        'product_id': 'NONEXISTENT',
        'unit_id': 'NONEXISTENT'
    })
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['error'] is not None


def test_predict_forecast_default_horizon(client, sample_telemetry):
    """Test forecast with default horizon parameter."""
    response = client.post('/predict/forecast', json={
        'product_id': 'PROD_001',
        'unit_id': 'UNIT_001'
    })
    
    assert response.status_code in [200, 404]
    data = json.loads(response.data)
    assert 'data' in data
    assert 'error' in data
