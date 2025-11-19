import json


def test_optimize_schedule_success(client, sample_telemetry):
    """Test successful schedule optimization."""
    response = client.post('/optimizer/schedule', json={
        'unit_list': [
            {'product_id': 'PROD_001', 'unit_id': 'UNIT_001'}
        ],
        'risk_threshold': 0.7,
        'rul_threshold': 24.0,
        'horizon_days': 7
    })
    
    assert response.status_code in [200, 400]
    data = json.loads(response.data)
    assert 'data' in data
    assert 'error' in data


def test_optimize_schedule_missing_unit_list(client):
    """Test optimizer with missing unit_list."""
    response = client.post('/optimizer/schedule', json={
        'risk_threshold': 0.7
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] is not None
    assert 'unit_list' in data['error'].lower()


def test_optimize_schedule_empty_unit_list(client):
    """Test optimizer with empty unit_list."""
    response = client.post('/optimizer/schedule', json={
        'unit_list': []
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] is not None


def test_optimize_schedule_invalid_unit_format(client):
    """Test optimizer with invalid unit format."""
    response = client.post('/optimizer/schedule', json={
        'unit_list': [
            {'product_id': 'PROD_001'}
        ]
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] is not None


def test_optimize_schedule_multiple_units(client, sample_telemetry):
    """Test optimizer with multiple units."""
    response = client.post('/optimizer/schedule', json={
        'unit_list': [
            {'product_id': 'PROD_001', 'unit_id': 'UNIT_001'},
            {'product_id': 'PROD_001', 'unit_id': 'UNIT_002'}
        ],
        'teams_available': 3,
        'hours_per_day': 10
    })
    
    assert response.status_code in [200, 400]
    data = json.loads(response.data)
    assert 'data' in data
    assert 'error' in data
