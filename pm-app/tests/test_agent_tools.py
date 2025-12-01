"""
Unit tests for LangChain agent tools.

Tests cover tool execution, input validation, error handling, and response formatting.
"""

import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from app.services.agent_tools import (
    PredictFailureTool,
    PredictRULTool,
    OptimizeScheduleTool,
    PredictFailureInput,
    PredictRULInput,
    OptimizeScheduleInput,
    get_all_tools
)


# ============================================================================
# Test Pydantic Input Schemas
# ============================================================================

class TestPredictFailureInput:
    """Tests for PredictFailureInput schema."""
    
    def test_valid_input(self):
        """Test valid input creates model successfully."""
        input_data = PredictFailureInput(
            product_id="L56614",
            unit_id="9435",
            horizon_hours=24
        )
        assert input_data.product_id == "L56614"
        assert input_data.unit_id == "9435"
        assert input_data.horizon_hours == 24
    
    def test_default_horizon_hours(self):
        """Test horizon_hours defaults to 24."""
        input_data = PredictFailureInput(
            product_id="L56614",
            unit_id="9435"
        )
        assert input_data.horizon_hours == 24
    
    def test_strips_whitespace(self):
        """Test that product_id and unit_id are stripped."""
        input_data = PredictFailureInput(
            product_id="  L56614  ",
            unit_id="  9435  "
        )
        assert input_data.product_id == "L56614"
        assert input_data.unit_id == "9435"
    
    def test_empty_product_id_raises_error(self):
        """Test empty product_id raises validation error."""
        with pytest.raises(ValidationError):
            PredictFailureInput(product_id="", unit_id="9435")
    
    def test_empty_unit_id_raises_error(self):
        """Test empty unit_id raises validation error."""
        with pytest.raises(ValidationError):
            PredictFailureInput(product_id="L56614", unit_id="")
    
    def test_horizon_hours_out_of_range(self):
        """Test horizon_hours outside valid range raises error."""
        with pytest.raises(ValidationError):
            PredictFailureInput(product_id="L56614", unit_id="9435", horizon_hours=200)
        
        with pytest.raises(ValidationError):
            PredictFailureInput(product_id="L56614", unit_id="9435", horizon_hours=0)


class TestPredictRULInput:
    """Tests for PredictRULInput schema."""
    
    def test_valid_input(self):
        """Test valid input creates model successfully."""
        input_data = PredictRULInput(
            product_id="M29501",
            unit_id="1234",
            horizon_steps=10
        )
        assert input_data.product_id == "M29501"
        assert input_data.unit_id == "1234"
        assert input_data.horizon_steps == 10
    
    def test_default_horizon_steps(self):
        """Test horizon_steps defaults to 10."""
        input_data = PredictRULInput(
            product_id="M29501",
            unit_id="1234"
        )
        assert input_data.horizon_steps == 10


class TestOptimizeScheduleInput:
    """Tests for OptimizeScheduleInput schema."""
    
    def test_valid_input(self):
        """Test valid input creates model successfully."""
        input_data = OptimizeScheduleInput(
            unit_list=[
                {"product_id": "L56614", "unit_id": "9435"},
                {"product_id": "M29501", "unit_id": "1234"}
            ],
            risk_threshold=0.7,
            rul_threshold=24.0,
            horizon_days=7
        )
        assert len(input_data.unit_list) == 2
        assert input_data.risk_threshold == 0.7
    
    def test_empty_unit_list_raises_error(self):
        """Test empty unit_list raises validation error."""
        with pytest.raises(ValidationError):
            OptimizeScheduleInput(unit_list=[])
    
    def test_unit_missing_keys_raises_error(self):
        """Test unit without required keys raises error."""
        with pytest.raises(ValidationError):
            OptimizeScheduleInput(
                unit_list=[{"product_id": "L56614"}]  # Missing unit_id
            )
    
    def test_risk_threshold_out_of_range(self):
        """Test risk_threshold outside [0, 1] raises error."""
        with pytest.raises(ValidationError):
            OptimizeScheduleInput(
                unit_list=[{"product_id": "L56614", "unit_id": "9435"}],
                risk_threshold=1.5
            )


# ============================================================================
# Test Tool Implementations
# ============================================================================

class TestPredictFailureTool:
    """Tests for PredictFailureTool."""
    
    @patch('app.services.agent_tools.predict_failure')
    def test_successful_execution(self, mock_predict_failure):
        """Test successful tool execution."""
        # Mock successful prediction
        mock_predict_failure.return_value = {
            'prediction': 1,
            'probabilities': {'failure': 0.82, 'no_failure': 0.18},
            'failure_type': 'TWF',
            'fallback_recommendation': 'Schedule immediate maintenance'
        }
        
        tool = PredictFailureTool()
        result = tool._run(product_id="L56614", unit_id="9435", horizon_hours=24)
        
        assert result['success'] is True
        assert result['failure_probability'] == 0.82
        assert result['will_fail'] is True
        assert result['failure_type'] == 'TWF'
        assert 'recommendation' in result
    
    @patch('app.services.agent_tools.predict_failure')
    def test_data_not_found_error(self, mock_predict_failure):
        """Test handling of data not found error."""
        # Mock ValueError (no data found)
        mock_predict_failure.side_effect = ValueError("No telemetry data found")
        
        tool = PredictFailureTool()
        result = tool._run(product_id="INVALID", unit_id="0000")
        
        assert result['success'] is False
        assert result['error_type'] == 'data_not_found'
        assert 'No telemetry data found' in result['error']
    
    @patch('app.services.agent_tools.predict_failure')
    def test_execution_error(self, mock_predict_failure):
        """Test handling of unexpected execution error."""
        # Mock unexpected exception
        mock_predict_failure.side_effect = RuntimeError("Database connection failed")
        
        tool = PredictFailureTool()
        result = tool._run(product_id="L56614", unit_id="9435")
        
        assert result['success'] is False
        assert result['error_type'] == 'execution_error'
        assert 'recommendation' in result


class TestPredictRULTool:
    """Tests for PredictRULTool."""
    
    @patch('app.services.agent_tools.predict_rul')
    def test_successful_execution(self, mock_predict_rul):
        """Test successful RUL prediction."""
        # Mock successful prediction
        mock_predict_rul.return_value = {
            'current_rul_hours': 18.5,
            'forecast_horizon_steps': 10
        }
        
        tool = PredictRULTool()
        result = tool._run(product_id="L56614", unit_id="9435", horizon_steps=10)
        
        assert result['success'] is True
        assert result['current_rul_hours'] == 18.5
        assert result['current_rul_days'] == 0.8  # 18.5 / 24
        assert result['urgency'] == 'high'  # < 24 hours
        assert result['critical'] is True
    
    @patch('app.services.agent_tools.predict_rul')
    def test_critical_urgency(self, mock_predict_rul):
        """Test critical urgency categorization."""
        mock_predict_rul.return_value = {
            'current_rul_hours': 8.0,
            'forecast_horizon_steps': 10
        }
        
        tool = PredictRULTool()
        result = tool._run(product_id="L56614", unit_id="9435")
        
        assert result['urgency'] == 'critical'
        assert 'immediate maintenance' in result['recommendation'].lower()
    
    @patch('app.services.agent_tools.predict_rul')
    def test_low_urgency(self, mock_predict_rul):
        """Test low urgency categorization."""
        mock_predict_rul.return_value = {
            'current_rul_hours': 120.0,
            'forecast_horizon_steps': 10
        }
        
        tool = PredictRULTool()
        result = tool._run(product_id="L56614", unit_id="9435")
        
        assert result['urgency'] == 'low'
        assert result['critical'] is False


class TestOptimizeScheduleTool:
    """Tests for OptimizeScheduleTool."""
    
    @patch('app.services.agent_tools.optimize_maintenance_schedule')
    def test_successful_execution(self, mock_optimize):
        """Test successful schedule optimization."""
        # Mock successful optimization
        mock_optimize.return_value = {
            'schedule_id': 'SCH_A4F2B8',
            'maintenance_scheduled': 2,
            'high_risk_units_found': 1,
            'recommendations': [
                {'product_id': 'L56614', 'priority': 'high'},
                {'product_id': 'M29501', 'priority': 'medium'}
            ]
        }
        
        tool = OptimizeScheduleTool()
        result = tool._run(
            unit_list=[
                {"product_id": "L56614", "unit_id": "9435"},
                {"product_id": "M29501", "unit_id": "1234"}
            ],
            risk_threshold=0.7,
            rul_threshold=24.0,
            horizon_days=7
        )
        
        assert result['success'] is True
        assert result['schedule_id'] == 'SCH_A4F2B8'
        assert result['units_scheduled'] == 2
        assert result['high_risk_units'] == 1
        assert len(result['recommendations']) <= 5  # Top 5 only
    
    @patch('app.services.agent_tools.optimize_maintenance_schedule')
    def test_invalid_input_error(self, mock_optimize):
        """Test handling of invalid input."""
        mock_optimize.side_effect = ValueError("Invalid unit list")
        
        tool = OptimizeScheduleTool()
        result = tool._run(
            unit_list=[{"product_id": "L56614", "unit_id": "9435"}],
            risk_threshold=0.7
        )
        
        assert result['success'] is False
        assert result['error_type'] == 'invalid_input'


# ============================================================================
# Test Tool Registry
# ============================================================================

def test_get_all_tools():
    """Test get_all_tools returns correct tools."""
    tools = get_all_tools()
    
    assert len(tools) == 3
    assert isinstance(tools[0], PredictFailureTool)
    assert isinstance(tools[1], PredictRULTool)
    assert isinstance(tools[2], OptimizeScheduleTool)
    
    # Verify all tools have required properties
    for tool in tools:
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'args_schema')
        assert hasattr(tool, '_run')


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestToolIntegration:
    """
    Integration tests requiring database and models.
    
    These tests are skipped in CI environments without database access.
    """
    
    @pytest.mark.skip(reason="Requires database with telemetry data")
    def test_predict_failure_with_real_data(self):
        """Test failure prediction with real database data."""
        tool = PredictFailureTool()
        result = tool._run(product_id="L56614", unit_id="9435")
        
        # Should return valid result structure
        assert 'success' in result
        if result['success']:
            assert 'failure_probability' in result
            assert 0 <= result['failure_probability'] <= 1
    
    @pytest.mark.skip(reason="Requires database with telemetry data")
    def test_predict_rul_with_real_data(self):
        """Test RUL prediction with real database data."""
        tool = PredictRULTool()
        result = tool._run(product_id="L56614", unit_id="9435")
        
        if result['success']:
            assert result['current_rul_hours'] >= 0
            assert result['urgency'] in ['critical', 'high', 'medium', 'low']
