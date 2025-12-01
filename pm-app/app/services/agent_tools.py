"""
LangChain tool wrappers for predictive maintenance AI copilot.

These tools provide structured interfaces for the LangChain agent to interact
with the underlying ML models and business logic.
"""

import logging
from typing import Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
from app.services.classification_service import predict_failure
from app.services.forecast_service import predict_rul
from app.services.optimizer_service import optimize_maintenance_schedule

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Input Schemas (Pydantic Models)
# ============================================================================

class PredictFailureInput(BaseModel):
    """Input schema for failure prediction tool."""
    
    product_id: str = Field(
        ...,
        description="Equipment product identifier (e.g., 'L56614', 'M29501', 'H30221')",
        min_length=1,
        max_length=50
    )
    unit_id: str = Field(
        ...,
        description="Specific unit identifier within the product line (e.g., '9435')",
        min_length=1,
        max_length=50
    )
    horizon_hours: int = Field(
        default=24,
        description="Prediction horizon in hours (default: 24, range: 1-168)",
        ge=1,
        le=168
    )
    
    @field_validator('product_id')
    @classmethod
    def validate_product_id(cls, v: str) -> str:
        """Validate product ID format."""
        if not v or not v.strip():
            raise ValueError('product_id cannot be empty')
        return v.strip()
    
    @field_validator('unit_id')
    @classmethod
    def validate_unit_id(cls, v: str) -> str:
        """Validate unit ID format."""
        if not v or not v.strip():
            raise ValueError('unit_id cannot be empty')
        return v.strip()


class PredictRULInput(BaseModel):
    """Input schema for RUL prediction tool."""
    
    product_id: str = Field(
        ...,
        description="Equipment product identifier",
        min_length=1,
        max_length=50
    )
    unit_id: str = Field(
        ...,
        description="Specific unit identifier",
        min_length=1,
        max_length=50
    )
    horizon_steps: int = Field(
        default=10,
        description="Number of future timesteps to forecast (default: 10, range: 1-168)",
        ge=1,
        le=168
    )
    
    @field_validator('product_id')
    @classmethod
    def validate_product_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('product_id cannot be empty')
        return v.strip()
    
    @field_validator('unit_id')
    @classmethod
    def validate_unit_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('unit_id cannot be empty')
        return v.strip()


class OptimizeScheduleInput(BaseModel):
    """Input schema for maintenance scheduling tool."""
    
    unit_list: list[dict[str, str]] = Field(
        ...,
        description="List of units to schedule. Each unit must have 'product_id' and 'unit_id' keys.",
        min_length=1,
        max_length=100
    )
    risk_threshold: float = Field(
        default=0.7,
        description="Failure probability threshold for scheduling (0-1, default: 0.7)",
        ge=0.0,
        le=1.0
    )
    rul_threshold: float = Field(
        default=24.0,
        description="RUL threshold in hours for scheduling (default: 24.0)",
        ge=0.0,
        le=1000.0
    )
    horizon_days: int = Field(
        default=7,
        description="Planning horizon in days (default: 7, range: 1-30)",
        ge=1,
        le=30
    )
    
    @field_validator('unit_list')
    @classmethod
    def validate_unit_list(cls, v: list) -> list:
        """Validate unit list structure."""
        if not v:
            raise ValueError('unit_list cannot be empty')
        
        for idx, unit in enumerate(v):
            if not isinstance(unit, dict):
                raise ValueError(f'Unit at index {idx} must be a dictionary')
            if 'product_id' not in unit or 'unit_id' not in unit:
                raise ValueError(f'Unit at index {idx} must have product_id and unit_id keys')
            if not unit['product_id'] or not unit['unit_id']:
                raise ValueError(f'Unit at index {idx} has empty product_id or unit_id')
        
        return v


# ============================================================================
# LangChain Tool Implementations
# ============================================================================

class PredictFailureTool(BaseTool):
    """
    Tool for predicting equipment failure probability and failure type.
    
    This tool uses the XGBoost classifier to analyze telemetry data and predict:
    - Failure probability (0-1 scale)
    - Predicted failure type if failure is likely (TWF, HDF, PWF, OSF, RNF)
    - Confidence metrics and recommendations
    """
    
    name: str = "predict_failure"
    description: str = (
        "Predict equipment failure probability and failure type for a specific unit. "
        "Use this when the user asks about failure risk, breakdown probability, equipment health status, "
        "or whether a machine is likely to fail. Returns failure probability (0-1), predicted failure type "
        "if failure is likely, and actionable recommendations."
    )
    args_schema: type[BaseModel] = PredictFailureInput
    return_direct: bool = False
    
    def _run(self, product_id: str, unit_id: str, horizon_hours: int = 24) -> dict[str, Any]:
        """
        Execute failure prediction.
        
        Args:
            product_id: Equipment product identifier
            unit_id: Unit identifier
            horizon_hours: Prediction horizon in hours
        
        Returns:
            Dict with failure probability, failure type, and recommendations
        """
        try:
            logger.info(f"PredictFailureTool: Executing for {product_id}/{unit_id}")
            
            result = predict_failure(
                model_name='xgb_classifier',
                product_id=product_id,
                unit_id=unit_id
            )
            
            # Simplify response for LLM consumption
            simplified = {
                'success': True,
                'product_id': product_id,
                'unit_id': unit_id,
                'failure_probability': result['probabilities']['failure'],
                'will_fail': result['prediction'] == 1,
                'failure_type': result.get('failure_type', 'Unknown'),
                'confidence': 'high' if result['probabilities']['failure'] > 0.8 or result['probabilities']['failure'] < 0.2 else 'medium',
                'recommendation': result.get('fallback_recommendation', 'Monitor equipment')
            }
            
            logger.info(f"PredictFailureTool: Success - Failure probability: {simplified['failure_probability']:.2%}")
            return simplified
            
        except ValueError as e:
            logger.warning(f"PredictFailureTool: Data not found - {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'data_not_found',
                'recommendation': f'No telemetry data found for {product_id}/{unit_id}. Verify equipment ID or check data pipeline.'
            }
        
        except Exception as e:
            logger.error(f"PredictFailureTool: Execution failed - {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': 'execution_error',
                'recommendation': 'Manual inspection required. Contact maintenance team.'
            }
    
    async def _arun(self, *args, **kwargs):
        """Async execution (not implemented)."""
        raise NotImplementedError("Async execution not supported yet")


class PredictRULTool(BaseTool):
    """
    Tool for predicting Remaining Useful Life (RUL) of equipment.
    
    This tool uses the XGBoost regressor to forecast how many hours of operation
    remain before maintenance is required.
    """
    
    name: str = "predict_rul"
    description: str = (
        "Predict Remaining Useful Life (RUL) in hours for equipment. "
        "Use this when the user asks about remaining time, how long until failure, "
        "when maintenance is needed, or equipment lifespan. Returns current RUL estimate "
        "and multi-step forecast for upcoming timesteps."
    )
    args_schema: type[BaseModel] = PredictRULInput
    return_direct: bool = False
    
    def _run(self, product_id: str, unit_id: str, horizon_steps: int = 10) -> dict[str, Any]:
        """
        Execute RUL prediction.
        
        Args:
            product_id: Equipment product identifier
            unit_id: Unit identifier
            horizon_steps: Number of future timesteps to forecast
        
        Returns:
            Dict with current RUL, forecast, and recommendations
        """
        try:
            logger.info(f"PredictRULTool: Executing for {product_id}/{unit_id}")
            
            result = predict_rul(
                model_name='xgb_regressor',
                product_id=product_id,
                unit_id=unit_id,
                horizon_steps=horizon_steps
            )
            
            current_rul = result['current_rul_hours']
            
            # Determine criticality
            if current_rul < 12:
                urgency = 'critical'
                recommendation = 'Schedule immediate maintenance within next 6 hours'
            elif current_rul < 24:
                urgency = 'high'
                recommendation = 'Schedule maintenance within next 12 hours'
            elif current_rul < 72:
                urgency = 'medium'
                recommendation = 'Schedule maintenance within next 2-3 days'
            else:
                urgency = 'low'
                recommendation = 'Continue monitoring. Maintenance can be planned normally'
            
            simplified = {
                'success': True,
                'product_id': product_id,
                'unit_id': unit_id,
                'current_rul_hours': current_rul,
                'current_rul_days': round(current_rul / 24, 1),
                'forecast_horizon_steps': result['forecast_horizon_steps'],
                'urgency': urgency,
                'critical': current_rul < 24,
                'recommendation': recommendation
            }
            
            logger.info(f"PredictRULTool: Success - RUL: {current_rul:.1f} hours ({urgency})")
            return simplified
            
        except ValueError as e:
            logger.warning(f"PredictRULTool: Data not found - {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'data_not_found',
                'recommendation': f'No RUL data found for {product_id}/{unit_id}. Verify equipment ID or check data pipeline.'
            }
        
        except Exception as e:
            logger.error(f"PredictRULTool: Execution failed - {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': 'execution_error',
                'recommendation': 'Manual inspection required. Contact maintenance team.'
            }
    
    async def _arun(self, *args, **kwargs):
        """Async execution (not implemented)."""
        raise NotImplementedError("Async execution not supported yet")


class OptimizeScheduleTool(BaseTool):
    """
    Tool for generating optimized maintenance schedules.
    
    This tool uses a greedy optimization algorithm to create maintenance schedules
    for multiple units based on failure risk and RUL thresholds.
    """
    
    name: str = "optimize_schedule"
    description: str = (
        "Generate an optimized maintenance schedule for multiple units based on failure risk and RUL thresholds. "
        "Use this when the user asks to schedule maintenance for multiple machines, create a maintenance plan, "
        "or optimize maintenance windows. Returns prioritized schedule with time windows and resource allocation."
    )
    args_schema: type[BaseModel] = OptimizeScheduleInput
    return_direct: bool = False
    
    def _run(
        self,
        unit_list: list[dict[str, str]],
        risk_threshold: float = 0.7,
        rul_threshold: float = 24.0,
        horizon_days: int = 7
    ) -> dict[str, Any]:
        """
        Execute maintenance schedule optimization.
        
        Args:
            unit_list: List of units to schedule (each with product_id and unit_id)
            risk_threshold: Failure probability threshold for scheduling
            rul_threshold: RUL threshold in hours
            horizon_days: Planning horizon in days
        
        Returns:
            Dict with schedule ID, scheduled units, and recommendations
        """
        try:
            logger.info(f"OptimizeScheduleTool: Executing for {len(unit_list)} units")
            
            result = optimize_maintenance_schedule(
                unit_list=unit_list,
                risk_threshold=risk_threshold,
                rul_threshold=rul_threshold,
                horizon_days=horizon_days
            )
            
            simplified = {
                'success': True,
                'schedule_id': result['schedule_id'],
                'total_units_analyzed': len(unit_list),
                'units_scheduled': result['maintenance_scheduled'],
                'high_risk_units': result['high_risk_units_found'],
                'planning_horizon_days': horizon_days,
                'recommendations': result['recommendations'][:5],  # Top 5 only
                'summary': f"Scheduled {result['maintenance_scheduled']} out of {len(unit_list)} units. {result['high_risk_units_found']} high-risk units identified."
            }
            
            logger.info(f"OptimizeScheduleTool: Success - Scheduled {simplified['units_scheduled']} units")
            return simplified
            
        except ValueError as e:
            logger.warning(f"OptimizeScheduleTool: Invalid input - {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'invalid_input',
                'recommendation': 'Verify unit IDs and try again.'
            }
        
        except Exception as e:
            logger.error(f"OptimizeScheduleTool: Execution failed - {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': 'execution_error',
                'recommendation': 'Manual scheduling required. Contact operations team.'
            }
    
    async def _arun(self, *args, **kwargs):
        """Async execution (not implemented)."""
        raise NotImplementedError("Async execution not supported yet")


# ============================================================================
# Tool Registry
# ============================================================================

def get_all_tools() -> list[BaseTool]:
    """
    Get all available tools for the maintenance copilot agent.
    
    Returns:
        List of instantiated LangChain tools
    """
    return [
        PredictFailureTool(),
        PredictRULTool(),
        OptimizeScheduleTool()
    ]
