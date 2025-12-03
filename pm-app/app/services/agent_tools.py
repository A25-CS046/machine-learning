"""
LangChain tool wrappers for predictive maintenance AI copilot.

These tools provide structured interfaces for the LangChain agent to interact
with the underlying ML models and business logic.
"""

import logging
import time
from typing import Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
from app.services.classification_service import predict_failure
from app.services.forecast_service import predict_rul
from app.services.optimizer_service import optimize_maintenance_schedule

logger = logging.getLogger(__name__)

# Configuration
MAX_UNITS_FOR_GLOBAL_RISK = 30
GLOBAL_RISK_CACHE_TTL = 300  # 5 minutes
DEFAULT_RISK_THRESHOLD = 0.5
DEFAULT_RUL_THRESHOLD_HOURS = 168.0  # 7 days

_global_risk_cache: dict[str, tuple[float, dict]] = {}

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


class ListUnitsInput(BaseModel):
    """Input schema for listing units tool."""
    
    engine_type: str | None = Field(
        default=None,
        description="Filter by engine type: 'L' (Low), 'M' (Medium), 'H' (High). Leave empty for all types."
    )
    limit: int = Field(
        default=50,
        description="Maximum number of units to return (default: 50, range: 1-500)",
        ge=1,
        le=500
    )
    
    @field_validator('engine_type')
    @classmethod
    def validate_engine_type(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().upper()
            if v and v not in ('L', 'M', 'H'):
                raise ValueError("engine_type must be 'L', 'M', or 'H'")
            return v if v else None
        return None


class AssessGlobalRiskInput(BaseModel):
    """Input schema for global risk assessment tool."""
    
    risk_threshold: float = Field(
        default=DEFAULT_RISK_THRESHOLD,
        description="Failure probability threshold for 'high-risk' classification (0-1, default: 0.5)",
        ge=0.0,
        le=1.0
    )
    rul_threshold_hours: float = Field(
        default=DEFAULT_RUL_THRESHOLD_HOURS,
        description="RUL hours threshold for 'at-risk' classification (default: 168 hours = 7 days for weekly horizon)",
        ge=1.0,
        le=720.0  # Max 30 days
    )
    engine_type: str | None = Field(
        default=None,
        description="Filter by engine type: 'L', 'M', 'H'. Leave empty for all types."
    )
    limit: int = Field(
        default=MAX_UNITS_FOR_GLOBAL_RISK,
        description=f"Maximum units to analyze (default: {MAX_UNITS_FOR_GLOBAL_RISK}, range: 1-100)",
        ge=1,
        le=100
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached results if available within TTL (default: True)"
    )
    
    @field_validator('engine_type')
    @classmethod
    def validate_engine_type(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().upper()
            if v and v not in ('L', 'M', 'H'):
                raise ValueError("engine_type must be 'L', 'M', or 'H'")
            return v if v else None
        return None


class PredictFailureTool(BaseTool):
    """Tool for predicting equipment failure probability."""
    
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
    """Tool for generating optimized maintenance schedules."""
    
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


class ListUnitsTool(BaseTool):
    """Tool for listing equipment units from telemetry data."""
    
    name: str = "list_units"
    description: str = (
        "List all unique equipment units available in the system. "
        "Use this when the user asks 'what units exist', 'show all machines', 'list equipment', "
        "or needs to identify available units before running analysis. "
        "Can filter by engine type (L=Low, M=Medium, H=High power)."
    )
    args_schema: type[BaseModel] = ListUnitsInput
    return_direct: bool = False
    
    def _run(self, engine_type: str | None = None, limit: int = 50) -> dict[str, Any]:
        """
        List unique equipment units.
        
        Args:
            engine_type: Optional filter by engine type (L, M, H)
            limit: Maximum number of units to return
        
        Returns:
            Dict with list of units and metadata
        """
        try:
            logger.info(f"ListUnitsTool: Fetching units (engine_type={engine_type}, limit={limit})")
            
            from sqlalchemy import distinct
            from app.db import get_db
            from app.models import Telemetry
            
            with get_db() as session:
                # Query distinct product_id, unit_id pairs
                query = session.query(
                    Telemetry.product_id,
                    Telemetry.unit_id,
                    Telemetry.engine_type
                ).distinct(Telemetry.product_id, Telemetry.unit_id)
                
                if engine_type:
                    query = query.filter(Telemetry.engine_type == engine_type)
                
                query = query.limit(limit)
                results = query.all()
                
                units = [
                    {
                        'product_id': r.product_id,
                        'unit_id': r.unit_id,
                        'engine_type': r.engine_type
                    }
                    for r in results
                ]
            
            # Group by engine type for summary
            engine_counts = {}
            for u in units:
                et = u['engine_type'] or 'Unknown'
                engine_counts[et] = engine_counts.get(et, 0) + 1
            
            result = {
                'success': True,
                'total_units': len(units),
                'engine_type_filter': engine_type,
                'engine_type_distribution': engine_counts,
                'units': units,
                'note': f"Showing {len(units)} units" + (f" (limited to {limit})" if len(units) == limit else "")
            }
            
            logger.info(f"ListUnitsTool: Found {len(units)} units")
            return result
            
        except Exception as e:
            logger.error(f"ListUnitsTool: Execution failed - {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': 'execution_error',
                'recommendation': 'Check database connection and try again.'
            }
    
    async def _arun(self, *args, **kwargs):
        """Async execution (not implemented)."""
        raise NotImplementedError("Async execution not supported yet")


class AssessGlobalRiskTool(BaseTool):
    """Tool for fleet-wide risk assessment."""
    
    name: str = "assess_global_risk"
    description: str = (
        "Perform fleet-wide risk assessment to identify high-risk equipment. "
        "Use this when the user asks 'which machines are at risk', 'show high-risk units', "
        "'what's the fleet status', 'mesin mana yang berisiko', 'overall risk assessment', "
        "or any question about fleet-wide health without specifying a specific unit. "
        "Returns ranked list of units by combined risk score, with recommendations."
    )
    args_schema: type[BaseModel] = AssessGlobalRiskInput
    return_direct: bool = False
    
    def _get_cache_key(
        self,
        risk_threshold: float,
        rul_threshold_hours: float,
        engine_type: str | None,
        limit: int
    ) -> str:
        """Generate cache key from parameters."""
        return f"global_risk:{risk_threshold}:{rul_threshold_hours}:{engine_type}:{limit}"
    
    def _check_cache(self, cache_key: str) -> dict[str, Any] | None:
        """Check if cached result exists and is still valid."""
        if cache_key in _global_risk_cache:
            timestamp, result = _global_risk_cache[cache_key]
            if time.time() - timestamp < GLOBAL_RISK_CACHE_TTL:
                logger.info(f"AssessGlobalRiskTool: Using cached result (age: {time.time() - timestamp:.1f}s)")
                result['from_cache'] = True
                return result
            else:
                # Expired, remove from cache
                del _global_risk_cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, result: dict[str, Any]) -> None:
        """Store result in cache."""
        _global_risk_cache[cache_key] = (time.time(), result.copy())
    
    def _compute_combined_risk(
        self,
        failure_prob: float,
        rul_hours: float,
        rul_threshold: float
    ) -> float:
        """Compute combined risk: max(failure_prob, rul_risk)."""
        risk_from_rul = max(0.0, min(1.0, 1.0 - (rul_hours / rul_threshold)))
        combined_risk = max(failure_prob, risk_from_rul)
        return round(combined_risk, 4)
    
    def _run(
        self,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        rul_threshold_hours: float = DEFAULT_RUL_THRESHOLD_HOURS,
        engine_type: str | None = None,
        limit: int = MAX_UNITS_FOR_GLOBAL_RISK,
        use_cache: bool = True
    ) -> dict[str, Any]:
        """
        Execute global risk assessment.
        
        Args:
            risk_threshold: Probability threshold for 'high-risk' classification
            rul_threshold_hours: RUL hours threshold for 'critical' classification
            engine_type: Optional filter by engine type
            limit: Maximum units to analyze
            use_cache: Whether to use cached results
        
        Returns:
            Dict with ranked units, fleet health summary, and recommendations
        """
        try:
            logger.info(f"AssessGlobalRiskTool: Starting assessment (threshold={risk_threshold}, rul={rul_threshold_hours}h, limit={limit})")
            
            # Check cache first
            cache_key = self._get_cache_key(risk_threshold, rul_threshold_hours, engine_type, limit)
            if use_cache:
                cached = self._check_cache(cache_key)
                if cached:
                    return cached
            
            # Step 1: Get all units
            list_units_tool = ListUnitsTool()
            units_result = list_units_tool._run(engine_type=engine_type, limit=limit)
            
            if not units_result.get('success'):
                return units_result
            
            units = units_result.get('units', [])
            total_fleet_size = len(units)
            
            if not units:
                return {
                    'success': True,
                    'total_units_analyzed': 0,
                    'message': 'No units found in the system.',
                    'high_risk_units': [],
                    'critical_rul_units': [],
                    'fleet_health_score': 1.0
                }
            
            # Step 2: Assess each unit
            assessed_units = []
            failed_assessments = []
            
            for idx, unit in enumerate(units):
                product_id = unit['product_id']
                unit_id = unit['unit_id']
                
                try:
                    # Get failure probability
                    failure_result = predict_failure(
                        model_name='xgb_classifier',
                        product_id=product_id,
                        unit_id=unit_id
                    )
                    failure_prob = failure_result['probabilities']['failure']
                    will_fail = failure_result['prediction'] == 1
                    failure_type = failure_result.get('failure_type')
                    
                    # Get RUL
                    rul_result = predict_rul(
                        model_name='xgb_regressor',
                        product_id=product_id,
                        unit_id=unit_id,
                        horizon_steps=7
                    )
                    rul_hours = rul_result['current_rul_hours']
                    
                    # Compute combined risk
                    combined_risk = self._compute_combined_risk(
                        failure_prob, rul_hours, rul_threshold_hours
                    )
                    
                    assessed_units.append({
                        'product_id': product_id,
                        'unit_id': unit_id,
                        'engine_type': unit.get('engine_type'),
                        'failure_probability': round(failure_prob, 4),
                        'will_fail_predicted': will_fail,
                        'failure_type': failure_type,
                        'rul_hours': round(rul_hours, 1),
                        'rul_days': round(rul_hours / 24, 1),
                        'combined_risk': combined_risk,
                        'is_high_risk': failure_prob >= risk_threshold,
                        'is_critical_rul': rul_hours <= rul_threshold_hours
                    })
                    
                except Exception as e:
                    failed_assessments.append({
                        'product_id': product_id,
                        'unit_id': unit_id,
                        'error': str(e)
                    })
                    logger.warning(f"Failed to assess {product_id}/{unit_id}: {e}")
            
            # Step 3: Rank by combined risk (descending)
            assessed_units.sort(key=lambda x: x['combined_risk'], reverse=True)
            
            # Step 4: Categorize units
            high_risk_units = [u for u in assessed_units if u['is_high_risk']]
            critical_rul_units = [u for u in assessed_units if u['is_critical_rul']]
            healthy_units = [u for u in assessed_units if not u['is_high_risk'] and not u['is_critical_rul']]
            
            # Step 5: Compute fleet health score (0-1, higher is healthier)
            if assessed_units:
                avg_risk = sum(u['combined_risk'] for u in assessed_units) / len(assessed_units)
                fleet_health_score = round(1.0 - avg_risk, 3)
            else:
                fleet_health_score = 1.0
            
            # Step 6: Generate recommendations
            recommendations = []
            
            # Top 5 highest risk units
            top_risk = assessed_units[:5]
            for i, unit in enumerate(top_risk, 1):
                if unit['combined_risk'] >= 0.7:
                    urgency = "IMMEDIATE"
                elif unit['combined_risk'] >= 0.5:
                    urgency = "HIGH"
                else:
                    urgency = "MEDIUM"
                
                recommendations.append({
                    'rank': i,
                    'unit': f"{unit['product_id']}/{unit['unit_id']}",
                    'combined_risk': unit['combined_risk'],
                    'failure_prob': unit['failure_probability'],
                    'rul_hours': unit['rul_hours'],
                    'urgency': urgency,
                    'action': f"Schedule maintenance within {max(1, int(unit['rul_hours'] / 2))} hours" if unit['rul_hours'] < 72 else "Monitor closely"
                })
            
            result = {
                'success': True,
                'from_cache': False,
                'total_units_analyzed': len(assessed_units),
                'total_fleet_size': total_fleet_size,
                'failed_assessments': len(failed_assessments),
                'thresholds_used': {
                    'risk_threshold': risk_threshold,
                    'rul_threshold_hours': rul_threshold_hours
                },
                'fleet_health_score': fleet_health_score,
                'fleet_status': 'HEALTHY' if fleet_health_score >= 0.7 else ('AT_RISK' if fleet_health_score >= 0.4 else 'CRITICAL'),
                'summary': {
                    'high_risk_count': len(high_risk_units),
                    'critical_rul_count': len(critical_rul_units),
                    'healthy_count': len(healthy_units)
                },
                'top_risk_units': recommendations,
                'high_risk_units': high_risk_units[:10],  # Top 10 only
                'critical_rul_units': sorted(critical_rul_units, key=lambda x: x['rul_hours'])[:10],
                'message': self._generate_summary_message(
                    len(assessed_units), len(high_risk_units), len(critical_rul_units),
                    fleet_health_score, risk_threshold, rul_threshold_hours
                )
            }
            
            # Cache result
            self._set_cache(cache_key, result)
            
            logger.info(f"AssessGlobalRiskTool: Completed - {len(high_risk_units)} high-risk, {len(critical_rul_units)} critical RUL")
            return result
            
        except Exception as e:
            logger.error(f"AssessGlobalRiskTool: Execution failed - {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': 'execution_error',
                'recommendation': 'Global risk assessment failed. Try again or check individual units.'
            }
    
    def _generate_summary_message(
        self,
        total: int,
        high_risk: int,
        critical_rul: int,
        health_score: float,
        risk_threshold: float,
        rul_threshold: float
    ) -> str:
        """Generate human-readable summary message."""
        if high_risk == 0 and critical_rul == 0:
            return f"Fleet is healthy. All {total} units operating within normal parameters."
        
        parts = []
        if high_risk > 0:
            parts.append(f"{high_risk} unit(s) exceed {risk_threshold:.0%} failure risk threshold")
        if critical_rul > 0:
            parts.append(f"{critical_rul} unit(s) have RUL below {rul_threshold:.0f} hours")
        
        status = "CRITICAL" if health_score < 0.4 else ("AT RISK" if health_score < 0.7 else "MODERATE")
        
        return f"Fleet status: {status}. Analyzed {total} units. {'. '.join(parts)}. Fleet health score: {health_score:.1%}."
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async execution not supported")


def get_all_tools() -> list[BaseTool]:
    """Get all available tools for the maintenance copilot agent."""
    return [
        PredictFailureTool(),
        PredictRULTool(),
        OptimizeScheduleTool(),
        ListUnitsTool(),
        AssessGlobalRiskTool()
    ]
