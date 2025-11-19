import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from app.db import get_db
from app.models import MaintenanceSchedule
from app.services.classification_service import predict_failure
from app.services.forecast_service import predict_rul

logger = logging.getLogger(__name__)


def optimize_maintenance_schedule(
    unit_list: list[dict[str, str]],
    risk_threshold: float = 0.7,
    rul_threshold: float = 24.0,
    horizon_days: int = 7,
    teams_available: int = 2,
    hours_per_day: int = 8,
    earliest_allowed: datetime | None = None,
    latest_allowed: datetime | None = None
) -> dict:
    """
    Generate optimized maintenance schedule based on risk and RUL thresholds.
    Uses greedy scheduling algorithm to allocate maintenance windows.
    """
    if earliest_allowed is None:
        earliest_allowed = datetime.now(timezone.utc)
    
    if latest_allowed is None:
        latest_allowed = earliest_allowed + timedelta(days=horizon_days)
    
    schedule_id = f"SCH_{uuid.uuid4().hex[:12].upper()}"
    
    risk_events = []
    
    for unit in unit_list:
        product_id = unit['product_id']
        unit_id = unit['unit_id']
        
        try:
            classification_result = predict_failure(
                model_name='xgb_classifier',
                product_id=product_id,
                unit_id=unit_id
            )
            
            failure_prob = classification_result['probabilities']['failure']
            
            forecast_result = predict_rul(
                model_name='xgb_regressor',
                product_id=product_id,
                unit_id=unit_id,
                horizon_steps=horizon_days * 24
            )
            
            current_rul = forecast_result['current_rul_hours']
            
            is_high_risk = failure_prob >= risk_threshold
            is_low_rul = current_rul <= rul_threshold
            
            if is_high_risk or is_low_rul:
                risk_score = failure_prob * (1.0 / max(current_rul, 1.0))
                
                risk_events.append({
                    'product_id': product_id,
                    'unit_id': unit_id,
                    'risk_score': risk_score,
                    'failure_prob': failure_prob,
                    'rul_hours': current_rul,
                    'classification': classification_result,
                    'forecast': forecast_result
                })
        except Exception as e:
            logger.warning(f"Failed to evaluate {product_id}/{unit_id}: {e}")
            continue
    
    risk_events.sort(key=lambda x: x['risk_score'], reverse=True)
    
    scheduled_maintenance = []
    time_slots_used = {}
    
    for event in risk_events:
        rul_hours = event['rul_hours']
        
        if rul_hours < 24:
            window_start = earliest_allowed
        else:
            window_start = earliest_allowed + timedelta(hours=rul_hours / 2)
        
        if window_start > latest_allowed:
            window_start = latest_allowed - timedelta(hours=hours_per_day)
        
        window_end = window_start + timedelta(hours=hours_per_day)
        
        day_key = window_start.date()
        if day_key not in time_slots_used:
            time_slots_used[day_key] = 0
        
        if time_slots_used[day_key] < teams_available:
            time_slots_used[day_key] += 1
            
            actions = [
                "Inspect critical components",
                "Perform preventive maintenance",
                "Replace worn parts if needed"
            ]
            
            if event['failure_prob'] > 0.9:
                actions.insert(0, "URGENT: Schedule immediate inspection")
            
            scheduled_maintenance.append({
                'product_id': event['product_id'],
                'unit_id': event['unit_id'],
                'recommended_start': window_start.isoformat(),
                'recommended_end': window_end.isoformat(),
                'risk_score': event['risk_score'],
                'reason': f"Failure probability: {event['failure_prob']:.2%}, RUL: {event['rul_hours']:.1f}h",
                'actions': actions
            })
        else:
            logger.info(f"No team available on {day_key} for {event['product_id']}/{event['unit_id']}")
    
    with get_db() as session:
        for rec in scheduled_maintenance:
            schedule_entry = MaintenanceSchedule(
                schedule_id=schedule_id,
                product_id=rec['product_id'],
                unit_id=rec['unit_id'],
                recommended_start=datetime.fromisoformat(rec['recommended_start']),
                recommended_end=datetime.fromisoformat(rec['recommended_end']),
                reason=rec['reason'],
                risk_score=rec['risk_score'],
                model_version='xgb_v1',
                actions=rec['actions'],
                constraints_applied={
                    'risk_threshold': risk_threshold,
                    'rul_threshold': rul_threshold,
                    'teams_available': teams_available,
                    'hours_per_day': hours_per_day
                },
                status='PENDING'
            )
            session.add(schedule_entry)
    
    return {
        'schedule_id': schedule_id,
        'total_units_evaluated': len(unit_list),
        'high_risk_units_found': len(risk_events),
        'maintenance_scheduled': len(scheduled_maintenance),
        'recommendations': scheduled_maintenance,
        'constraints_applied': {
            'risk_threshold': risk_threshold,
            'rul_threshold': rul_threshold,
            'horizon_days': horizon_days,
            'teams_available': teams_available,
            'hours_per_day': hours_per_day,
            'earliest_allowed': earliest_allowed.isoformat(),
            'latest_allowed': latest_allowed.isoformat()
        }
    }
