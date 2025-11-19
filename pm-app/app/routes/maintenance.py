import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import and_
from app.db import get_db
from app.models import MaintenanceSchedule
from app.routes.classification import parse_timestamp

logger = logging.getLogger(__name__)

maintenance_bp = Blueprint('maintenance', __name__)


@maintenance_bp.route('/maintenance/schedule', methods=['GET'])
def get_maintenance_schedules():
    """
    GET /maintenance/schedule
    
    Query Parameters:
        product_id (str, optional): Filter by product ID
        unit_id (str, optional): Filter by unit ID
        status (str, optional): Filter by status (PENDING, COMPLETED, CANCELLED)
        start_date (str, optional): ISO timestamp for earliest recommended_start
        end_date (str, optional): ISO timestamp for latest recommended_end
    """
    try:
        product_id = request.args.get('product_id')
        unit_id = request.args.get('unit_id')
        status = request.args.get('status')
        start_date = parse_timestamp(request.args.get('start_date'))
        end_date = parse_timestamp(request.args.get('end_date'))
        
        with get_db() as session:
            query = session.query(MaintenanceSchedule)
            
            if product_id:
                query = query.filter(MaintenanceSchedule.product_id == product_id)
            
            if unit_id:
                query = query.filter(MaintenanceSchedule.unit_id == unit_id)
            
            if status:
                query = query.filter(MaintenanceSchedule.status == status)
            
            if start_date:
                query = query.filter(MaintenanceSchedule.recommended_start >= start_date)
            
            if end_date:
                query = query.filter(MaintenanceSchedule.recommended_end <= end_date)
            
            schedules = query.order_by(MaintenanceSchedule.recommended_start).all()
            
            result = [{
                'id': s.id,
                'schedule_id': s.schedule_id,
                'product_id': s.product_id,
                'unit_id': s.unit_id,
                'recommended_start': s.recommended_start.isoformat() if s.recommended_start else None,
                'recommended_end': s.recommended_end.isoformat() if s.recommended_end else None,
                'reason': s.reason,
                'risk_score': s.risk_score,
                'model_version': s.model_version,
                'actions': s.actions,
                'constraints_applied': s.constraints_applied,
                'created_at': s.created_at.isoformat() if s.created_at else None,
                'status': s.status
            } for s in schedules]
        
        return jsonify({'data': result, 'error': None}), 200
        
    except Exception as e:
        logger.error(f"Failed to retrieve schedules: {e}", exc_info=True)
        return jsonify({'data': None, 'error': 'Internal server error'}), 500
