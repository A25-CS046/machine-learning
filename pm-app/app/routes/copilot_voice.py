"""
Voice input endpoint for copilot.
Accepts audio, transcribes via Gemini, and invokes the copilot agent.
"""

import logging
from flask import Blueprint, request, jsonify

from app.config import load_config
from app.services.transcription_service import (
    transcribe_audio_bytes,
    validate_mime_type,
    ALLOWED_MIME_TYPES,
)
from app.services.langchain_agent_service import get_agent_service

logger = logging.getLogger(__name__)

voice_bp = Blueprint('voice', __name__)


@voice_bp.route('/copilot/voice', methods=['POST'])
def copilot_voice():
    """
    Accept audio input, transcribe it, and invoke the copilot agent.
    
    Request:
        Content-Type: multipart/form-data
        Fields:
            - audio (required): Audio file
            - session_id (optional): Session ID for conversation context
            - language (optional): Language hint (ignored for now)
    
    Response:
        {
            "data": {
                "transcript": "...",
                "copilot_reply": {...},
                "mime_type": "audio/webm",
                "length_bytes": 12345
            },
            "error": null
        }
    """
    config = load_config()
    
    # Validate audio file exists
    if 'audio' not in request.files:
        return jsonify({'data': None, 'error': 'Missing required field: audio'}), 400
    
    audio_file = request.files['audio']
    
    if not audio_file.filename:
        return jsonify({'data': None, 'error': 'No audio file selected'}), 400
    
    # Read audio bytes
    audio_bytes = audio_file.read()
    
    # Validate not empty
    if len(audio_bytes) == 0:
        return jsonify({'data': None, 'error': 'Uploaded audio file is empty'}), 400
    
    # Validate size limit
    if len(audio_bytes) > config.gemini.max_audio_bytes:
        max_mb = config.gemini.max_audio_bytes // (1024 * 1024)
        return jsonify({
            'data': None,
            'error': f'Audio file too large. Maximum size: {max_mb}MB'
        }), 413
    
    # Determine and validate mime type
    mime_type = audio_file.mimetype or 'audio/webm'
    if not validate_mime_type(mime_type):
        allowed = ', '.join(sorted(ALLOWED_MIME_TYPES))
        return jsonify({
            'data': None,
            'error': f'Unsupported audio mime type: {mime_type}. Allowed: {allowed}'
        }), 400
    
    # Get optional session_id
    session_id = request.form.get('session_id') or request.args.get('session_id') or 'default'
    
    try:
        # Step 1: Transcribe audio
        logger.info(f"Processing voice input: {len(audio_bytes)} bytes, {mime_type}, session={session_id}")
        transcript = transcribe_audio_bytes(audio_bytes, mime_type=mime_type)
        
        if not transcript:
            return jsonify({
                'data': None,
                'error': 'Transcription returned empty result'
            }), 400
        
        # Step 2: Invoke copilot agent with transcript
        agent_service = get_agent_service()
        copilot_result = agent_service.invoke_agent(
            session_id=session_id,
            user_message=transcript
        )
        
        # Serialize intermediate_steps to avoid ToolAgentAction serialization error
        if 'intermediate_steps' in copilot_result:
            serialized_steps = []
            for step in copilot_result['intermediate_steps']:
                if isinstance(step, tuple) and len(step) == 2:
                    action, output = step
                    output_str = str(output)[:500] if not isinstance(output, str) else output[:500]
                    serialized_steps.append({
                        'tool': getattr(action, 'tool', 'unknown'),
                        'tool_input': getattr(action, 'tool_input', {}),
                        'output': output_str
                    })
                else:
                    serialized_steps.append(str(step)[:500])
            copilot_result['intermediate_steps'] = serialized_steps
        
        # Build response
        response_data = {
            'transcript': transcript,
            'copilot_reply': copilot_result,
            'mime_type': mime_type,
            'length_bytes': len(audio_bytes),
            'session_id': session_id
        }
        
        logger.info(f"Voice request complete: transcript={len(transcript)} chars")
        return jsonify({'data': response_data, 'error': None}), 200
    
    except ValueError as e:
        logger.warning(f"Voice input validation error: {e}")
        return jsonify({'data': None, 'error': str(e)}), 400
    
    except RuntimeError as e:
        logger.error(f"Voice processing error: {e}")
        return jsonify({'data': None, 'error': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in voice endpoint: {e}", exc_info=True)
        return jsonify({
            'data': None,
            'error': 'Internal error while processing audio input'
        }), 500
