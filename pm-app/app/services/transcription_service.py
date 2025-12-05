"""
Speech-to-text transcription service using Gemini.
"""

import logging
import tempfile
import os
from pathlib import Path

import google.generativeai as genai

from app.config import load_config

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
}

TRANSCRIPTION_PROMPT = (
    "Transcribe this audio accurately to plain text. "
    "Do not summarize, do not translate, and do not add commentary. "
    "Preserve domain terms like RUL, failure probability, torque, maintenance, etc. "
    "Return only the transcript text."
)

_genai_configured = False


def _ensure_genai_configured() -> None:
    """Configure Gemini SDK once."""
    global _genai_configured
    if _genai_configured:
        return
    
    config = load_config()
    if not config.gemini.api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    
    genai.configure(api_key=config.gemini.api_key)
    _genai_configured = True
    logger.info("Gemini SDK configured for transcription")


def validate_mime_type(mime_type: str) -> bool:
    """Check if mime type is allowed."""
    return mime_type in ALLOWED_MIME_TYPES


def transcribe_audio_bytes(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    model_name: str | None = None,
) -> str:
    """
    Transcribe audio bytes to text using Gemini.
    
    Args:
        audio_bytes: Raw audio data
        mime_type: Audio MIME type (default: audio/webm)
        model_name: Gemini model to use (default: from config)
    
    Returns:
        Plain text transcript
    
    Raises:
        ValueError: Invalid input (empty bytes, unsupported mime type)
        RuntimeError: API key missing or transcription failed
    """
    if not audio_bytes:
        raise ValueError("Audio bytes cannot be empty")
    
    if not validate_mime_type(mime_type):
        raise ValueError(f"Unsupported audio mime type: {mime_type}")
    
    _ensure_genai_configured()
    
    config = load_config()
    model_name = model_name or config.gemini.stt_model
    
    # Determine file extension from mime type
    ext_map = {
        "audio/webm": ".webm",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
    }
    extension = ext_map.get(mime_type, ".webm")
    
    try:
        # Write to temp file for Gemini upload
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        logger.info(f"Transcribing audio: {len(audio_bytes)} bytes, {mime_type}, model={model_name}")
        
        # Upload file to Gemini
        audio_file = genai.upload_file(tmp_path, mime_type=mime_type)
        
        # Generate transcription
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            [audio_file, TRANSCRIPTION_PROMPT],
            request_options={"timeout": config.gemini.timeout}
        )
        
        # Clean up uploaded file
        try:
            audio_file.delete()
        except Exception:
            pass
        
        transcript = response.text.strip()
        logger.info(f"Transcription complete: {len(transcript)} characters")
        
        return transcript
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise RuntimeError(f"Transcription failed: {e}") from e
    
    finally:
        # Clean up temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
