"""
Gemini API client for direct copilot interactions.

Provides low-level access to Gemini API with tool definitions.
For production, use langchain_agent_service.py instead.
"""

import logging
import json
from typing import Any
import requests
from app.config import load_config

logger = logging.getLogger(__name__)


GEMINI_SYSTEM_PROMPT = """You are an AI assistant for the AEGIS Predictive Maintenance system.
Help users understand equipment health, predict failures, and optimize maintenance.

Rules:
1. Use provided tools for predictions. Never estimate values.
2. Ask for missing parameters before tool calls.
3. Provide actionable maintenance recommendations.

Tools: predict_failure, predict_rul, optimize_schedule"""


GEMINI_TOOL_DEFINITIONS = [
    {
        "name": "predict_failure",
        "description": "Predict failure probability and type for a unit. Returns probability (0-1) and failure type.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product identifier"},
                "unit_id": {"type": "string", "description": "Unit identifier"},
                "horizon_hours": {"type": "integer", "description": "Prediction horizon (default: 24)", "default": 24}
            },
            "required": ["product_id", "unit_id"]
        }
    },
    {
        "name": "predict_rul",
        "description": "Predict remaining useful life in hours. Returns current RUL and forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product identifier"},
                "unit_id": {"type": "string", "description": "Unit identifier"},
                "horizon_steps": {"type": "integer", "description": "Forecast steps (default: 10)", "default": 10}
            },
            "required": ["product_id", "unit_id"]
        }
    },
    {
        "name": "optimize_schedule",
        "description": "Generate optimized maintenance schedule for multiple units.",
        "parameters": {
            "type": "object",
            "properties": {
                "unit_list": {
                    "type": "array",
                    "description": "Units to schedule",
                    "items": {
                        "type": "object",
                        "properties": {"product_id": {"type": "string"}, "unit_id": {"type": "string"}},
                        "required": ["product_id", "unit_id"]
                    }
                },
                "risk_threshold": {"type": "number", "description": "Failure threshold (default: 0.7)", "default": 0.7},
                "rul_threshold": {"type": "number", "description": "RUL threshold hours (default: 24)", "default": 24.0},
                "horizon_days": {"type": "integer", "description": "Planning horizon (default: 7)", "default": 7}
            },
            "required": ["unit_list"]
        }
    }
]


class GeminiCopilotClient:
    """Direct Gemini API client with tool calling support."""
    
    def __init__(self):
        self.config = load_config()
        if not self.config.gemini.api_key:
            logger.warning("GEMINI_API_KEY not configured")
        self.api_key = self.config.gemini.api_key
        self.model_name = self.config.gemini.model_name
        self.api_url = self.config.gemini.api_url
        self.timeout = self.config.gemini.timeout
    
    def _build_request_payload(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Build Gemini API request payload."""
        contents = []
        
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append({
                'role': role,
                'parts': [{'text': msg['content']}]
            })
        
        payload = {
            'contents': contents,
            'systemInstruction': {
                'parts': [{'text': GEMINI_SYSTEM_PROMPT}]
            }
        }
        
        if tools:
            payload['tools'] = [{
                'functionDeclarations': tools
            }]
        
        return payload
    
    def chat(self, messages: list[dict], enable_tools: bool = True) -> dict:
        """Send chat request to Gemini API with optional tool calling."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        url = f"{self.api_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        
        tools = GEMINI_TOOL_DEFINITIONS if enable_tools else None
        payload = self._build_request_payload(messages, tools)
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}")
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def extract_tool_calls(self, gemini_response: dict) -> list[dict]:
        """Extract tool calls from Gemini response."""
        tool_calls = []
        
        if 'candidates' not in gemini_response:
            return tool_calls
        
        for candidate in gemini_response['candidates']:
            if 'content' not in candidate:
                continue
            
            for part in candidate['content'].get('parts', []):
                if 'functionCall' in part:
                    func_call = part['functionCall']
                    tool_calls.append({
                        'name': func_call['name'],
                        'arguments': func_call.get('args', {})
                    })
        
        return tool_calls
    
    def extract_text_response(self, gemini_response: dict) -> str | None:
        """Extract text response from Gemini response."""
        if 'candidates' not in gemini_response:
            return None
        
        for candidate in gemini_response['candidates']:
            if 'content' not in candidate:
                continue
            
            for part in candidate['content'].get('parts', []):
                if 'text' in part:
                    return part['text']
        
        return None


_client_instance: GeminiCopilotClient | None = None


def get_gemini_client() -> GeminiCopilotClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiCopilotClient()
    return _client_instance
