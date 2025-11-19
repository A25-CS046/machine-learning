import logging
import json
from typing import Any
import requests
from app.config import load_config

logger = logging.getLogger(__name__)


GEMINI_SYSTEM_PROMPT = """You are an AI assistant for the AEGIS Predictive Maintenance system.

Your role is to help users understand equipment health, predict failures, and optimize maintenance schedules.

CRITICAL RULES:
1. ALWAYS use the provided tools for predictions and calculations. NEVER make up or estimate numerical values.
2. If a tool call fails or returns an error, explain the error to the user and suggest manual inspection.
3. When asked about failure risk or RUL, ALWAYS call the appropriate prediction tool.
4. Ask for missing parameters (product_id, unit_id) before making tool calls.
5. Provide actionable maintenance recommendations based on tool results.
6. If confidence is low or data is insufficient, recommend manual verification.

Available tools:
- predict_failure: Get failure probability and failure type for a specific unit
- predict_rul: Get remaining useful life forecast for a specific unit
- optimize_schedule: Generate optimal maintenance schedule for multiple units

Always provide clear, concise explanations of prediction results and their implications for maintenance planning."""


GEMINI_TOOL_DEFINITIONS = [
    {
        "name": "predict_failure",
        "description": "Predict equipment failure probability and failure type for a given unit. Returns failure probability (0-1), predicted failure type if failure is likely, and confidence metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product identifier for the equipment"
                },
                "unit_id": {
                    "type": "string",
                    "description": "The specific unit identifier within the product line"
                },
                "horizon_hours": {
                    "type": "integer",
                    "description": "Prediction horizon in hours (default: 24)",
                    "default": 24
                }
            },
            "required": ["product_id", "unit_id"]
        }
    },
    {
        "name": "predict_rul",
        "description": "Predict remaining useful life (RUL) in hours for equipment. Returns current RUL estimate and forecast for upcoming timesteps.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product identifier for the equipment"
                },
                "unit_id": {
                    "type": "string",
                    "description": "The specific unit identifier within the product line"
                },
                "horizon_steps": {
                    "type": "integer",
                    "description": "Number of future timesteps to forecast (default: 10)",
                    "default": 10
                }
            },
            "required": ["product_id", "unit_id"]
        }
    },
    {
        "name": "optimize_schedule",
        "description": "Generate an optimized maintenance schedule for multiple units based on failure risk and RUL thresholds. Returns prioritized maintenance recommendations with time windows.",
        "parameters": {
            "type": "object",
            "properties": {
                "unit_list": {
                    "type": "array",
                    "description": "List of units to schedule maintenance for",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "unit_id": {"type": "string"}
                        },
                        "required": ["product_id", "unit_id"]
                    }
                },
                "risk_threshold": {
                    "type": "number",
                    "description": "Failure probability threshold for scheduling (0-1, default: 0.7)",
                    "default": 0.7
                },
                "rul_threshold": {
                    "type": "number",
                    "description": "RUL threshold in hours for scheduling (default: 24.0)",
                    "default": 24.0
                },
                "horizon_days": {
                    "type": "integer",
                    "description": "Planning horizon in days (default: 7)",
                    "default": 7
                }
            },
            "required": ["unit_list"]
        }
    }
]


class GeminiCopilotClient:
    def __init__(self):
        self.config = load_config()
        
        if not self.config.gemini.api_key:
            logger.warning("GEMINI_API_KEY not configured. Copilot will not function.")
        
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
        """
        Send chat request to Gemini API.
        
        Args:
            messages: List of {role: str, content: str} message dicts
            enable_tools: Whether to enable tool calling
        
        Returns:
            Dictionary with Gemini response
        """
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
