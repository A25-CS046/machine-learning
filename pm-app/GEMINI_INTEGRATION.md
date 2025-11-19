# Gemini 2.5 Pro Copilot Integration Plan

## Overview

This document describes the modular implementation of Gemini 2.5 Pro as the conversational AI copilot for the AEGIS Predictive Maintenance system. The copilot uses HTTP-based tool calling to invoke backend prediction and optimization functions.

## Architecture

### High-Level Flow

```
┌─────────────┐                                    ┌──────────────────┐
│  Frontend   │──── POST /copilot/chat ────────────▶│  Flask Backend   │
│   (User)    │◀─── Natural Language Response ─────│  (pm-app)        │
└─────────────┘                                    └────────┬─────────┘
                                                            │
                                                            ▼
                                                   ┌────────────────────┐
                                                   │ GeminiCopilotClient│
                                                   │ (copilot_service.py)│
                                                   └────────┬───────────┘
                                                            │
                  ┌─────────────────────────────────────────┼─────────────────────────────────┐
                  │                                         │                                 │
                  ▼                                         ▼                                 ▼
         ┌────────────────┐                       ┌─────────────────┐            ┌────────────────────┐
         │ Gemini 2.5 Pro │                       │  Tool Executor  │            │  Backend Services  │
         │   (HTTP API)   │──tool_calls──────────▶│ (copilot_tools) │───────────▶│  - classification  │
         │                │◀──tool_results────────│                 │◀───────────│  - forecast        │
         └────────────────┘                       └─────────────────┘            │  - optimizer       │
                                                                                   └────────────────────┘
```

## Implementation Details

### 1. Service Layer: `app/services/copilot_service.py`

#### Key Components

**GeminiCopilotClient Class**
```python
class GeminiCopilotClient:
    def __init__(self):
        # Loads GEMINI_API_KEY, GEMINI_MODEL_NAME from config
    
    def chat(self, messages: list[dict], enable_tools: bool) -> dict:
        # Sends request to Gemini API with tool definitions
        # Returns full Gemini response
    
    def extract_tool_calls(self, gemini_response: dict) -> list[dict]:
        # Parses functionCall from Gemini response
    
    def extract_text_response(self, gemini_response: dict) -> str:
        # Extracts natural language text from response
```

**System Prompt**
```python
GEMINI_SYSTEM_PROMPT = """You are an AI assistant for the AEGIS Predictive Maintenance system.

CRITICAL RULES:
1. ALWAYS use the provided tools for predictions. NEVER make up numerical values.
2. If a tool call fails, explain the error and suggest manual inspection.
3. When asked about failure risk or RUL, ALWAYS call the appropriate tool.
4. Ask for missing parameters (product_id, unit_id) before making tool calls.
..."""
```

This prompt is injected as `systemInstruction` in every Gemini API request.

**Tool Definitions**
```python
GEMINI_TOOL_DEFINITIONS = [
    {
        "name": "predict_failure",
        "description": "Predict equipment failure probability...",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", ...},
                "unit_id": {"type": "string", ...},
                "horizon_hours": {"type": "integer", "default": 24}
            },
            "required": ["product_id", "unit_id"]
        }
    },
    # ... predict_rul, optimize_schedule
]
```

These are sent to Gemini in the `tools` field using the `functionDeclarations` format.

### 2. Route Layer: `app/routes/copilot_tools.py`

#### Tool Endpoints (for direct invocation)

```python
POST /copilot/tools/predict_failure
POST /copilot/tools/predict_rul
POST /copilot/tools/optimize_schedule
```

These are simplified wrappers around the main service functions, designed to return LLM-friendly responses.

**Example: predict_failure tool response**
```json
{
  "data": {
    "product_id": "PROD_001",
    "unit_id": "UNIT_001",
    "failure_probability": 0.95,
    "will_fail": true,
    "failure_type": "Equipment Failure Predicted",
    "recommendation": "Schedule immediate inspection"
  },
  "error": null
}
```

#### Main Copilot Endpoint

```python
POST /copilot/chat
```

**Request Format**
```json
{
  "messages": [
    {"role": "user", "content": "What's the failure risk for UNIT_001?"},
    {"role": "assistant", "content": "Let me check that for you."},
    {"role": "user", "content": "Also show me the RUL forecast."}
  ],
  "context": {
    "product_id": "PROD_001"
  }
}
```

**Response Format**
```json
{
  "data": {
    "reply": "Based on the analysis, UNIT_001 has a 95% failure probability...",
    "tool_calls": [
      {"name": "predict_failure", "arguments": {"product_id": "PROD_001", "unit_id": "UNIT_001"}}
    ],
    "tool_results": [
      {"tool_name": "predict_failure", "success": true, "result": {...}}
    ],
    "raw_model_response": {...}
  },
  "error": null
}
```

### 3. Request Flow (Step-by-Step)

#### Single-Turn Conversation

1. **User sends message**
   ```
   POST /copilot/chat
   {"messages": [{"role": "user", "content": "Check UNIT_001 failure risk"}]}
   ```

2. **Backend calls Gemini API**
   ```python
   client = get_gemini_client()
   gemini_response = client.chat(messages, enable_tools=True)
   ```

3. **Gemini returns tool call**
   ```json
   {
     "candidates": [{
       "content": {
         "parts": [{
           "functionCall": {
             "name": "predict_failure",
             "args": {"product_id": "PROD_001", "unit_id": "UNIT_001"}
           }
         }]
       }
     }]
   }
   ```

4. **Backend executes tool**
   ```python
   tool_calls = client.extract_tool_calls(gemini_response)
   for tool_call in tool_calls:
       if tool_call['name'] == 'predict_failure':
           result = predict_failure(
               model_name='xgb_classifier',
               product_id=tool_call['arguments']['product_id'],
               unit_id=tool_call['arguments']['unit_id']
           )
   ```

5. **Backend returns result**
   ```json
   {
     "data": {
       "reply": "UNIT_001 has a 95% failure probability. Immediate maintenance recommended.",
       "tool_calls": [...],
       "tool_results": [{"tool_name": "predict_failure", "success": true, "result": {...}}]
     },
     "error": null
   }
   ```

#### Multi-Turn Conversation (Future Enhancement)

For multi-turn conversations where Gemini needs to call tools and then generate a final answer, the backend would:

1. Detect tool calls in Gemini response
2. Execute tools and collect results
3. Append tool results to conversation history
4. Call Gemini API again with updated messages
5. Extract final text response
6. Return to frontend

This is **not currently implemented** but the architecture supports it.

### 4. Tool Execution Mapping

```python
# In copilot_tools.py
tool_name = tool_call['name']
tool_args = tool_call['arguments']

if tool_name == 'predict_failure':
    result = predict_failure(
        model_name='xgb_classifier',
        product_id=tool_args['product_id'],
        unit_id=tool_args['unit_id']
    )
elif tool_name == 'predict_rul':
    result = predict_rul(
        model_name='xgb_regressor',
        product_id=tool_args['product_id'],
        unit_id=tool_args['unit_id'],
        horizon_steps=tool_args.get('horizon_steps', 10)
    )
elif tool_name == 'optimize_schedule':
    result = optimize_maintenance_schedule(
        unit_list=tool_args['unit_list'],
        risk_threshold=tool_args.get('risk_threshold', 0.7),
        rul_threshold=tool_args.get('rul_threshold', 24.0),
        horizon_days=tool_args.get('horizon_days', 7)
    )
```

Tools are executed **synchronously** in the same request context, not via separate HTTP calls.

### 5. Error Handling

#### Tool Execution Errors
```python
try:
    result = predict_failure(...)
    tool_results.append({'tool_name': 'predict_failure', 'success': True, 'result': result})
except Exception as e:
    logger.error(f"Tool execution failed: {e}")
    tool_results.append({'tool_name': 'predict_failure', 'success': False, 'error': str(e)})
```

#### Gemini API Errors
```python
try:
    gemini_response = client.chat(messages)
except RuntimeError as e:
    return jsonify({'data': None, 'error': str(e)}), 503
```

#### Fallback Behavior
When a tool fails, the system prompt instructs Gemini to:
- Explain the error to the user
- Recommend manual inspection
- Suggest a safe maintenance window

### 6. Configuration

**Required Environment Variables**
```bash
GEMINI_API_KEY=your_google_ai_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_TIMEOUT=60
```

**Obtaining API Key**
1. Go to https://aistudio.google.com/apikey
2. Create a new API key
3. Set `GEMINI_API_KEY` environment variable

### 7. Testing Strategy

#### Unit Tests (Mock Gemini API)
```python
def test_copilot_chat_with_tool_call(client, mocker):
    mock_gemini_response = {
        'candidates': [{
            'content': {
                'parts': [{
                    'functionCall': {
                        'name': 'predict_failure',
                        'args': {'product_id': 'PROD_001', 'unit_id': 'UNIT_001'}
                    }
                }]
            }
        }]
    }
    
    mocker.patch('app.services.copilot_service.requests.post', return_value=mock_gemini_response)
    
    response = client.post('/copilot/chat', json={
        'messages': [{'role': 'user', 'content': 'Check UNIT_001'}]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert len(data['data']['tool_calls']) == 1
```

#### Integration Tests (Real Gemini API - Optional)
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason='No API key')
def test_copilot_real_gemini(client):
    response = client.post('/copilot/chat', json={
        'messages': [{'role': 'user', 'content': 'Hello'}]
    })
    assert response.status_code == 200
```

### 8. Performance Considerations

**Latency Breakdown**
- Network to Gemini API: ~500-1000ms
- Gemini inference: ~1000-3000ms
- Tool execution: ~100-500ms (depends on DB queries)
- **Total**: ~1600-4500ms per request

**Optimization Strategies**
1. Cache common queries (e.g., recent predictions)
2. Parallel tool execution (if multiple tools called)
3. Stream responses (not currently supported by Gemini HTTP API)
4. Use Gemini Flash for faster responses

### 9. Deployment

**Docker Compose**
```yaml
services:
  app:
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      GEMINI_MODEL_NAME: gemini-2.0-flash-exp
```

**Kubernetes ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pm-app-config
data:
  GEMINI_MODEL_NAME: "gemini-2.0-flash-exp"
  GEMINI_API_URL: "https://generativelanguage.googleapis.com/v1beta"
```

**Kubernetes Secret**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: pm-app-secrets
type: Opaque
data:
  GEMINI_API_KEY: <base64-encoded-key>
```

### 10. Future Enhancements

#### Multi-Turn Conversations
Implement conversation state management:
```python
def chat_with_multi_turn(messages, max_turns=5):
    for turn in range(max_turns):
        response = gemini_client.chat(messages)
        tool_calls = extract_tool_calls(response)
        
        if not tool_calls:
            return extract_text_response(response)
        
        tool_results = execute_tools(tool_calls)
        messages.append({'role': 'function', 'content': json.dumps(tool_results)})
    
    return "Maximum turns reached"
```

#### Async Tool Execution
```python
import asyncio

async def execute_tools_async(tool_calls):
    tasks = [execute_tool_async(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

#### Streaming Responses
```python
@copilot_bp.route('/copilot/chat/stream', methods=['POST'])
def copilot_chat_stream():
    def generate():
        for chunk in gemini_client.chat_stream(messages):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

#### Context Management
```python
class ConversationContext:
    def __init__(self, user_id):
        self.user_id = user_id
        self.messages = []
        self.default_product_id = None
        self.default_unit_id = None
    
    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})
    
    def get_context_for_tools(self):
        return {
            'product_id': self.default_product_id,
            'unit_id': self.default_unit_id
        }
```

## Summary

The Gemini 2.5 Pro integration provides:

✅ **Modular design** - Easy to swap Gemini for another LLM  
✅ **Tool calling** - Gemini invokes backend functions via structured tool definitions  
✅ **Safety guardrails** - System prompt prevents hallucinated predictions  
✅ **Error handling** - Graceful degradation on tool or API failures  
✅ **Testability** - Mockable HTTP client for unit tests  
✅ **Production-ready** - Proper timeout handling, logging, and error responses  

The implementation is **complete and functional** for single-turn conversations with tool calling. Multi-turn and streaming are documented for future enhancement.
