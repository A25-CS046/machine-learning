"""
LangChain agent service for predictive maintenance copilot.

This module implements the core agent logic using LangChain's OpenAI Functions Agent
with Gemini 2.5 Pro as the LLM backend.
"""

import logging
import os
from typing import Any
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

from app.config import load_config
from app.services.agent_tools import get_all_tools

logger = logging.getLogger(__name__)


# ============================================================================
# System Prompt for Maintenance Copilot
# ============================================================================

MAINTENANCE_COPILOT_SYSTEM_PROMPT = """You are an AI assistant for the AEGIS Predictive Maintenance system.

Your role is to help industrial engineers understand equipment health, predict failures, and optimize maintenance schedules.

## CRITICAL RULES:
1. ALWAYS use the provided tools for predictions and calculations. NEVER make up or estimate numerical values.
2. If a tool call fails or returns an error, explain the error to the user and suggest next steps (manual inspection, data verification, etc.).
3. When asked about equipment status, health, or to "check" a unit, ALWAYS call BOTH predict_failure AND predict_rul tools to give a complete picture.
4. Only ask for clarification if the unit_id is truly missing. If the user provides a unit ID, proceed with tool calls immediately.
5. Provide actionable maintenance recommendations based on tool results.
6. If confidence is low or data is insufficient, recommend manual verification by maintenance team.
7. Be concise but thorough. Use technical language appropriate for industrial engineers.
8. When scheduling maintenance, prioritize high-risk and low-RUL equipment first.

## Available Tools:
- **predict_failure**: Get failure probability and failure type for a specific unit
- **predict_rul**: Get remaining useful life forecast in hours for a specific unit
- **optimize_schedule**: Generate optimal maintenance schedule for multiple units

## Response Guidelines:
- For failure predictions: Report probability as percentage, explain failure type, provide urgency level
- For RUL predictions: Report hours and days, categorize urgency (critical/high/medium/low)
- For scheduling: Summarize units scheduled, highlight high-priority items, note any conflicts
- Always include your reasoning and confidence level
- Suggest preventive actions based on predictions
- When a user asks to "check" a unit, provide BOTH failure probability AND RUL in your response

## Example Interactions:
User: "Check unit L56614/9435"
Assistant: [Calls predict_failure AND predict_rul] "Unit L56614/9435 status: Failure probability is 82% (Tool Wear Failure predicted). Remaining useful life: 18 hours (CRITICAL). Recommendation: Schedule immediate maintenance within 12 hours."

User: "What's the failure risk for unit M29501?"
Assistant: [Calls predict_failure] "Unit M29501 has a 15% failure probability - low risk. No immediate action required. Next scheduled check in 7 days."

User: "Schedule maintenance for units L56614, M29501, H30221"
Assistant: [Calls optimize_schedule] "Created schedule SCH_A4F2B8. Scheduled 3 units: L56614 (critical, tomorrow 08:00), M29501 (high priority, day after tomorrow), H30221 (routine, end of week)."
"""


# ============================================================================
# Custom Callbacks for Logging & Observability
# ============================================================================

class MaintenanceAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for maintenance agent events."""
    
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs) -> None:
        """Log when a tool execution starts."""
        tool_name = serialized.get('name', 'unknown')
        logger.info(f"[AGENT] Tool started: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log when a tool execution completes."""
        logger.info(f"[AGENT] Tool completed successfully")
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Log when a tool execution fails."""
        logger.error(f"[AGENT] Tool execution failed: {error}")
    
    def on_agent_action(self, action, **kwargs) -> None:
        """Log agent actions."""
        logger.info(f"[AGENT] Action: {action.tool} with inputs: {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Log when agent completes."""
        logger.info(f"[AGENT] Agent finished: {finish.return_values.get('output', '')[:100]}")


# ============================================================================
# Agent Factory & Management
# ============================================================================

class MaintenanceAgentService:
    """Service for managing LangChain maintenance copilot agent."""
    
    def __init__(self):
        self.config = load_config()
        self._agent_cache: dict[str, AgentExecutor] = {}
        self._llm = None
    
    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or create Gemini LLM instance."""
        if self._llm is None:
            if not self.config.gemini.api_key:
                raise ValueError("GEMINI_API_KEY not configured")
            
            # Use gemini-2.5-pro model
            model_name = "gemini-2.5-pro"
            
            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.config.gemini.api_key,
                temperature=0.0,
                convert_system_message_to_human=False,
                max_retries=3
            )
            
            logger.info(f"Initialized Gemini LLM: {model_name}")
        
        return self._llm
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create agent prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", MAINTENANCE_COPILOT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Get conversation memory for session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            ConversationBufferWindowMemory instance
        """
        # Use PostgresChatMessageHistory for persistent storage
        message_history = PostgresChatMessageHistory(
            connection_string=self.config.database.url,
            session_id=session_id,
            table_name='conversation_history'
        )
        
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=message_history,
            k=10,  # Keep last 10 messages
            output_key="output"
        )
    
    def create_agent(self, session_id: str = "default") -> AgentExecutor:
        """
        Create a new agent executor for a session.
        
        Args:
            session_id: Unique session identifier for conversation memory
        
        Returns:
            AgentExecutor instance
        """
        # Check cache
        if session_id in self._agent_cache:
            logger.debug(f"Returning cached agent for session: {session_id}")
            return self._agent_cache[session_id]
        
        logger.info(f"Creating new agent for session: {session_id}")
        
        # Get components
        llm = self._get_llm()
        tools = get_all_tools()
        prompt = self._create_prompt()
        memory = self._get_memory(session_id)
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create executor with callbacks
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,  # Enable logging
            max_iterations=5,  # Prevent infinite loops
            max_execution_time=60,  # 60 second timeout
            handle_parsing_errors=True,  # Graceful error handling
            return_intermediate_steps=True,  # Return tool execution details
            callbacks=[MaintenanceAgentCallbackHandler()],
            early_stopping_method="generate",  # Better Gemini compatibility
        )
        
        # Cache agent
        self._agent_cache[session_id] = agent_executor
        
        logger.info(f"Agent created successfully for session: {session_id}")
        return agent_executor
    
    def invoke_agent(
        self,
        session_id: str,
        user_message: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Invoke the agent with a user message.
        
        Args:
            session_id: Session identifier
            user_message: User's input message
            context: Optional context dictionary
        
        Returns:
            Dict with agent response and metadata
        """
        try:
            agent_executor = self.create_agent(session_id)
            
            # Prepare input
            agent_input = {"input": user_message}
            if context:
                agent_input["context"] = context
            
            # Execute agent
            start_time = datetime.now(timezone.utc)
            result = agent_executor.invoke(agent_input)
            end_time = datetime.now(timezone.utc)
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract results
            response = {
                'success': True,
                'reply': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', []),
                'execution_time_seconds': execution_time,
                'session_id': session_id,
                'timestamp': end_time.isoformat()
            }
            
            logger.info(f"Agent invocation successful ({execution_time:.2f}s)")
            return response
        
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if successful
        """
        try:
            if session_id in self._agent_cache:
                agent = self._agent_cache[session_id]
                agent.memory.clear()
                del self._agent_cache[session_id]
            
            logger.info(f"Session cleared: {session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False


# ============================================================================
# Singleton Instance
# ============================================================================

_agent_service_instance: MaintenanceAgentService | None = None


def get_agent_service() -> MaintenanceAgentService:
    """
    Get singleton instance of MaintenanceAgentService.
    
    Returns:
        MaintenanceAgentService instance
    """
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = MaintenanceAgentService()
    return _agent_service_instance
