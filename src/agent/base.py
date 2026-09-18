"""Base agent classes and interfaces for AgentLens.

This module provides the foundational abstractions for building agents,
including the base agent class, state management, and response structures.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


class AgentState(Enum):
    """Agent internal state phase."""

    INIT = "init"
    PLANNING = "planning"
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    FINALIZING = "finalizing"


@dataclass
class Tool:
    """Represents a tool that the agent can use.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        func: The actual function to execute
        parameters: JSON schema for tool parameters
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate tool definition."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not callable(self.func):
            raise ValueError("Tool function must be callable")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    """Represents a tool invocation during agent execution.

    Attributes:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        result: Result from the tool execution
        start_time: When the tool call started
        end_time: When the tool call ended
        error: Error message if tool call failed
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None
    success: bool = True

    def complete(self, result: Any, error: Optional[str] = None) -> None:
        """Mark tool call as complete."""
        self.end_time = time.time()
        self.result = result
        self.error = error
        self.success = error is None

    @property
    def duration_ms(self) -> float:
        """Get tool call duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@dataclass
class AgentResponse:
    """Response from agent execution.

    Attributes:
        request_id: Unique identifier for this request
        output: The final output from the agent
        status: Execution status
        state: Final agent state
        tool_calls: List of tool calls made during execution
        reasoning: The agent's reasoning trace
        tokens_used: Token usage statistics
        duration_ms: Total execution time in milliseconds
        error: Error message if execution failed
        metadata: Additional metadata
    """

    request_id: str
    output: Optional[str] = None
    status: AgentStatus = AgentStatus.IDLE
    state: AgentState = AgentState.INIT
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    tokens_used: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        return {
            "request_id": self.request_id,
            "output": self.output,
            "status": self.status.value,
            "state": self.state.value,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": str(tc.result)[:500] if tc.result else None,
                    "duration_ms": tc.duration_ms,
                    "success": tc.success,
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ],
            "reasoning": self.reasoning,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents in AgentLens.

    This class provides the foundational structure for agent implementations,
    including state management, tool handling, and execution tracing.

    Subclasses must implement the `_think` and `_act` methods to define
    the agent's behavior.

    Example:
        ```python
        class MyAgent(BaseAgent):
            def __init__(self, llm, tools=None):
                super().__init__(llm, tools)

            def _think(self, state: AgentState) -> str:
                # Implement reasoning logic
                return "Reasoning about the task..."

            def _act(self, thought: str) -> ToolCall:
                # Implement action logic
                return ToolCall(tool_name="search", arguments={"query": "test"})
        ```
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        tracing_enabled: bool = True,
    ):
        """Initialize the base agent.

        Args:
            llm: Language model instance for reasoning
            tools: List of tools available to the agent
            max_iterations: Maximum number of think-act cycles
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature for LLM
            tracing_enabled: Whether to enable execution tracing
        """
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Execution state
        self._current_state = AgentState.INIT
        self._request_id: Optional[str] = None
        self._start_time: float = 0.0
        self._tracing_enabled = tracing_enabled
        self._trace: List[Dict[str, Any]] = []

        # Tool call tracking
        self._pending_tool_call: Optional[ToolCall] = None

    @property
    def request_id(self) -> Optional[str]:
        """Get current request ID."""
        return self._request_id

    @property
    def current_state(self) -> AgentState:
        """Get current agent state."""
        return self._current_state

    @property
    def trace(self) -> List[Dict[str, Any]]:
        """Get execution trace."""
        return self._trace.copy()

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit.

        Args:
            tool: Tool to add
        """
        self.tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the agent's toolkit.

        Args:
            name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool if found, None otherwise
        """
        return self.tools.get(name)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools in JSON schema format.

        Returns:
            List of tool definitions
        """
        return [tool.to_dict() for tool in self.tools.values()]

    def _record_trace(
        self,
        state: AgentState,
        thought: Optional[str] = None,
        action: Optional[ToolCall] = None,
        observation: Optional[str] = None,
    ) -> None:
        """Record a step in the execution trace.

        Args:
            state: Current agent state
            thought: Reasoning thought
            action: Tool call action
            observation: Observation result
        """
        if not self._tracing_enabled:
            return

        self._trace.append({
            "timestamp": time.time(),
            "state": state.value,
            "thought": thought,
            "action": {
                "tool_name": action.tool_name,
                "arguments": action.arguments,
            } if action else None,
            "observation": observation,
        })

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute the agent on a task.

        This is the main entry point for running the agent. It manages
        the execution loop, state transitions, and response construction.

        Args:
            task: The task to execute
            context: Optional context information

        Returns:
            AgentResponse with execution results
        """
        self._request_id = str(uuid.uuid4())
        self._start_time = time.time()
        self._current_state = AgentState.INIT
        self._trace = []

        response = AgentResponse(request_id=self._request_id)

        try:
            # Build initial prompt
            prompt = self._build_prompt(task, context)

            # Main execution loop
            for iteration in range(self.max_iterations):
                # Planning phase
                self._current_state = AgentState.PLANNING
                self._record_trace(self._current_state)

                # Reasoning phase
                self._current_state = AgentState.REASONING
                thought = await self._think(prompt, iteration)
                response.reasoning.append(thought)
                self._record_trace(self._current_state, thought=thought)

                # Check if should act or finish
                if self._should_finish(thought):
                    self._current_state = AgentState.FINALIZING
                    response.output = self._extract_output(thought)
                    response.status = AgentStatus.COMPLETED
                    response.state = self._current_state
                    break

                # Acting phase
                self._current_state = AgentState.ACTING
                tool_call = await self._act(thought)
                response.tool_calls.append(tool_call)
                self._record_trace(self._current_state, thought=thought, action=tool_call)

                # Observing phase
                self._current_state = AgentState.OBSERVING
                observation = await self._execute_tool(tool_call)
                self._record_trace(self._current_state, action=tool_call, observation=observation)

                # Update prompt with new information
                prompt = self._update_prompt(prompt, thought, tool_call, observation)

                # Check for termination conditions
                if self._should_terminate(observation):
                    response.output = observation
                    response.status = AgentStatus.COMPLETED
                    response.state = AgentState.FINALIZING
                    break

            else:
                # Max iterations reached
                response.status = AgentStatus.MAX_ITERATIONS_REACHED
                response.output = "Maximum iterations reached without conclusion."
                response.state = AgentState.FINALIZING

        except Exception as e:
            response.status = AgentStatus.FAILED
            response.error = str(e)
            response.state = self._current_state

        finally:
            response.duration_ms = (time.time() - self._start_time) * 1000
            response.metadata = {
                "iterations": len(response.tool_calls) + 1,
                "trace": self._trace,
            }

        return response

    @abstractmethod
    async def _think(self, prompt: str, iteration: int) -> str:
        """Generate a thought based on current state.

        This method should use the LLM to generate reasoning about
        the current task state.

        Args:
            prompt: Current conversation prompt
            iteration: Current iteration number

        Returns:
            Reasoning thought from the LLM
        """
        pass

    @abstractmethod
    async def _act(self, thought: str) -> ToolCall:
        """Determine the next action based on thought.

        This method should parse the thought and determine which
        tool to call, if any.

        Args:
            thought: The reasoning thought from _think

        Returns:
            ToolCall to execute
        """
        pass

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            Observation result as string
        """
        tool = self.tools.get(tool_call.tool_name)
        if not tool:
            return f"Error: Tool '{tool_call.tool_name}' not found"

        try:
            result = tool.func(**tool_call.arguments)
            # Handle async results
            if hasattr(result, "__await__"):
                result = await result
            tool_call.complete(result)
            return str(result)
        except Exception as e:
            tool_call.complete(None, str(e))
            return f"Error: {str(e)}"

    def _build_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the initial prompt for the agent.

        Args:
            task: The task to execute
            context: Optional context

        Returns:
            Formatted prompt string
        """
        tools_desc = "\n".join(
            f"- {name}: {t.description}" for name, t in self.tools.items()
        )

        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"

        return f"""You are an AI agent with access to the following tools:

{tools_desc}

Task: {task}{context_str}

Respond with your reasoning followed by an action. If no tool is needed, indicate this clearly.
"""

    def _update_prompt(
        self,
        prompt: str,
        thought: str,
        action: ToolCall,
        observation: str,
    ) -> str:
        """Update prompt with new interaction.

        Args:
            prompt: Current prompt
            thought: Reasoning thought
            action: Tool call taken
            observation: Observation result

        Returns:
            Updated prompt string
        """
        return f"""{prompt}

Thought: {thought}
Action: {action.tool_name}({json.dumps(action.arguments)})
Observation: {observation}
"""

    def _should_finish(self, thought: str) -> bool:
        """Determine if the agent should finish based on thought.

        Args:
            thought: Current reasoning thought

        Returns:
            True if agent should complete
        """
        finish_keywords = ["final answer", "conclusion", "finished", "complete", "done"]
        thought_lower = thought.lower()
        return any(keyword in thought_lower for keyword in finish_keywords)

    def _should_terminate(self, observation: str) -> bool:
        """Determine if execution should terminate based on observation.

        Args:
            observation: Current observation

        Returns:
            True if should terminate
        """
        return observation.startswith("Error:") or "TERMINATE" in observation

    def _extract_output(self, thought: str) -> str:
        """Extract final output from thought.

        Args:
            thought: Final reasoning thought

        Returns:
            Extracted output string
        """
        # Simple extraction - in production would be more sophisticated
        lines = thought.split("\n")
        for line in lines:
            if any(keyword in line.lower() for keyword in ["final", "answer", "result"]):
                if ":" in line:
                    return line.split(":", 1)[1].strip()
        return thought.strip()
