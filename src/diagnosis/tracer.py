"""Execution tracer for AgentLens.

Captures step-by-step execution traces of agent runs,
enabling debugging, analysis, and visualization of agent behavior.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TraceEventType(Enum):
    """Types of events in a trace."""

    THOUGHT = "thought"
    ACTION = "action"
    ACTION_RESULT = "action_result"
    OBSERVATION = "observation"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"


@dataclass
class TraceStep:
    """A single step in an execution trace.

    Attributes:
        id: Unique identifier for this step
        event_type: Type of event that occurred
        timestamp: When the event occurred
        state: Agent state at this step
        thought: Reasoning thought (if applicable)
        action: Action taken (if applicable)
        observation: Observation result (if applicable)
        tool_name: Name of tool used (if applicable)
        tool_args: Arguments passed to tool (if applicable)
        tool_result: Result from tool (if applicable)
        duration_ms: Duration of this step in milliseconds
        metadata: Additional metadata
        error: Error message (if applicable)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: TraceEventType = TraceEventType.THOUGHT
    timestamp: float = field(default_factory=time.time)
    state: str = "init"
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "state": self.state,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class Trace:
    """Complete execution trace for an agent run.

    Attributes:
        id: Unique trace identifier
        request_id: Associated request ID
        agent_type: Type of agent (e.g., "ReAct", "Planner")
        task: The task being executed
        start_time: When execution started
        end_time: When execution ended
        steps: All trace steps
        status: Final status
        total_duration_ms: Total execution time
        metadata: Additional trace metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    agent_type: str = "unknown"
    task: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    steps: List[TraceStep] = field(default_factory=list)
    status: str = "running"
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)

    def complete(self, status: str = "completed") -> None:
        """Mark trace as complete."""
        self.end_time = time.time()
        self.status = status
        self.total_duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "request_id": self.request_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "datetime": datetime.fromtimestamp(self.start_time).isoformat(),
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata,
            "step_count": len(self.steps),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the trace."""
        tool_calls = [s for s in self.steps if s.event_type == TraceEventType.TOOL_END]
        thoughts = [s for s in self.steps if s.event_type == TraceEventType.THOUGHT]

        return {
            "id": self.id,
            "status": self.status,
            "duration_ms": self.total_duration_ms,
            "step_count": len(self.steps),
            "thought_count": len(thoughts),
            "tool_call_count": len(tool_calls),
            "tool_names": list(set(s.tool_name for s in tool_calls if s.tool_name)),
        }


class ExecutionTracer:
    """Captures execution traces for agent runs.

    Provides context managers and decorators for easy trace capture,
    as well as storage and retrieval of traces.
    """

    def __init__(self, max_traces: int = 1000):
        """Initialize the tracer.

        Args:
            max_traces: Maximum number of traces to store
        """
        self.max_traces = max_traces
        self._traces: Dict[str, Trace] = {}
        self._active_trace: Optional[Trace] = None
        self._step_start_times: Dict[str, float] = {}

    def start_trace(
        self,
        request_id: str,
        agent_type: str,
        task: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """Start a new execution trace.

        Args:
            request_id: Request identifier
            agent_type: Type of agent
            task: Task being executed
            metadata: Optional metadata

        Returns:
            New Trace object
        """
        trace = Trace(
            request_id=request_id,
            agent_type=agent_type,
            task=task,
            metadata=metadata or {},
        )
        self._active_trace = trace
        self._traces[trace.id] = trace

        # Evict old traces if over capacity
        while len(self._traces) > self.max_traces:
            oldest_id = min(self._traces.keys(), key=lambda x: self._traces[x].start_time)
            del self._traces[oldest_id]

        return trace

    def end_trace(self, status: str = "completed") -> Optional[Trace]:
        """End the current trace.

        Args:
            status: Final status

        Returns:
            Completed Trace or None
        """
        if self._active_trace:
            self._active_trace.complete(status)
            trace = self._active_trace
            self._active_trace = None
            return trace
        return None

    def record_thought(
        self,
        thought: str,
        state: str = "reasoning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceStep]:
        """Record a thought step.

        Args:
            thought: The reasoning thought
            state: Current agent state
            metadata: Additional metadata

        Returns:
            Created TraceStep or None
        """
        if not self._active_trace:
            return None

        step = TraceStep(
            event_type=TraceEventType.THOUGHT,
            state=state,
            thought=thought,
            metadata=metadata or {},
        )
        self._active_trace.add_step(step)
        return step

    def record_action(
        self,
        action: str,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        state: str = "acting",
    ) -> Optional[TraceStep]:
        """Record an action step.

        Args:
            action: Action description
            tool_name: Name of tool
            tool_args: Tool arguments
            state: Current agent state

        Returns:
            Created TraceStep or None
        """
        if not self._active_trace:
            return None

        step = TraceStep(
            event_type=TraceEventType.ACTION,
            state=state,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args or {},
        )
        self._active_trace.add_step(step)
        self._step_start_times[step.id] = time.time()
        return step

    def record_observation(
        self,
        observation: str,
        tool_result: Optional[str] = None,
        state: str = "observing",
    ) -> Optional[TraceStep]:
        """Record an observation step.

        Args:
            observation: Observation description
            tool_result: Result from tool
            state: Current agent state

        Returns:
            Created TraceStep or None
        """
        if not self._active_trace:
            return None

        # Calculate duration for the last action
        duration_ms = 0.0
        if self._active_trace.steps:
            last_step = self._active_trace.steps[-1]
            if last_step.id in self._step_start_times:
                duration_ms = (time.time() - self._step_start_times.pop(last_step.id)) * 1000

        step = TraceStep(
            event_type=TraceEventType.OBSERVATION,
            state=state,
            observation=observation,
            tool_result=tool_result,
            duration_ms=duration_ms,
        )
        self._active_trace.add_step(step)
        return step

    def record_error(
        self,
        error: str,
        state: str = "error",
    ) -> Optional[TraceStep]:
        """Record an error.

        Args:
            error: Error message
            state: Agent state at error

        Returns:
            Created TraceStep or None
        """
        if not self._active_trace:
            return None

        step = TraceStep(
            event_type=TraceEventType.ERROR,
            state=state,
            error=error,
        )
        self._active_trace.add_step(step)
        return step

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace if found, None otherwise
        """
        return self._traces.get(trace_id)

    def get_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Trace]:
        """Get traces with optional filtering.

        Args:
            limit: Maximum number to return
            offset: Offset for pagination
            agent_type: Filter by agent type
            status: Filter by status

        Returns:
            List of matching traces
        """
        traces = list(self._traces.values())

        # Apply filters
        if agent_type:
            traces = [t for t in traces if t.agent_type == agent_type]
        if status:
            traces = [t for t in traces if t.status == status]

        # Sort by start_time descending
        traces.sort(key=lambda x: x.start_time, reverse=True)

        return traces[offset:offset + limit]

    def get_active_trace(self) -> Optional[Trace]:
        """Get the currently active trace.

        Returns:
            Active trace or None
        """
        return self._active_trace

    def clear_traces(self) -> None:
        """Clear all stored traces."""
        self._traces.clear()
        self._active_trace = None

    def export_trace(self, trace_id: str, format: str = "json") -> Optional[str]:
        """Export a trace in specified format.

        Args:
            trace_id: Trace to export
            format: Output format (json, text)

        Returns:
            Exported trace string or None
        """
        trace = self._traces.get(trace_id)
        if not trace:
            return None

        if format == "json":
            return json.dumps(trace.to_dict(), indent=2)
        elif format == "text":
            return self._format_as_text(trace)
        return None

    def _format_as_text(self, trace: Trace) -> str:
        """Format trace as readable text.

        Args:
            trace: Trace to format

        Returns:
            Formatted text
        """
        lines = [
            f"=== Trace {trace.id} ===",
            f"Agent: {trace.agent_type}",
            f"Task: {trace.task}",
            f"Status: {trace.status}",
            f"Duration: {trace.total_duration_ms:.2f}ms",
            f"Steps: {len(trace.steps)}",
            "",
        ]

        for i, step in enumerate(trace.steps):
            lines.append(f"[{i+1}] {step.event_type.value.upper()} ({step.state})")
            if step.thought:
                lines.append(f"    Thought: {step.thought[:100]}...")
            if step.tool_name:
                lines.append(f"    Tool: {step.tool_name}")
            if step.observation:
                lines.append(f"    Observation: {step.observation[:100]}...")
            if step.error:
                lines.append(f"    ERROR: {step.error}")
            lines.append("")

        return "\n".join(lines)
