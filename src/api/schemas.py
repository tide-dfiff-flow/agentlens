"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Tool definition for agent execution."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for parameters")


class ExecuteRequest(BaseModel):
    """Request to execute an agent."""

    agent_type: str = Field(default="react", description="Agent type (react, planner)")
    task: str = Field(..., description="Task to execute")
    tools: List[ToolDefinition] = Field(default_factory=list, description="Available tools")
    max_iterations: int = Field(default=10, ge=1, le=100, description="Max iterations")
    model: str = Field(default="gpt-4", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    enable_tracing: bool = Field(default=True, description="Enable execution tracing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExecuteResponse(BaseModel):
    """Response from agent execution."""

    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Execution status")
    output: Optional[str] = Field(None, description="Agent output")
    duration_ms: float = Field(..., description="Execution duration in ms")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls made")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning steps")
    tokens_used: Dict[str, int] = Field(default_factory=dict, description="Token usage")
    trace_id: Optional[str] = Field(None, description="Associated trace ID")
    error: Optional[str] = Field(None, description="Error message if failed")


class TraceResponse(BaseModel):
    """Response containing a trace."""

    id: str
    request_id: str
    agent_type: str
    task: str
    status: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: float
    step_count: int
    steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response containing performance metrics."""

    request_id: str
    agent_type: str
    total_duration_ms: float
    llm_latency_ms: float
    tool_latency_ms: float
    token_usage: Dict[str, int]
    estimated_cost_usd: float
    error_count: int
    iterations: int
    steps_per_second: float


class DiagnosisResponse(BaseModel):
    """Response containing diagnosis results."""

    trace_id: str
    issues: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    timestamp: float


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
