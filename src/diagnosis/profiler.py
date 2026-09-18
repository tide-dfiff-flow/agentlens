"""Performance profiler for AgentLens.

Tracks performance metrics for agent executions including
latency, token usage, cost, and resource utilization.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics from an agent execution.

    Attributes:
        request_id: Associated request ID
        agent_type: Type of agent
        start_time: When execution started
        end_time: When execution ended
        total_duration_ms: Total execution time
        llm_latency_ms: Time spent in LLM calls
        tool_latency_ms: Time spent in tool execution
        token_usage: Token consumption breakdown
        estimated_cost: Estimated API cost
        error_count: Number of errors encountered
        iterations: Number of agent loops
        steps_per_second: Execution throughput
        metadata: Additional metrics
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: str = "unknown"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    estimated_cost: float = 0.0
    error_count: int = 0
    iterations: int = 0
    steps_per_second: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Token pricing (per 1M tokens) - example rates
    TOKEN_PRICING = {
        "gpt-4": {"prompt": 30.0, "completion": 60.0},
        "gpt-3.5-turbo": {"prompt": 1.5, "completion": 2.0},
        "claude-3": {"prompt": 3.0, "completion": 15.0},
    }

    def complete(self) -> None:
        """Mark metrics as complete and calculate derived metrics."""
        self.end_time = time.time()
        self.total_duration_ms = (self.end_time - self.start_time) * 1000

        if self.total_duration_ms > 0:
            self.steps_per_second = (self.iterations / self.total_duration_ms) * 1000

    def add_llm_latency(self, latency_ms: float) -> None:
        """Add LLM call latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.llm_latency_ms += latency_ms

    def add_tool_latency(self, latency_ms: float) -> None:
        """Add tool execution latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.tool_latency_ms += latency_ms

    def add_tokens(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "gpt-4",
    ) -> None:
        """Add token usage.

        Args:
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            model: Model name for cost calculation
        """
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += prompt_tokens + completion_tokens

        # Calculate cost
        pricing = self.TOKEN_PRICING.get(model, {"prompt": 0, "completion": 0})
        self.estimated_cost = (
            (self.token_usage["prompt_tokens"] / 1_000_000) * pricing["prompt"]
            + (self.token_usage["completion_tokens"] / 1_000_000) * pricing["completion"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "agent_type": self.agent_type,
            "start_time": self.start_time,
            "datetime": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "token_usage": self.token_usage,
            "estimated_cost_usd": round(self.estimated_cost, 6),
            "error_count": self.error_count,
            "iterations": self.iterations,
            "steps_per_second": round(self.steps_per_second, 4),
            "metadata": self.metadata,
        }


class PerformanceProfiler:
    """Collects and aggregates performance metrics.

    Provides profiling capabilities for agent executions with
    automatic metric collection and aggregation.
    """

    def __init__(self, default_model: str = "gpt-4"):
        """Initialize the profiler.

        Args:
            default_model: Default model for cost estimation
        """
        self.default_model = default_model
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._active_metrics: Optional[PerformanceMetrics] = None
        self._llm_call_times: List[float] = []
        self._tool_call_times: Dict[str, List[float]] = {}

    def start_profile(
        self,
        request_id: str,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        """Start profiling an execution.

        Args:
            request_id: Request identifier
            agent_type: Type of agent
            metadata: Optional metadata

        Returns:
            PerformanceMetrics instance
        """
        metrics = PerformanceMetrics(
            request_id=request_id,
            agent_type=agent_type,
            metadata=metadata or {},
        )
        self._active_metrics = metrics
        self._metrics[request_id] = metrics
        return metrics

    def end_profile(self, request_id: str) -> Optional[PerformanceMetrics]:
        """End profiling an execution.

        Args:
            request_id: Request identifier

        Returns:
            Completed metrics or None
        """
        metrics = self._metrics.get(request_id)
        if metrics:
            metrics.complete()
            self._active_metrics = None
        return metrics

    def record_llm_call(
        self,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: Optional[str] = None,
    ) -> None:
        """Record an LLM call.

        Args:
            latency_ms: Call latency
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            model: Model name
        """
        if self._active_metrics:
            self._active_metrics.add_llm_latency(latency_ms)
            self._active_metrics.add_tokens(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model or self.default_model,
            )
            self._llm_call_times.append(latency_ms)

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a tool call.

        Args:
            tool_name: Name of tool called
            latency_ms: Call latency
            success: Whether call succeeded
        """
        if self._active_metrics:
            self._active_metrics.add_tool_latency(latency_ms)
            if not success:
                self._active_metrics.error_count += 1

            if tool_name not in self._tool_call_times:
                self._tool_call_times[tool_name] = []
            self._tool_call_times[tool_name].append(latency_ms)

    def record_iteration(self) -> None:
        """Record an agent iteration."""
        if self._active_metrics:
            self._active_metrics.iterations += 1

    def get_metrics(self, request_id: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a request.

        Args:
            request_id: Request identifier

        Returns:
            Metrics or None
        """
        return self._metrics.get(request_id)

    def get_all_metrics(self, limit: int = 100) -> List[PerformanceMetrics]:
        """Get all recorded metrics.

        Args:
            limit: Maximum number to return

        Returns:
            List of metrics
        """
        metrics = sorted(
            self._metrics.values(),
            key=lambda x: x.start_time,
            reverse=True,
        )
        return metrics[:limit]

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all executions.

        Returns:
            Dictionary of aggregated stats
        """
        if not self._metrics:
            return {}

        durations = [m.total_duration_ms for m in self._metrics.values() if m.total_duration_ms > 0]
        costs = [m.estimated_cost for m in self._metrics.values()]
        errors = sum(m.error_count for m in self._metrics.values())
        total_tokens = sum(m.token_usage["total_tokens"] for m in self._metrics.values())

        return {
            "total_executions": len(self._metrics),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "total_cost_usd": sum(costs),
            "total_errors": errors,
            "error_rate": errors / len(self._metrics) if self._metrics else 0,
            "total_tokens": total_tokens,
            "avg_llm_latency_ms": (
                sum(self._llm_call_times) / len(self._llm_call_times)
                if self._llm_call_times
                else 0
            ),
            "tool_call_stats": {
                tool: {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }
                for tool, times in self._tool_call_times.items()
            },
        }

    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self._metrics.clear()
        self._active_metrics = None
        self._llm_call_times.clear()
        self._tool_call_times.clear()
