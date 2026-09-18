"""FastAPI application for AgentLens.

Provides REST API endpoints for agent execution, tracing,
and performance monitoring.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentlens.src.agent.base import Tool
from agentlens.src.agent.react import ReActAgent
from agentlens.src.agent.planner import PlannerAgent
from agentlens.src.diagnosis.tracer import ExecutionTracer
from agentlens.src.diagnosis.profiler import PerformanceProfiler
from agentlens.src.diagnosis.diagnostician import Diagnostician
from agentlens.src.api.schemas import (
    ExecuteRequest,
    ExecuteResponse,
    TraceResponse,
    MetricsResponse,
    DiagnosisResponse,
    HealthResponse,
    ErrorResponse,
)


# Global instances
tracer = ExecutionTracer()
profiler = PerformanceProfiler()
diagnostician = Diagnostician()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    print("AgentLens API starting up...")
    yield
    # Shutdown
    print("AgentLens API shutting down...")


app = FastAPI(
    title="AgentLens API",
    description="Intelligent Agent Diagnosis & Optimization Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple mock LLM for testing
class MockLLM:
    """Mock LLM for testing without API calls."""

    async def ainvoke(self, messages: List[Dict[str, str]]) -> Any:
        """Mock LLM call."""
        content = messages[-1]["content"] if messages else ""

        # Simple keyword-based responses
        if "search" in content.lower():
            response = """Thought: I should search for information about this.

Action: search({"query": "artificial intelligence"})
Observation: Search results for 'artificial intelligence': AI is a broad field of computer science..."

Final Answer: AI stands for Artificial Intelligence, which is the simulation of human intelligence by machines."""
        elif "calculate" in content.lower():
            response = """Thought: I should perform a calculation.

Action: calculate({"expression": "2 + 2"})
Observation: Result: 4

Final Answer: The result is 4."""
        elif "final" in content.lower() or "complete" in content.lower():
            response = "Final Answer: Task completed successfully."
        else:
            response = """Thought: Let me analyze this task and provide a thoughtful response.

Final Answer: This is a test response from the mock LLM."""

        class Response:
            content = response

        return Response()


# Global mock LLM instance
mock_llm = MockLLM()


def create_agent(
    agent_type: str,
    tools: List[Tool],
    max_iterations: int,
    temperature: float,
    enable_tracing: bool,
) -> Any:
    """Create an agent instance based on type.

    Args:
        agent_type: Type of agent
        tools: Available tools
        max_iterations: Max iterations
        temperature: Sampling temperature
        enable_tracing: Enable tracing

    Returns:
        Agent instance
    """
    if agent_type == "react":
        return ReActAgent(
            llm=mock_llm,
            tools=tools,
            max_iterations=max_iterations,
            temperature=temperature,
            tracing_enabled=enable_tracing,
        )
    elif agent_type == "planner":
        return PlannerAgent(
            llm=mock_llm,
            tools=tools,
            max_iterations=max_iterations,
            temperature=temperature,
            tracing_enabled=enable_tracing,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.time(),
    )


@app.post("/execute", response_model=ExecuteResponse)
async def execute_agent(request: ExecuteRequest):
    """Execute an agent on a task.

    Args:
        request: Execution request

    Returns:
        Execution response
    """
    try:
        # Create tools from request
        tools = [
            Tool(
                name=t.name,
                description=t.description,
                func=lambda **kwargs: f"Mock result for {t.name}",
                parameters=t.parameters,
            )
            for t in request.tools
        ]

        # Create agent
        agent = create_agent(
            agent_type=request.agent_type,
            tools=tools,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
            enable_tracing=request.enable_tracing,
        )

        # Start profiling
        profiler.start_profile(
            request_id=agent._request_id or "unknown",
            agent_type=request.agent_type,
            metadata=request.metadata,
        )

        # Start tracing
        if request.enable_tracing:
            tracer.start_trace(
                request_id=agent._request_id or "unknown",
                agent_type=request.agent_type,
                task=request.task,
                metadata=request.metadata,
            )

        # Execute agent
        response = await agent.execute(request.task)

        # End profiling
        profiler.end_profile(response.request_id)

        # End tracing
        if request.enable_tracing:
            tracer.end_trace(status=response.status.value)

        # Get trace ID
        trace_id = None
        if request.enable_tracing:
            active_trace = tracer.get_active_trace()
            if active_trace:
                trace_id = active_trace.id

        return ExecuteResponse(
            request_id=response.request_id,
            status=response.status.value,
            output=response.output,
            duration_ms=response.duration_ms,
            tool_calls=[tc.to_dict() if hasattr(tc, "to_dict") else {"tool_name": tc.tool_name} for tc in response.tool_calls],
            reasoning=response.reasoning,
            tokens_used=response.tokens_used,
            trace_id=trace_id,
            error=response.error,
        )

    except Exception as e:
        return ExecuteResponse(
            request_id="error",
            status="failed",
            duration_ms=0,
            error=str(e),
        )


@app.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str):
    """Get a trace by ID.

    Args:
        trace_id: Trace identifier

    Returns:
        Trace data
    """
    trace = tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    return TraceResponse(
        id=trace.id,
        request_id=trace.request_id,
        agent_type=trace.agent_type,
        task=trace.task,
        status=trace.status,
        start_time=trace.start_time,
        end_time=trace.end_time,
        total_duration_ms=trace.total_duration_ms,
        step_count=len(trace.steps),
        steps=[s.to_dict() for s in trace.steps],
        metadata=trace.metadata,
    )


@app.get("/traces", response_model=List[TraceResponse])
async def list_traces(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    agent_type: Optional[str] = None,
    status: Optional[str] = None,
):
    """List traces with optional filtering.

    Args:
        limit: Maximum results
        offset: Pagination offset
        agent_type: Filter by agent type
        status: Filter by status

    Returns:
        List of traces
    """
    traces = tracer.get_traces(
        limit=limit,
        offset=offset,
        agent_type=agent_type,
        status=status,
    )

    return [
        TraceResponse(
            id=t.id,
            request_id=t.request_id,
            agent_type=t.agent_type,
            task=t.task,
            status=t.status,
            start_time=t.start_time,
            end_time=t.end_time,
            total_duration_ms=t.total_duration_ms,
            step_count=len(t.steps),
            steps=[s.to_dict() for s in t.steps],
            metadata=t.metadata,
        )
        for t in traces
    ]


@app.get("/metrics/{request_id}", response_model=MetricsResponse)
async def get_metrics(request_id: str):
    """Get metrics for a specific request.

    Args:
        request_id: Request identifier

    Returns:
        Performance metrics
    """
    metrics = profiler.get_metrics(request_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")

    return MetricsResponse(
        request_id=metrics.request_id,
        agent_type=metrics.agent_type,
        total_duration_ms=metrics.total_duration_ms,
        llm_latency_ms=metrics.llm_latency_ms,
        tool_latency_ms=metrics.tool_latency_ms,
        token_usage=metrics.token_usage,
        estimated_cost_usd=metrics.estimated_cost,
        error_count=metrics.error_count,
        iterations=metrics.iterations,
        steps_per_second=metrics.steps_per_second,
    )


@app.get("/metrics", response_model=List[MetricsResponse])
async def list_metrics(limit: int = Query(default=100, ge=1, le=1000)):
    """List all performance metrics.

    Args:
        limit: Maximum results

    Returns:
        List of metrics
    """
    all_metrics = profiler.get_all_metrics(limit=limit)

    return [
        MetricsResponse(
            request_id=m.request_id,
            agent_type=m.agent_type,
            total_duration_ms=m.total_duration_ms,
            llm_latency_ms=m.llm_latency_ms,
            tool_latency_ms=m.tool_latency_ms,
            token_usage=m.token_usage,
            estimated_cost_usd=m.estimated_cost,
            error_count=m.error_count,
            iterations=m.iterations,
            steps_per_second=m.steps_per_second,
        )
        for m in all_metrics
    ]


@app.get("/stats")
async def get_stats():
    """Get aggregated statistics.

    Returns:
        Aggregated stats
    """
    return profiler.get_aggregated_stats()


@app.post("/diagnose/{trace_id}", response_model=DiagnosisResponse)
async def diagnose_trace(trace_id: str):
    """Diagnose issues in a trace.

    Args:
        trace_id: Trace identifier

    Returns:
        Diagnosis results
    """
    trace = tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    issues = diagnostician.diagnose(trace)

    return DiagnosisResponse(
        trace_id=trace_id,
        issues=[i.to_dict() for i in issues],
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
