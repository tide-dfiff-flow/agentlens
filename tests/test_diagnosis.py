"""Unit tests for diagnosis module."""

import pytest
import time
from agentlens.src.diagnosis.tracer import ExecutionTracer, Trace, TraceStep, TraceEventType
from agentlens.src.diagnosis.profiler import PerformanceProfiler, PerformanceMetrics
from agentlens.src.diagnosis.diagnostician import Diagnostician, DiagnosisResult, IssueCategory, IssueSeverity


class TestTracer:
    """Tests for ExecutionTracer."""

    def test_tracer_creation(self):
        """Test creating a tracer."""
        tracer = ExecutionTracer()
        assert tracer.max_traces == 1000

    def test_start_and_end_trace(self):
        """Test starting and ending a trace."""
        tracer = ExecutionTracer()
        trace = tracer.start_trace("req-1", "react", "test task")
        assert trace is not None
        assert trace.agent_type == "react"
        assert trace.task == "test task"

        completed = tracer.end_trace("completed")
        assert completed.status == "completed"
        assert completed.end_time is not None

    def test_record_thought(self):
        """Test recording a thought."""
        tracer = ExecutionTracer()
        tracer.start_trace("req-1", "react", "test")

        step = tracer.record_thought("This is my thought", state="reasoning")
        assert step is not None
        assert step.thought == "This is my thought"

        tracer.end_trace()

    def test_record_action(self):
        """Test recording an action."""
        tracer = ExecutionTracer()
        tracer.start_trace("req-1", "react", "test")

        step = tracer.record_action(
            action="search",
            tool_name="search",
            tool_args={"query": "test"},
        )
        assert step is not None
        assert step.tool_name == "search"

        tracer.end_trace()

    def test_get_trace(self):
        """Test getting a trace."""
        tracer = ExecutionTracer()
        trace = tracer.start_trace("req-1", "react", "test")
        tracer.end_trace()

        found = tracer.get_trace(trace.id)
        assert found is not None
        assert found.id == trace.id

    def test_get_traces(self):
        """Test listing traces."""
        tracer = ExecutionTracer()
        for i in range(5):
            t = tracer.start_trace(f"req-{i}", "react", f"task {i}")
            tracer.end_trace()

        traces = tracer.get_traces(limit=10)
        assert len(traces) == 5


class TestProfiler:
    """Tests for PerformanceProfiler."""

    def test_profiler_creation(self):
        """Test creating a profiler."""
        profiler = PerformanceProfiler()
        assert profiler.default_model == "gpt-4"

    def test_start_and_end_profile(self):
        """Test profiling lifecycle."""
        profiler = PerformanceProfiler()
        metrics = profiler.start_profile("req-1", "react")

        assert metrics is not None
        assert metrics.agent_type == "react"

        completed = profiler.end_profile("req-1")
        assert completed.total_duration_ms > 0

    def test_record_llm_call(self):
        """Test recording LLM calls."""
        profiler = PerformanceProfiler()
        profiler.start_profile("req-1", "react")
        profiler.record_llm_call(latency_ms=100, prompt_tokens=50, completion_tokens=30)

        profiler.end_profile("req-1")
        metrics = profiler.get_metrics("req-1")

        assert metrics.llm_latency_ms >= 100
        assert metrics.token_usage["prompt_tokens"] == 50

    def test_record_tool_call(self):
        """Test recording tool calls."""
        profiler = PerformanceProfiler()
        profiler.start_profile("req-1", "react")
        profiler.record_tool_call("search", latency_ms=50, success=True)

        profiler.end_profile("req-1")
        metrics = profiler.get_metrics("req-1")

        assert metrics.tool_latency_ms >= 50


class TestDiagnostician:
    """Tests for Diagnostician."""

    def test_diagnostician_creation(self):
        """Test creating a diagnostician."""
        diag = Diagnostician()
        assert len(diag._diagnostic_rules) > 0

    def test_diagnose_empty_trace(self):
        """Test diagnosing an empty trace."""
        diag = Diagnostician()
        trace = Trace(request_id="req-1", agent_type="react", task="test")

        results = diag.diagnose(trace)
        # Empty trace may not trigger any issues
        assert isinstance(results, list)

    def test_diagnose_high_iterations(self):
        """Test detecting high iteration count."""
        diag = Diagnostician()
        trace = Trace(request_id="req-1", agent_type="react", task="test")

        # Add many steps to simulate high iterations
        for i in range(25):
            step = TraceStep(event_type=TraceEventType.THOUGHT, state="reasoning")
            trace.add_step(step)

        trace.complete("completed")

        results = diag.diagnose(trace)
        issue = next((r for r in results if r.title == "High Iteration Count"), None)
        assert issue is not None
        assert issue.severity == IssueSeverity.WARNING


class TestDiagnosisResult:
    """Tests for DiagnosisResult."""

    def test_diagnosis_result_creation(self):
        """Test creating a diagnosis result."""
        result = DiagnosisResult(
            category=IssueCategory.PERFORMANCE,
            severity=IssueSeverity.WARNING,
            title="Test Issue",
            description="Test description",
        )
        assert result.category == IssueCategory.PERFORMANCE
        assert result.severity == IssueSeverity.WARNING

    def test_diagnosis_result_to_dict(self):
        """Test serialization."""
        result = DiagnosisResult(
            title="Test",
            description="Test desc",
        )
        data = result.to_dict()
        assert data["title"] == "Test"
        assert data["category"] == "performance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
