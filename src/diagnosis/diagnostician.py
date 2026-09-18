"""Intelligent diagnostician for AgentLens.

Analyzes execution traces to identify common issues and
provide actionable recommendations for agent optimization.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agentlens.src.diagnosis.tracer import Trace, TraceStep, TraceEventType


class IssueSeverity(Enum):
    """Severity levels for diagnosed issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of issues that can be diagnosed."""

    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    BEHAVIOR = "behavior"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"


@dataclass
class DiagnosisResult:
    """Result of diagnosing an issue.

    Attributes:
        id: Unique identifier
        category: Issue category
        severity: Issue severity
        title: Short issue title
        description: Detailed description
        evidence: Supporting evidence (trace excerpts, metrics)
        recommendations: List of recommendations
        trace_id: Associated trace ID
        timestamp: When diagnosis was made
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: IssueCategory = IssueCategory.PERFORMANCE
    severity: IssueSeverity = IssueSeverity.INFO
    title: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }


class Diagnostician:
    """Analyzes traces to identify issues and provide recommendations.

    The diagnostician can detect common agent issues including:
    - Infinite loops
    - Tool calling failures
    - Performance bottlenecks
    - Reasoning inconsistencies
    - Memory issues
    """

    # Thresholds for issue detection
    MAX_ITERATIONS_REASONABLE = 20
    MAX_TOOL_CALLS_REASONABLE = 50
    MAX_LATENCY_MS = 30000  # 30 seconds
    MIN_TOOL_SUCCESS_RATE = 0.8

    def __init__(self):
        """Initialize the diagnostician."""
        self._diagnostic_rules: List[DiagnosticRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default diagnostic rules."""
        self._diagnostic_rules = [
            DiagnosticRule(
                name="high_iteration_count",
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.WARNING,
                check=self._check_high_iterations,
            ),
            DiagnosticRule(
                name="tool_failure_loop",
                category=IssueCategory.RELIABILITY,
                severity=IssueSeverity.ERROR,
                check=self._check_tool_failure_loop,
            ),
            DiagnosticRule(
                name="long_execution_time",
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.WARNING,
                check=self._check_long_execution,
            ),
            DiagnosticRule(
                name="repeated_thoughts",
                category=IssueCategory.BEHAVIOR,
                severity=IssueSeverity.INFO,
                check=self._check_repeated_thoughts,
            ),
            DiagnosticRule(
                name="missing_tool_definitions",
                category=IssueCategory.CONFIGURATION,
                severity=IssueSeverity.ERROR,
                check=self._check_missing_tools,
            ),
            DiagnosticRule(
                name="no_tool_usage",
                category=IssueCategory.BEHAVIOR,
                severity=IssueSeverity.INFO,
                check=self._check_no_tool_usage,
            ),
            DiagnosticRule(
                name="memory_starvation",
                category=IssueCategory.RESOURCE,
                severity=IssueSeverity.WARNING,
                check=self._check_memory_issues,
            ),
        ]

    def diagnose(self, trace: Trace) -> List[DiagnosisResult]:
        """Diagnose issues in a trace.

        Args:
            trace: Trace to analyze

        Returns:
            List of diagnosis results
        """
        results = []

        for rule in self._diagnostic_rules:
            result = rule.check(trace)
            if result:
                result.trace_id = trace.id
                results.append(result)

        return results

    def diagnose_batch(self, traces: List[Trace]) -> Dict[str, List[DiagnosisResult]]:
        """Diagnose issues across multiple traces.

        Args:
            traces: List of traces to analyze

        Returns:
            Dictionary mapping trace_id to diagnosis results
        """
        results = {}
        for trace in traces:
            results[trace.id] = self.diagnose(trace)
        return results

    def _check_high_iterations(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check for unusually high iteration count."""
        if len(trace.steps) > self.MAX_ITERATIONS_REASONABLE:
            return DiagnosisResult(
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.WARNING,
                title="High Iteration Count",
                description=f"Agent executed {len(trace.steps)} steps, which exceeds the recommended maximum of {self.MAX_ITERATIONS_REASONABLE}. This may indicate an infinite loop or inefficient reasoning.",
                evidence={
                    "step_count": len(trace.steps),
                    "threshold": self.MAX_ITERATIONS_REASONABLE,
                },
                recommendations=[
                    "Review agent prompts for clarity",
                    "Consider adding a task decomposition step",
                    "Implement early termination conditions",
                    "Check for circular reasoning patterns",
                ],
            )
        return None

    def _check_tool_failure_loop(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check for repeated tool call failures."""
        tool_failures: Dict[str, int] = {}
        consecutive_failures = 0
        max_consecutive = 0

        for step in trace.steps:
            if step.event_type == TraceEventType.TOOL_END:
                if step.error:
                    tool_name = step.tool_name or "unknown"
                    tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1
                    consecutive_failures += 1
                    max_consecutive = max(max_consecutive, consecutive_failures)
                else:
                    consecutive_failures = 0

        if max_consecutive >= 3 or any(count >= 5 for count in tool_failures.values()):
            return DiagnosisResult(
                category=IssueCategory.RELIABILITY,
                severity=IssueSeverity.ERROR,
                title="Tool Failure Loop Detected",
                description="Agent is repeatedly calling failing tools, which may indicate a loop trying the same failing approach.",
                evidence={
                    "tool_failures": tool_failures,
                    "max_consecutive": max_consecutive,
                },
                recommendations=[
                    "Review tool error messages for root cause",
                    "Implement tool selection logic with failure awareness",
                    "Add alternative tool options for the same task",
                    "Consider adding a max retry count per tool",
                ],
            )
        return None

    def _check_long_execution(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check for unusually long execution time."""
        if trace.total_duration_ms > self.MAX_LATENCY_MS:
            return DiagnosisResult(
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.WARNING,
                title="Long Execution Time",
                description=f"Execution took {trace.total_duration_ms/1000:.1f} seconds, exceeding the recommended threshold of {self.MAX_LATENCY_MS/1000} seconds.",
                evidence={
                    "duration_ms": trace.total_duration_ms,
                    "threshold_ms": self.MAX_LATENCY_MS,
                },
                recommendations=[
                    "Profile individual steps to find bottleneck",
                    "Consider reducing max iterations",
                    "Optimize tool implementations",
                    "Check for network latency issues",
                ],
            )
        return None

    def _check_repeated_thoughts(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check for repeated reasoning patterns."""
        thoughts = [
            step.thought[:100].lower()
            for step in trace.steps
            if step.thought and step.event_type == TraceEventType.THOUGHT
        ]

        seen: Dict[str, int] = {}
        for thought in thoughts:
            seen[thought] = seen.get(thought, 0) + 1

        repeats = {k: v for k, v in seen.items() if v >= 3}

        if repeats:
            return DiagnosisResult(
                category=IssueCategory.BEHAVIOR,
                severity=IssueSeverity.INFO,
                title="Repeated Reasoning Patterns",
                description="Agent is repeating similar thoughts multiple times, which may indicate a reasoning loop.",
                evidence={
                    "repeat_count": len(repeats),
                    "sample_repeats": list(repeats.items())[:5],
                },
                recommendations=[
                    "Add reflection step to break loops",
                    "Implement thought diversity in prompts",
                    "Consider using a different reasoning strategy",
                ],
            )
        return None

    def _check_missing_tools(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check for attempts to use undefined tools."""
        action_steps = [
            step
            for step in trace.steps
            if step.event_type == TraceEventType.ACTION and step.tool_name
        ]

        missing_tools: Set[str] = set()
        for step in action_steps:
            if step.tool_name and step.tool_name.startswith("__"):
                continue  # Skip internal actions
            if "not found" in (step.observation or "").lower():
                missing_tools.add(step.tool_name)

        if missing_tools:
            return DiagnosisResult(
                category=IssueCategory.CONFIGURATION,
                severity=IssueSeverity.ERROR,
                title="Missing Tool Definitions",
                description="Agent attempted to use tools that are not defined in its toolkit.",
                evidence={
                    "missing_tools": list(missing_tools),
                },
                recommendations=[
                    "Add missing tool definitions to agent",
                    "Review tool naming consistency",
                    "Implement tool availability check before action",
                ],
            )
        return None

    def _check_no_tool_usage(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check if agent never uses tools."""
        tool_calls = [
            step for step in trace.steps if step.event_type == TraceEventType.TOOL_END
        ]

        if len(tool_calls) == 0 and len(trace.steps) > 3:
            return DiagnosisResult(
                category=IssueCategory.BEHAVIOR,
                severity=IssueSeverity.INFO,
                title="No Tool Usage",
                description="Agent completed execution without using any tools, which may indicate it didn't need external capabilities or is avoiding tool use inappropriately.",
                evidence={
                    "total_steps": len(trace.steps),
                    "tool_calls": 0,
                },
                recommendations=[
                    "Verify task requires tool use",
                    "Review tool availability in agent config",
                    "Check if agent is choosing manual answers over tools",
                ],
            )
        return None

    def _check_memory_issues(self, trace: Trace) -> Optional[DiagnosisResult]:
        """Check for memory-related issues."""
        memory_ops = [
            step
            for step in trace.steps
            if step.event_type in (TraceEventType.MEMORY_READ, TraceEventType.MEMORY_WRITE)
        ]

        if len(memory_ops) == 0 and trace.metadata.get("expected_memory_ops", 0) > 0:
            return DiagnosisResult(
                category=IssueCategory.RESOURCE,
                severity=IssueSeverity.WARNING,
                title="Low Memory Utilization",
                description="Agent has access to memory but isn't using it, which may indicate memory is not properly configured or agent doesn't recognize when to use it.",
                evidence={
                    "expected_ops": trace.metadata.get("expected_memory_ops", 0),
                    "actual_ops": 0,
                },
                recommendations=[
                    "Verify memory system is properly initialized",
                    "Add memory retrieval prompts to agent",
                    "Review memory consolidation settings",
                ],
            )
        return None


class DiagnosticRule:
    """A diagnostic rule for the diagnostician."""

    def __init__(
        self,
        name: str,
        category: IssueCategory,
        severity: IssueSeverity,
        check: callable,
    ):
        """Initialize a diagnostic rule.

        Args:
            name: Rule name
            category: Issue category
            severity: Severity level
            check: Check function that takes a Trace and returns DiagnosisResult or None
        """
        self.name = name
        self.category = category
        self.severity = severity
        self.check = check


from typing import Set
