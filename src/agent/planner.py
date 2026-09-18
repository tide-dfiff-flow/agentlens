"""Planner Agent implementation for AgentLens.

The Planner Agent uses task decomposition to break down complex tasks
into smaller, manageable subtasks that can be executed sequentially
or in parallel.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agentlens.src.agent.base import (
    AgentState,
    AgentStatus,
    BaseAgent,
    Tool,
    ToolCall,
)


class SubTask:
    """Represents a subtask in a plan.

    Attributes:
        id: Unique identifier for the subtask
        description: Human-readable task description
        status: Current status (pending, in_progress, completed, failed)
        result: Result from executing this subtask
        dependencies: List of subtask IDs that must complete first
    """

    def __init__(
        self,
        id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
    ):
        self.id = id
        self.description = description
        self.status = "pending"
        self.result: Optional[Any] = None
        self.dependencies = dependencies or []
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "result": str(self.result)[:500] if self.result else None,
            "dependencies": self.dependencies,
            "error": self.error,
        }


class Plan:
    """Represents a task plan with multiple subtasks.

    Attributes:
        task: The original task
        subtasks: List of subtasks in execution order
        current_index: Index of currently executing subtask
    """

    def __init__(self, task: str, subtasks: Optional[List[SubTask]] = None):
        self.task = task
        self.subtasks = subtasks or []
        self.current_index = 0
        self._completed_ids: set = set()

    def add_subtask(self, subtask: SubTask) -> None:
        """Add a subtask to the plan."""
        self.subtasks.append(subtask)

    def get_next_subtask(self) -> Optional[SubTask]:
        """Get the next subtask that can be executed.

        Returns:
            Next executable subtask or None if all complete
        """
        for subtask in self.subtasks:
            if subtask.status == "pending":
                # Check dependencies
                deps_met = all(dep in self._completed_ids for dep in subtask.dependencies)
                if deps_met:
                    return subtask
        return None

    def mark_completed(self, subtask_id: str, result: Any) -> None:
        """Mark a subtask as completed.

        Args:
            subtask_id: ID of completed subtask
            result: Execution result
        """
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "completed"
                subtask.result = result
                self._completed_ids.add(subtask_id)
                break

    def mark_failed(self, subtask_id: str, error: str) -> None:
        """Mark a subtask as failed.

        Args:
            subtask_id: ID of failed subtask
            error: Error message
        """
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "failed"
                subtask.error = error
                break

    def is_complete(self) -> bool:
        """Check if all subtasks are complete."""
        return all(t.status in ("completed", "failed") for t in self.subtasks)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task": self.task,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "current_index": self.current_index,
            "is_complete": self.is_complete(),
        }


class PlannerAgent(BaseAgent):
    """Planner Agent that decomposes tasks into subtasks.

    The Planner Agent follows a Plan-and-Execute pattern:
    1. Analyze task and create a plan (decompose into subtasks)
    2. Execute subtasks one by one or in parallel when possible
    3. Aggregate results into final answer

    This is particularly useful for complex tasks that can be broken down
    into independent components.

    Example:
        ```python
        from agentlens import PlannerAgent, Tool

        # Define execution tools
        tools = [
            Tool(name="search", description="Search for info", func=search_fn),
            Tool(name="calculate", description="Perform calculation", func=calc_fn),
        ]

        agent = PlannerAgent(
            llm=my_llm,
            tools=tools,
            max_plan_depth=5
        )

        response = await agent.execute("Compare AI trends in 2020 vs 2024")
        ```

    Reference: Plan-and-Execute pattern from various agent frameworks.
    """

    TASK_PATTERN = re.compile(
        r"(?:\d+[\.\)]\s*)?(.+?)(?=\n(?:\d+[\.\)])|$)",
        re.DOTALL,
    )

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 20,
        max_plan_depth: int = 10,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        parallel_execution: bool = True,
        tracing_enabled: bool = True,
    ):
        """Initialize Planner Agent.

        Args:
            llm: Language model instance
            tools: List of available tools
            max_iterations: Maximum total iterations (plan + execution)
            max_plan_depth: Maximum subtasks in a plan
            max_tokens: Max tokens per LLM response
            temperature: LLM sampling temperature
            parallel_execution: Enable parallel subtask execution
            tracing_enabled: Enable execution tracing
        """
        super().__init__(
            llm=llm,
            tools=tools,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            temperature=temperature,
            tracing_enabled=tracing_enabled,
        )
        self.max_plan_depth = max_plan_depth
        self.parallel_execution = parallel_execution
        self._current_plan: Optional[Plan] = None
        self._plan_prompt = self._default_plan_prompt()

    def _default_plan_prompt(self) -> str:
        """Generate default planning prompt.

        Returns:
            Default prompt for task decomposition
        """
        return """You are a task planner. Your job is to break down complex tasks into smaller subtasks.

Guidelines:
- Decompose the task into 3-8 clear, actionable subtasks
- Each subtask should be independent and specific
- Order subtasks logically (some may depend on others)
- Number subtasks clearly (1., 2., 3., etc.)

Output format:
Return a JSON array of subtasks, each with:
- "id": A unique identifier (e.g., "step_1")
- "description": Clear description of what to do
- "dependencies": List of step IDs that must complete first (can be empty)

Example for "Compare Python vs JavaScript":
[
  {"id": "step_1", "description": "Research Python characteristics", "dependencies": []},
  {"id": "step_2", "description": "Research JavaScript characteristics", "dependencies": []},
  {"id": "step_3", "description": "Compare performance metrics", "dependencies": ["step_1", "step_2"]},
  {"id": "step_4", "description": "Summarize findings", "dependencies": ["step_3"]}
]

Now decompose this task:
"""

    async def _think(self, prompt: str, iteration: int) -> str:
        """Generate plan using LLM.

        Args:
            prompt: Current prompt
            iteration: Current iteration

        Returns:
            Generated plan as JSON string
        """
        # Check if we need to create a plan
        if self._current_plan is None or self._current_plan.is_complete():
            messages = [
                {"role": "system", "content": self._plan_prompt},
                {"role": "user", "content": f"Task: {prompt}"},
            ]

            try:
                if hasattr(self.llm, "ainvoke"):
                    response = await self.llm.ainvoke(messages)
                    plan_text = response.content if hasattr(response, "content") else str(response)
                elif hasattr(self.llm, "invoke"):
                    response = self.llm.invoke(messages)
                    plan_text = response.content if hasattr(response, "content") else str(response)
                else:
                    plan_text = await self._simple_llm_call(prompt)

                # Parse plan
                self._current_plan = self._parse_plan(plan_text, prompt)
                return json.dumps(self._current_plan.to_dict())

            except Exception as e:
                return f"Error creating plan: {str(e)}"

        # Return current plan status
        return json.dumps(self._current_plan.to_dict())

    async def _simple_llm_call(self, prompt: str) -> str:
        """Simple LLM fallback.

        Args:
            prompt: Prompt text

        Returns:
            LLM response
        """
        if hasattr(self.llm, "__call__"):
            return self.llm(prompt)
        raise ValueError("LLM must have ainvoke or invoke method")

    def _parse_plan(self, plan_text: str, task: str) -> Plan:
        """Parse LLM output into a Plan.

        Args:
            plan_text: LLM response text
            task: Original task

        Returns:
            Parsed Plan object
        """
        plan = Plan(task=task)

        # Try JSON parsing
        try:
            # Extract JSON array from response
            json_match = re.search(r"\[.*\]", plan_text, re.DOTALL)
            if json_match:
                subtasks_data = json.loads(json_match.group())
                for data in subtasks_data[:self.max_plan_depth]:
                    subtask = SubTask(
                        id=data["id"],
                        description=data["description"],
                        dependencies=data.get("dependencies", []),
                    )
                    plan.add_subtask(subtask)
                return plan
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback to text parsing
        lines = plan_text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # Match numbered items
            match = re.match(r"(?:\d+[\.\)]\s*)?(.+)", line)
            if match:
                subtask = SubTask(
                    id=f"step_{i+1}",
                    description=match.group(1).strip(),
                )
                plan.add_subtask(subtask)

        return plan

    async def _act(self, thought: str) -> ToolCall:
        """Execute next step in plan.

        Args:
            thought: Plan description

        Returns:
            ToolCall for next subtask
        """
        if self._current_plan is None:
            return ToolCall(tool_name="__finish__", arguments={"reasoning": "No plan available"})

        # Get next executable subtask
        subtask = self._current_plan.get_next_subtask()

        if subtask is None:
            # All tasks complete
            return ToolCall(
                tool_name="__finish__",
                arguments={"result": "All subtasks completed"},
            )

        return ToolCall(
            tool_name="__execute_subtask__",
            arguments={
                "subtask_id": subtask.id,
                "description": subtask.description,
            },
        )

    def _should_finish(self, thought: str) -> bool:
        """Check if execution should finish.

        Args:
            thought: Current thought

        Returns:
            True if should finish
        """
        if self._current_plan is None:
            return True
        return self._current_plan.is_complete()

    def _extract_output(self, thought: str) -> str:
        """Extract final output from plan results.

        Args:
            thought: Final thought

        Returns:
            Aggregated output
        """
        if self._current_plan is None:
            return "No plan was executed"

        results = []
        for subtask in self._current_plan.subtasks:
            if subtask.status == "completed":
                results.append(f"## {subtask.id}: {subtask.description}\n{subtask.result}")
            elif subtask.status == "failed":
                results.append(f"## {subtask.id}: {subtask.description}\nFailed: {subtask.error}")

        return "\n\n".join(results) if results else "No results available"
