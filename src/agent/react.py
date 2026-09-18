"""ReAct Agent implementation for AgentLens.

ReAct (Reasoning + Acting) is a prominent agent architecture that combines
reasoning traces with action execution in an interleaved manner.

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., 2022)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from agentlens.src.agent.base import (
    AgentState,
    AgentStatus,
    BaseAgent,
    Tool,
    ToolCall,
)


class ReActAgent(BaseAgent):
    """ReAct Agent implementation.

    The ReAct agent follows the reasoning loop:
    1. Thought: Analyze current state and decide what to do
    2. Action: Execute a tool or provide final answer
    3. Observation: Observe the result of the action
    4. Repeat until completion

    This implementation provides:
    - Interleaved reasoning and acting
    - Tool selection based on LLM reasoning
    - Structured output parsing
    - Comprehensive execution tracing

    Example:
        ```python
        from agentlens import ReActAgent, Tool

        # Define tools
        def search(query: str) -> str:
            return f"Results for: {query}"

        tools = [
            Tool(name="search", description="Search the web", func=search)
        ]

        # Create agent
        agent = ReActAgent(
            llm=my_llm,
            tools=tools,
            max_iterations=10
        )

        # Execute
        response = await agent.execute("Search for recent AI news")
        print(response.output)
        ```
    """

    # Pattern for parsing LLM output
    THOUGHT_PATTERN = re.compile(
        r"Thought[:\s]*(.+?)(?=\n(?:Action|Observation|Final|$))",
        re.IGNORECASE | re.DOTALL,
    )
    ACTION_PATTERN = re.compile(
        r"Action[:\s]*(\w+)(?:\((.*?)\))?",
        re.IGNORECASE | re.DOTALL,
    )
    FINAL_ANSWER_PATTERN = re.compile(
        r"(?:Final\s*Answer|Answer)[:\s]*(.+)",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        tracing_enabled: bool = True,
    ):
        """Initialize ReAct agent.

        Args:
            llm: Language model instance (must have ainvoke method for async)
            tools: List of available tools
            max_iterations: Maximum think-act cycles
            max_tokens: Max tokens per LLM response
            temperature: LLM sampling temperature
            system_prompt: Custom system prompt
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
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Generate default system prompt for ReAct.

        Returns:
            Default system prompt
        """
        tools_desc = []
        for name, tool in self.tools.items():
            params = json.dumps(tool.parameters) if tool.parameters else "{}"
            tools_desc.append(f"- {name}: {tool.description}")

        tools_text = "\n".join(tools_desc) if tools_desc else "No tools available."

        return f"""You are a ReAct agent that combines reasoning with acting.

Follow this loop:
1. Thought: Analyze the current situation and reason about what to do
2. Action: Call a tool (if needed) using the format: Action: tool_name({{"arg": "value"}})
3. Observation: Wait for the result
4. Repeat until you can provide a Final Answer

Available tools:
{tools_text}

Guidelines:
- Always start with a Thought before taking an Action
- If you need information, use a tool rather than guessing
- If a tool fails, try an alternative approach
- Once you have the answer, provide it as: Final Answer: [your answer]
- Be concise but thorough in your reasoning
"""

    async def _think(self, prompt: str, iteration: int) -> str:
        """Generate reasoning thought using LLM.

        Args:
            prompt: Current conversation prompt
            iteration: Current iteration number

        Returns:
            LLM generated thought
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            # Call LLM - handle both sync and async implementations
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(messages)
                thought = response.content if hasattr(response, "content") else str(response)
            elif hasattr(self.llm, "invoke"):
                response = self.llm.invoke(messages)
                thought = response.content if hasattr(response, "content") else str(response)
            else:
                # Fallback for simple string-based LLMs
                thought = await self._simple_llm_call(prompt)

            return thought.strip()

        except Exception as e:
            return f"Error in reasoning: {str(e)}"

    async def _simple_llm_call(self, prompt: str) -> str:
        """Simple LLM call fallback for basic implementations.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response
        """
        if hasattr(self.llm, "__call__"):
            return self.llm(prompt)
        raise ValueError("LLM must have ainvoke, invoke, or __call__ method")

    async def _act(self, thought: str) -> ToolCall:
        """Parse thought and determine action.

        Args:
            thought: LLM generated thought

        Returns:
            ToolCall to execute
        """
        # Check for final answer
        final_match = self.FINAL_ANSWER_PATTERN.search(thought)
        if final_match:
            # Return a special "final" tool call
            return ToolCall(
                tool_name="__final__",
                arguments={"answer": final_match.group(1).strip()},
            )

        # Parse action
        action_match = self.ACTION_PATTERN.search(thought)
        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)

            # Parse arguments
            arguments = {}
            if args_str:
                try:
                    arguments = json.loads(args_str)
                except json.JSONDecodeError:
                    # Try simple key=value parsing
                    arguments = self._parse_simple_args(args_str)

            return ToolCall(tool_name=tool_name, arguments=arguments)

        # No action found - return finish with thought as output
        return ToolCall(
            tool_name="__finish__",
            arguments={"reasoning": thought},
        )

    def _parse_simple_args(self, args_str: str) -> Dict[str, Any]:
        """Parse simple argument strings.

        Args:
            args_str: Argument string like 'query="test"' or 'x=1, y=2'

        Returns:
            Parsed arguments dictionary
        """
        result = {}
        # Match key="value" or key='value' or key=value patterns
        pattern = re.compile(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
        for match in pattern.finditer(args_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)
            result[key] = value
        return result

    def _should_finish(self, thought: str) -> bool:
        """Check if agent should finish.

        Args:
            thought: Current thought

        Returns:
            True if should finish
        """
        return bool(self.FINAL_ANSWER_PATTERN.search(thought)) or "__finish__" in thought.lower()

    def _extract_output(self, thought: str) -> str:
        """Extract final answer from thought.

        Args:
            thought: Final thought

        Returns:
            Extracted answer
        """
        final_match = self.FINAL_ANSWER_PATTERN.search(thought)
        if final_match:
            return final_match.group(1).strip()

        # Try to extract last action result
        return thought.split("Observation:")[-1].strip() if "Observation:" in thought else thought.strip()


class ReActResult:
    """Structured result from ReAct agent execution.

    Provides easy access to common result fields and utilities
    for further processing.
    """

    def __init__(self, response: Any):
        """Initialize from agent response.

        Args:
            response: AgentResponse from ReActAgent
        """
        self.response = response

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.response.status == AgentStatus.COMPLETED

    @property
    def output(self) -> Optional[str]:
        """Get the final output."""
        return self.response.output

    @property
    def reasoning_steps(self) -> List[str]:
        """Get all reasoning steps."""
        return self.response.reasoning

    @property
    def tool_calls(self) -> List[ToolCall]:
        """Get all tool calls made."""
        return self.response.tool_calls

    @property
    def total_duration_ms(self) -> float:
        """Get total execution time."""
        return self.response.duration_ms

    @property
    def trace(self) -> List[Dict[str, Any]]:
        """Get execution trace."""
        return self.response.metadata.get("trace", [])

    def summary(self) -> str:
        """Generate a text summary of the execution.

        Returns:
            Summary string
        """
        lines = [
            f"Status: {self.response.status.value}",
            f"Duration: {self.total_duration_ms:.2f}ms",
            f"Tool calls: {len(self.tool_calls)}",
            f"Output: {self.output or 'N/A'}",
        ]
        return "\n".join(lines)
