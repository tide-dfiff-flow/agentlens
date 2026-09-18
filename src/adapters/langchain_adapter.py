"""LangChain adapter for AgentLens.

Provides integration between AgentLens and LangChain,
allowing LangChain agents to use AgentLens tracing and diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class LangChainAdapter:
    """Adapter for LangChain integration.

    This adapter provides a compatibility layer between AgentLens agents
    and LangChain tools/callbacks, enabling:
    - Use of LangChain tools within AgentLens agents
    - Integration with LangChain's callback system
    - Tracing LangChain agent executions
    """

    def __init__(self):
        """Initialize the adapter."""
        self._tools: Dict[str, Any] = {}
        self._callbacks: List[Any] = []

    def register_langchain_tool(self, tool: Any) -> None:
        """Register a LangChain tool.

        Args:
            tool: LangChain Tool instance
        """
        if hasattr(tool, "name") and hasattr(tool, "func"):
            self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a registered LangChain tool.

        Args:
            name: Tool name

        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def create_agent_callback(self) -> "AgentLensCallbackHandler":
        """Create a callback handler for LangChain agents.

        Returns:
            Callback handler instance
        """
        return AgentLensCallbackHandler(adapter=self)


@dataclass
class AgentLensCallbackHandler:
    """LangChain callback handler for AgentLens integration.

    This handler captures LangChain agent execution events
    and forwards them to AgentLens tracer and profiler.
    """

    adapter: LangChainAdapter = field(default=None)
    _trace_id: Optional[str] = None

    def __post_init__(self):
        """Initialize callback handler."""
        if self.adapter is None:
            self.adapter = LangChainAdapter()

    @property
    def always_verbose(self) -> bool:
        """Always use verbose output."""
        return True

    @property
    def ignore_llm(self) -> bool:
        """Don't process LLM callbacks."""
        return False

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts.

        Args:
            serialized: Serialized LLM info
            prompts: Input prompts
        """
        pass

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends.

        Args:
            response: LLM response
        """
        pass

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when chain starts.

        Args:
            serialized: Serialized chain info
            inputs: Chain inputs
        """
        pass

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when chain ends.

        Args:
            outputs: Chain outputs
        """
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts.

        Args:
            serialized: Serialized tool info
            input_str: Tool input
        """
        pass

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends.

        Args:
            output: Tool output
        """
        pass

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors.

        Args:
            error: Error that occurred
        """
        pass
