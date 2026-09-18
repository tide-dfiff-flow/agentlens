"""Unit tests for agent module."""

import pytest
from agentlens.src.agent.base import BaseAgent, Tool, ToolCall, AgentState, AgentStatus


class MockLLM:
    """Mock LLM for testing."""

    async def ainvoke(self, messages):
        class Response:
            content = "This is a test response"
        return Response()


def mock_tool_func(query: str) -> str:
    """Mock tool function."""
    return f"Result for: {query}"


class TestTool:
    """Tests for Tool class."""

    def test_tool_creation(self):
        """Test creating a tool."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            func=mock_tool_func,
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert callable(tool.func)

    def test_tool_to_dict(self):
        """Test tool serialization."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            func=mock_tool_func,
        )
        data = tool.to_dict()
        assert data["name"] == "test_tool"
        assert data["description"] == "A test tool"


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(tool_name="test", arguments={"arg": "value"})
        assert tc.tool_name == "test"
        assert tc.arguments == {"arg": "value"}
        assert tc.success

    def test_tool_call_complete(self):
        """Test completing a tool call."""
        tc = ToolCall(tool_name="test", arguments={})
        tc.complete("result")
        assert tc.result == "result"
        assert tc.success
        assert tc.end_time is not None

    def test_tool_call_error(self):
        """Test tool call with error."""
        tc = ToolCall(tool_name="test", arguments={})
        tc.complete(None, "Error occurred")
        assert tc.error == "Error occurred"
        assert not tc.success

    def test_duration_ms(self):
        """Test duration calculation."""
        import time
        tc = ToolCall(tool_name="test", arguments={})
        time.sleep(0.01)
        tc.complete("result")
        assert tc.duration_ms >= 10


class MinimalAgent(BaseAgent):
    """Minimal agent for testing."""

    async def _think(self, prompt, iteration):
        return f"Thought {iteration}"

    async def _act(self, thought):
        return ToolCall(tool_name="test", arguments={})


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_agent_creation(self):
        """Test creating an agent."""
        agent = MinimalAgent(
            llm=MockLLM(),
            tools=[],
            max_iterations=5,
        )
        assert agent.max_iterations == 5
        assert len(agent.tools) == 0

    def test_add_tool(self):
        """Test adding a tool."""
        agent = MinimalAgent(llm=MockLLM(), tools=[])
        tool = Tool(name="test", description="Test", func=mock_tool_func)
        agent.add_tool(tool)
        assert "test" in agent.tools

    def test_remove_tool(self):
        """Test removing a tool."""
        agent = MinimalAgent(llm=MockLLM(), tools=[])
        tool = Tool(name="test", description="Test", func=mock_tool_func)
        agent.add_tool(tool)
        assert agent.remove_tool("test")
        assert "test" not in agent.tools

    def test_get_tool(self):
        """Test getting a tool."""
        agent = MinimalAgent(llm=MockLLM(), tools=[])
        tool = Tool(name="test", description="Test", func=mock_tool_func)
        agent.add_tool(tool)
        found = agent.get_tool("test")
        assert found is not None
        assert found.name == "test"

    def test_get_available_tools(self):
        """Test getting available tools."""
        agent = MinimalAgent(llm=MockLLM(), tools=[])
        tool = Tool(name="test", description="Test", func=mock_tool_func)
        agent.add_tool(tool)
        tools = agent.get_available_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
