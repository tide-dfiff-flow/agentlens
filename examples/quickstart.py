"""Quick start example for AgentLens.

This example demonstrates how to use AgentLens to create and run
a simple ReAct agent with tools.
"""

import asyncio
from agentlens.src.agent.react import ReActAgent
from agentlens.src.agent.base import Tool


# Define some simple tools
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Found information about AI, ML, and Deep Learning."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def get_date() -> str:
    """Get the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


# Simple mock LLM for demonstration
class MockLLM:
    """Mock LLM that provides predetermined responses."""

    async def ainvoke(self, messages):
        class Response:
            content = """Thought: I need to search for information about AI agents.

Action: search({"query": "AI agents"})
Observation: Results for 'AI agents': AI agents are software programs that can autonomously perform tasks.

Final Answer: AI agents are autonomous software programs that can perform tasks without human intervention. They use reasoning and acting patterns to accomplish goals."""
        return Response()


async def main():
    """Run the quick start example."""
    print("=" * 60)
    print("AgentLens Quick Start Example")
    print("=" * 60)

    # Create tools
    tools = [
        Tool(
            name="search",
            description="Search for information on the web",
            func=search,
        ),
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            func=calculate,
        ),
        Tool(
            name="get_date",
            description="Get the current date",
            func=get_date,
        ),
    ]

    # Create agent
    agent = ReActAgent(
        llm=MockLLM(),
        tools=tools,
        max_iterations=5,
    )

    print("\n1. Agent created with tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")

    # Run agent
    print("\n2. Executing agent task...")
    response = await agent.execute("Tell me about AI agents")

    # Print results
    print("\n3. Execution Results:")
    print(f"   Status: {response.status.value}")
    print(f"   Duration: {response.duration_ms:.2f}ms")
    print(f"   Tool calls: {len(response.tool_calls)}")

    if response.output:
        print(f"\n4. Final Output:")
        print(f"   {response.output}")

    # Show trace
    if agent.trace:
        print(f"\n5. Execution Trace ({len(agent.trace)} steps):")
        for i, step in enumerate(agent.trace[:5], 1):
            print(f"   Step {i}: {step.get('state', 'unknown')}")
        if len(agent.trace) > 5:
            print(f"   ... and {len(agent.trace) - 5} more steps")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
