# AgentLens

**Intelligent Agent Diagnosis & Optimization Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

AgentLens is a comprehensive debugging and monitoring platform for AI Agent systems. It provides execution tracing, state visualization, performance profiling, and intelligent diagnosis to help developers build better agents faster.

## 🎯 Features

- **Agent Engine Core**: Built-in ReAct and Planner agents with extensible architecture
- **Execution Tracing**: Step-by-step visibility into agent reasoning and actions
- **Performance Profiling**: Track latency, token usage, and costs
- **Intelligent Diagnosis**: Automatic detection of common agent issues
- **REST API**: Easy integration with existing systems
- **MCP Protocol**: Standards-compliant Model Context Protocol server
- **LangChain Adapter**: Seamlessly integrate with LangChain

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/best-taste/agentlens.git
cd agentlens

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## 🚀 Quick Start

```python
import asyncio
from agentlens import ReActAgent, Tool

# Define a tool
def search(query: str) -> str:
    return f"Results for: {query}"

# Create agent
agent = ReActAgent(
    llm=your_llm,
    tools=[Tool(name="search", description="Search the web", func=search)],
)

# Execute
response = await agent.execute("Search for AI news")

print(response.output)
```

## 📚 Examples

### ReAct Agent

```python
from agentlens import ReActAgent, Tool

agent = ReActAgent(
    llm=my_llm,
    tools=[search_tool, calculate_tool],
    max_iterations=10,
)

result = await agent.execute("What is 2 + 2?")
```

### Planner Agent

```python
from agentlens import PlannerAgent

agent = PlannerAgent(
    llm=my_llm,
    tools=[tool1, tool2],
    max_plan_depth=8,
)

result = await agent.execute("Compare Python vs JavaScript")
```

### Using the API Server

```bash
# Start the API server
python -m agentlens.src.api.main

# Execute an agent
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "react", "task": "Hello world"}'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AgentLens                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Dashboard   │  │     API      │  │  MCP Server  │     │
│  │  (Frontend)   │  │  (FastAPI)   │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                    Diagnosis Engine                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐            │
│  │  Tracer  │  │ Profiler │  │ Diagnostician  │            │
│  └──────────┘  └──────────┘  └────────────────┘            │
├─────────────────────────────────────────────────────────────┤
│                      Agent Engine                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐              │
│  │  ReAct  │  │ Planner  │  │   Memory     │              │
│  └──────────┘  └──────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Optional: Set API host/port
export AGENTLENS_HOST="0.0.0.0"
export AGENTLENS_PORT="8000"
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on insights from ReAct, Plan-and-Execute, and other agent patterns
- Inspired by LangChain, CrewAI, and swarms frameworks
