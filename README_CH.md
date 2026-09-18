# AgentLens

**智能Agent诊断与优化平台**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

AgentLens是一个全面的AI Agent系统调试和监控平台。它提供执行追踪、状态可视化、性能分析和智能诊断，帮助开发者更快地构建更好的Agent系统。

## 🎯 核心功能

- **Agent引擎核心**: 内置ReAct和Planner agent，支持可扩展架构
- **执行追踪**: 逐步可视化Agent的推理和行动过程
- **性能分析**: 追踪延迟、Token使用量和成本
- **智能诊断**: 自动检测常见Agent问题
- **REST API**: 轻松与现有系统集成
- **MCP协议**: 标准化的Model Context Protocol服务器
- **LangChain适配器**: 与LangChain无缝集成

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/best-taste/agentlens.git
cd agentlens

# 安装依赖
pip install -r requirements.txt

# 或以包形式安装
pip install -e .
```

## 🚀 快速开始

```python
import asyncio
from agentlens import ReActAgent, Tool

# 定义工具
def search(query: str) -> str:
    return f"搜索结果: {query}"

# 创建Agent
agent = ReActAgent(
    llm=your_llm,
    tools=[Tool(name="search", description="网络搜索", func=search)],
)

# 执行任务
response = await agent.execute("搜索AI新闻")

print(response.output)
```

## 📚 示例

### ReAct Agent

```python
from agentlens import ReActAgent, Tool

agent = ReActAgent(
    llm=my_llm,
    tools=[search_tool, calculate_tool],
    max_iterations=10,
)

result = await agent.execute("2+2等于多少?")
```

### Planner Agent

```python
from agentlens import PlannerAgent

agent = PlannerAgent(
    llm=my_llm,
    tools=[tool1, tool2],
    max_plan_depth=8,
)

result = await agent.execute("比较Python和JavaScript的优缺点")
```

### 使用API服务器

```bash
# 启动API服务器
python -m agentlens.src.api.main

# 执行Agent
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "react", "task": "你好世界"}'
```

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        AgentLens                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Dashboard   │  │     API      │  │  MCP Server  │     │
│  │  (前端)       │  │  (FastAPI)   │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                    诊断引擎                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐          │
│  │  追踪器   │  │  性能分析 │  │    智能诊断    │          │
│  └──────────┘  └──────────┘  └────────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                      Agent引擎                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐            │
│  │  ReAct  │  │ Planner  │  │   Memory    │            │
│  └──────────┘  └──────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 📄 许可证

MIT许可证 - 详见 [LICENSE](LICENSE)。

## 🙏 致谢

- 基于ReAct、Plan-and-Execute等Agent模式的最佳实践
- 借鉴了LangChain、CrewAI和swarms框架的设计理念
