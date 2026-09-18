"""MCP (Model Context Protocol) server implementation.

The MCP is a standard protocol for AI agent tool communication.
This module provides an MCP-compatible server for AgentLens.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class MCPMessage:
    """Represents an MCP protocol message.

    Attributes:
        jsonrpc: JSON-RPC version (always "2.0")
        method: Method name
        params: Method parameters
        id: Message ID for responses
    """

    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            MCPMessage instance
        """
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            params=data.get("params"),
            id=data.get("id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class MCPTool:
    """Represents a tool in the MCP protocol.

    Attributes:
        name: Tool name
        description: Tool description
        input_schema: JSON schema for tool input
    """

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPServer:
    """MCP protocol server implementation.

    Provides a standards-compliant MCP server that can:
    - List available tools
    - Execute tool calls
    - Handle MCP protocol messages

    Example:
        ```python
        server = MCPServer()

        @server.tool(name="search", description="Search the web")
        def search(query: str) -> str:
            return f"Results for: {query}"

        await server.start()
        ```
    """

    def __init__(self, name: str = "agentlens", version: str = "1.0.0"):
        """Initialize MCP server.

        Args:
            name: Server name
            version: Server version
        """
        self.name = name
        self.version = version
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, MCPTool] = {}
        self._request_handlers: Dict[str, Callable] = {}
        self._notification_handlers: Dict[str, Callable] = {}
        self._running = False

        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default MCP request handlers."""
        self._request_handlers["initialize"] = self._handle_initialize
        self._request_handlers["tools/list"] = self._handle_tools_list
        self._request_handlers["tools/call"] = self._handle_tools_call

        self._notification_handlers["initialized"] = self._handle_initialized
        self._notification_handlers["exit"] = self._handle_exit

    def tool(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator to register a tool.

        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for input

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.register_tool(
                name=name,
                func=func,
                description=description,
                input_schema=input_schema or {},
            )
            return func
        return decorator

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the server.

        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            input_schema: JSON schema for input
        """
        self._tools[name] = func
        self._tool_schemas[name] = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema or {},
        )

    async def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle an incoming MCP message.

        Args:
            message: Incoming message

        Returns:
            Response message or None for notifications
        """
        # Check if it's a request or notification
        if message.id is not None:
            # Request - needs response
            handler = self._request_handlers.get(message.method)
            if handler:
                result = await handler(message.params or {})
                return MCPMessage(
                    method=message.method,
                    id=message.id,
                    params={"result": result},
                )
            else:
                return MCPMessage(
                    method=message.method,
                    id=message.id,
                    params={"error": {"code": -32601, "message": "Method not found"}},
                )
        else:
            # Notification - no response needed
            handler = self._notification_handlers.get(message.method)
            if handler:
                await handler(message.params or {})
            return None

    async def handle_message_batch(self, messages: List[MCPMessage]) -> List[Optional[MCPMessage]]:
        """Handle multiple MCP messages.

        Args:
            messages: List of messages

        Returns:
            List of responses (None for notifications)
        """
        return [await self.handle_message(msg) for msg in messages]

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request.

        Args:
            params: Initialize parameters

        Returns:
            Server capabilities
        """
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }

    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request.

        Args:
            params: Request parameters

        Returns:
            List of available tools
        """
        return {
            "tools": [t.to_dict() for t in self._tool_schemas.values()],
        }

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request.

        Args:
            params: Call parameters including name and arguments

        Returns:
            Tool call result
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Tool '{tool_name}' not found",
                    }
                ],
                "isError": True,
            }

        try:
            tool = self._tools[tool_name]
            result = tool(**arguments) if arguments else tool()

            # Handle async results
            if hasattr(result, "__await__"):
                result = await result

            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result),
                    }
                ],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}",
                    }
                ],
                "isError": True,
            }

    async def _handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle initialized notification.

        Args:
            params: Notification parameters
        """
        pass

    async def _handle_exit(self, params: Dict[str, Any]) -> None:
        """Handle exit notification.

        Args:
            params: Notification parameters
        """
        self._running = False

    async def start(self) -> None:
        """Start the MCP server."""
        self._running = True

    async def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
