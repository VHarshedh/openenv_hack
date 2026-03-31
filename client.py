# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Triage Environment Client.

This module provides the client for connecting to a Support Environment server.
SupportEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with SupportEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("read_ticket")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class SupportEnv(MCPToolClient):
    """
    Client for the Customer Support Triage Environment.

    This client provides a simple interface for interacting with the
    Support Environment via MCP tools. It inherits all functionality
    from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Example:
        >>> with SupportEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...
        ...     # List available tools
        ...     tools = env.list_tools()
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
        ...
        ...     # Read the ticket
        ...     ticket = env.call_tool("read_ticket")
        ...     print(ticket)
        ...
        ...     # Search knowledge base
        ...     policy = env.call_tool("search_knowledge_base", query="password reset")
        ...     print(policy)
    """

    pass  # MCPToolClient provides all needed functionality
