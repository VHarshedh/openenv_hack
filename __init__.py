# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Triage Environment — MCP-based ticket resolution.

This environment exposes all functionality through MCP tools:
- `read_ticket()`: Read the customer's issue text.
- `search_knowledge_base(query)`: Find relevant policies.
- `check_billing(user_id)`: Check a user's billing history.
- `escalate_ticket(department)`: Escalate to a specialist team.
- `resolve_ticket(message)`: Resolve the ticket with a message.

Example:
    >>> from support_env import SupportEnv
    >>>
    >>> with SupportEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     ticket = env.call_tool("read_ticket")
    ...     print(ticket)
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import SupportEnv

__all__ = ["SupportEnv", "CallToolAction", "ListToolsAction"]
