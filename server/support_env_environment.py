# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Triage Environment Implementation.

A pure MCP environment that simulates a customer support triage workflow.
Agents must use tools to read tickets, search knowledge bases, check billing,
and decide whether to resolve or escalate — across 4 difficulty levels.

All interactions happen through MCP tools:
- `read_ticket()`: Returns the customer's issue text.
- `search_knowledge_base(query)`: Returns policy text based on keywords.
- `check_billing(user_id)`: Returns recent transactions for a user.
- `escalate_ticket(department)`: Ends the episode via escalation.
- `resolve_ticket(message)`: Ends the episode with a resolution message.
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

USER_DATABASE = {
    "USR001": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "plan": "Premium",
        "join_date": "2024-01-15",
        "status": "active",
    },
    "USR002": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "plan": "Free",
        "join_date": "2025-06-20",
        "status": "active",
    },
    "USR003": {
        "name": "Carol Davis",
        "email": "carol@example.com",
        "plan": "Business",
        "join_date": "2023-11-01",
        "status": "active",
    },
    "USR004": {
        "name": "Dan Lee",
        "email": "dan@example.com",
        "plan": "Premium",
        "join_date": "2025-02-10",
        "status": "suspended",
    },
}

KNOWLEDGE_BASE = {
    "password_reset": (
        "POLICY: Password Reset Procedure\n"
        "1. Verify the user's identity via their registered email.\n"
        "2. Send a password reset link to the user's email.\n"
        "3. The link expires in 24 hours.\n"
        "4. Inform the user to check spam/junk folders.\n"
        "Resolution: Send the reset link and confirm with the user."
    ),
    "refund": (
        "POLICY: Refund Policy\n"
        "1. Refunds are available within 30 days of purchase.\n"
        "2. After 30 days, refunds require manager approval.\n"
        "3. Subscription downgrades can be offered as an alternative.\n"
        "4. Document the reason for the refund request.\n"
        "Resolution: If within 30 days, process immediately. Otherwise, escalate."
    ),
    "billing": (
        "POLICY: Billing Disputes\n"
        "1. Check the user's recent transactions for discrepancies.\n"
        "2. Verify the charge matches the user's subscription plan.\n"
        "3. If a duplicate charge is found, issue an immediate refund.\n"
        "4. For disputed valid charges, explain the billing breakdown."
    ),
    "outage": (
        "POLICY: Service Outage Protocol\n"
        "1. Check the system status page for known incidents.\n"
        "2. If a known outage, inform the user and provide an ETA.\n"
        "3. If not a known outage, escalate to Engineering immediately.\n"
        "4. Never promise a specific resolution time for unknown issues."
    ),
    "account": (
        "POLICY: Account Management\n"
        "1. Users can update their email and password from settings.\n"
        "2. Plan changes take effect at the next billing cycle.\n"
        "3. Account deletion requires a 14-day cooling-off period. CONFLICT RULE: If a user requests account deletion AND a refund simultaneously, do not process either request. Escalate to BILLING immediately.\n"
        "4. Suspended accounts lose UI access and will experience 500 errors. If you check billing and see a 'suspended' status, you must escalate to SECURITY immediately."
    ),
}

BILLING_RECORDS = {
    "USR001": [
        {"date": "2026-03-01", "amount": 29.99, "description": "Premium Plan - Monthly"},
        {"date": "2026-02-01", "amount": 29.99, "description": "Premium Plan - Monthly"},
        {"date": "2026-01-01", "amount": 29.99, "description": "Premium Plan - Monthly"},
    ],
    "USR002": [
        {"date": "2026-03-15", "amount": 0.00, "description": "Free Plan"},
        {"date": "2025-12-20", "amount": 9.99, "description": "One-time Add-on Purchase"},
    ],
    "USR003": [
        {"date": "2026-03-01", "amount": 99.99, "description": "Business Plan - Monthly"},
        {"date": "2026-03-01", "amount": 99.99, "description": "Business Plan - Monthly (DUPLICATE)"},
        {"date": "2026-02-01", "amount": 99.99, "description": "Business Plan - Monthly"},
        {"date": "2025-11-15", "amount": 49.99, "description": "Enterprise Add-on"},
    ],
    "USR004": [
        {"date": "2026-02-10", "amount": 29.99, "description": "Premium Plan - Monthly"},
        {"date": "2026-02-10", "amount": 29.99, "description": "Premium Plan - Monthly (DUPLICATE)"},
    ],
}


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = [
    # Task 1 — Easy: Password reset
    {
        "difficulty": "easy",
        "description": "Password reset",
        "ticket_text": (
            "Hi, I forgot my password and can't log into my account. "
            "My username is alice@example.com. Can you help me reset it? "
            "I've already tried the 'Forgot Password' link but didn't receive an email."
        ),
        "user_id": "USR001",
        "category": "password_reset",
        "correct_action": "resolve",
        "required_tools": ["read_ticket", "search_knowledge_base"],
        "correct_resolution_keywords": ["reset", "link", "email", "spam"],
        "correct_escalation_dept": None,
    },
    # Task 2 — Medium: Duplicate Charge Identification
    {
        "difficulty": "medium",
        "description": "Duplicate Charge Identification",
        "ticket_text": (
            "Hello, I was checking my credit card statement and noticed I was "
            "charged $99.99 twice on March 1st for my Business Plan. "
            "Can you please refund the extra charge? My user ID is USR003."
        ),
        "user_id": "USR003",
        "category": "billing",
        "correct_action": "resolve",
        "required_tools": ["read_ticket", "search_knowledge_base", "check_billing"],
        "correct_resolution_keywords": ["refund", "duplicate", "processed", "issued", "extra"],
        "correct_escalation_dept": None,
    },
    # Task 3 — Hard: Multi-Step Suspension Diagnosis
    {
        "difficulty": "hard",
        "description": "Multi-Step Suspension Diagnosis",
        "ticket_text": (
            "I'm getting a 500 error every time I try to access my dashboard. "
            "I've been a Premium member for months. My user ID is USR004. "
            "Please fix this server issue immediately!"
        ),
        "user_id": "USR004",
        "category": "account",
        "correct_action": "escalate",
        "required_tools": ["read_ticket", "check_billing", "search_knowledge_base"],
        "correct_resolution_keywords": [],
        "correct_escalation_dept": "security", # Must find that the user is 'suspended'
    },
    # Task 4 — Trap: Cross-Policy Conflict
    {
        "difficulty": "trap",
        "description": "Cross-Policy Conflict",
        "ticket_text": (
            "I want to delete my account (USR001) immediately and get a refund "
            "for my payment from 3 days ago. I'm moving to a competitor today."
        ),
        "user_id": "USR001",
        "category": "refund",
        "correct_action": "escalate",
        "required_tools": ["read_ticket", "check_billing", "search_knowledge_base"],
        "correct_resolution_keywords": [],
        "correct_escalation_dept": "billing", # Requires identifying the 14-day cooling-off conflict
    },
]


class SupportEnvironment(MCPEnvironment):
    """
    Customer Support Triage environment.

    Agents interact via MCP tools to diagnose and resolve customer tickets
    across 4 difficulty levels. Partial trajectory rewards are accumulated
    based on correct tool usage patterns.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    # ── CLASS-LEVEL shared state for stateless HTTP (REST) support ──
    _task_index: int = 0
    _current_task: dict = TASKS[0]
    _episode_done: bool = False
    _trajectory_reward: float = 0.0
    _tools_used: list[str] = []
    _queries_searched: list[str] = []
    _shared_state: Optional[State] = None

    def __init__(self):
        """Initialize the support triage environment with MCP tools."""
        cls = self.__class__

        # ── build the FastMCP server with 5 tools ──
        mcp = FastMCP("support_env")

        @mcp.tool
        def read_ticket() -> str:
            """Read the current customer support ticket."""
            cls._record_tool_use("read_ticket")
            return cls._current_task["ticket_text"]

        @mcp.tool
        def search_knowledge_base(query: str) -> str:
            """Search the internal knowledge base for relevant policies."""
            cls._record_tool_use("search_knowledge_base")
            cls._queries_searched.append(query.lower())
            query_lower = query.lower()
            results = []
            for key, text in KNOWLEDGE_BASE.items():
                words = [w for w in key.split("_") if w]
                if not words:
                    continue
                if len(words) == 1:
                    if words[0] in query_lower:
                        results.append(text)
                elif all(w in query_lower for w in words):
                    results.append(text)
            if results:
                return "\n\n---\n\n".join(results)
            return (
                "No matching policies found for your query. "
                "Try keywords: password, refund, billing, outage, account."
            )

        @mcp.tool
        def check_billing(user_id: str) -> str:
            """Check a user's billing history and recent transactions."""
            cls._record_tool_use("check_billing")
            user = USER_DATABASE.get(user_id)
            if not user:
                return f"Error: User '{user_id}' not found in database."

            records = BILLING_RECORDS.get(user_id, [])
            lines = [
                f"=== Account: {user['name']} ===",
                f"Email: {user['email']}",
                f"Plan: {user['plan']}",
                f"Status: {user['status']}",
                f"Member since: {user['join_date']}",
                "",
                "Recent Transactions:",
            ]
            for r in records:
                lines.append(f"  {r['date']}  ${r['amount']:.2f}  {r['description']}")
            return "\n".join(lines)

        @mcp.tool
        def escalate_ticket(department: str) -> str:
            """Escalate the ticket to a specialized department."""
            cls._record_tool_use("escalate_ticket")
            cls._finalize_episode("escalate", department=department)
            ref = str(uuid4())[:8].upper()
            return (
                f"Ticket escalated to {department.upper()} department. "
                f"Reference: ESC-{ref}. "
                f"The specialized team will follow up within 24 hours."
            )

        @mcp.tool
        def resolve_ticket(message: str) -> str:
            """Resolve the ticket with a response message to the customer."""
            cls._record_tool_use("resolve_ticket")
            cls._finalize_episode("resolve", message=message)
            ref = str(uuid4())[:8].upper()
            return (
                f"Ticket resolved. Reference: RES-{ref}. "
                f"Resolution sent to customer: {message}"
            )

        # Pass the MCP server to the base class
        super().__init__(mcp)
        if cls._shared_state is None:
            cls._shared_state = State(episode_id=str(uuid4()), step_count=0)
        self._state = cls._shared_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _record_tool_use(cls, tool_name: str) -> None:
        """Track tool usage and award partial rewards."""
        if cls._episode_done:
            return  # no rewards after episode ends

        cls._tools_used.append(tool_name)
        task = cls._current_task
        is_first_use = cls._tools_used.count(tool_name) == 1

        # +0.1 for the FIRST use of a tool in the required set (no double-counting)
        if tool_name in task["required_tools"] and is_first_use:
            cls._trajectory_reward += 0.1

        # Extra +0.1 if read_ticket is the very first tool used
        if tool_name == "read_ticket" and len(cls._tools_used) == 1:
            cls._trajectory_reward += 0.1

    @classmethod
    def _finalize_episode(
        cls,
        action_type: str,
        department: str | None = None,
        message: str | None = None,
    ) -> None:
        """Compute final reward and mark episode as done."""
        if cls._episode_done:
            return

        cls._episode_done = True
        task = cls._current_task

        # ── Penalty: finishing without checking required tools ──
        required = set(task["required_tools"])
        used = set(cls._tools_used)
        missing = required - used - {"resolve_ticket", "escalate_ticket"}
        if missing:
            cls._trajectory_reward -= 0.2 * len(missing)

        # ── Final score based on correct action ──
        correct_action = task["correct_action"]

        if action_type == correct_action:
            if action_type == "escalate":
                correct_dept = task.get("correct_escalation_dept", "")
                if department and correct_dept and department.lower() == correct_dept.lower():
                    if task.get("difficulty") == "hard":
                        if "check_billing" in cls._tools_used and "search_knowledge_base" in cls._tools_used:
                            cls._trajectory_reward += 0.5
                        else:
                            cls._trajectory_reward += 0.3
                    else:
                        cls._trajectory_reward += 0.5  # correct department
                else:
                    cls._trajectory_reward += 0.3  # right action, wrong dept
            elif action_type == "resolve":
                # Check if resolution message contains expected keywords
                keywords = task.get("correct_resolution_keywords", [])
                if message and keywords:
                    hits = sum(1 for kw in keywords if kw.lower() in message.lower())
                    ratio = hits / len(keywords) if keywords else 0
                    cls._trajectory_reward += 0.3 + (0.2 * ratio)
                else:
                    cls._trajectory_reward += 0.3
        else:
            # Wrong action type entirely
            cls._trajectory_reward -= 0.3

        # Conflict Awareness (Task 4)
        if task.get("difficulty") == "trap":
            searched_refund = any("refund" in q for q in cls._queries_searched)
            searched_account = any("account" in q for q in cls._queries_searched)
            if searched_refund ^ searched_account:
                cls._trajectory_reward -= 0.2

        if cls._shared_state and cls._shared_state.step_count > 3:
            cls._trajectory_reward -= 0.05 * (cls._shared_state.step_count - 3)

        # Clamp final reward to [0.0, 1.0]
        cls._trajectory_reward = max(0.0, min(1.0, cls._trajectory_reward))

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment to the next task.

        Cycles through 4 difficulty levels (easy → medium → hard → trap).

        Returns:
            Observation with the task briefing in metadata.
        """
        cls = self.__class__
        cls._current_task = TASKS[cls._task_index % len(TASKS)]
        cls._task_index += 1
        cls._episode_done = False
        cls._trajectory_reward = 0.0
        cls._tools_used = []
        cls._queries_searched = []

        cls._shared_state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._state = cls._shared_state

        task = cls._current_task
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task_number": cls._task_index,
                "difficulty": task["difficulty"],
                "message": (
                    f"New support ticket received (Difficulty: {task['difficulty'].upper()}). "
                    f"Use the available tools to diagnose and resolve the issue. "
                    f"Start by calling read_ticket() to see the customer's message."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (not supported)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction for MCP interactions."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step — delegates to MCPEnvironment, then injects rewards.

        The base class handles ListToolsAction and CallToolAction routing
        to the FastMCP server. We preserve the original observation type
        so that the HTTP serializer can correctly serialize MCP-specific
        fields (tools, tool_name, result).
        """
        cls = self.__class__
        cls._shared_state.step_count += 1
        self._state = cls._shared_state

        # For ListToolsAction, just pass through without modifying
        if isinstance(action, ListToolsAction):
            return super().step(action, timeout_s=timeout_s, **kwargs)

        # Let the base class handle the action
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # For CallToolAction, preserve the CallToolObservation type
        if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
            if cls._episode_done:
                obs.done = True
                obs.reward = cls._trajectory_reward
                obs.metadata = {
                    **(obs.metadata or {}),
                    "trajectory_reward": cls._trajectory_reward,
                    "tools_used": cls._tools_used.copy(),
                    "difficulty": cls._current_task["difficulty"],
                    "task_number": cls._task_index,
                }
            else:
                obs.reward = cls._trajectory_reward
                obs.metadata = {
                    **(obs.metadata or {}),
                    "partial_reward": cls._trajectory_reward,
                    "tools_used_so_far": cls._tools_used.copy(),
                }
            return obs

        # Fallback for non-MCP actions
        if cls._episode_done:
            return Observation(
                done=True,
                reward=cls._trajectory_reward,
                metadata={
                    **(obs.metadata if hasattr(obs, "metadata") and obs.metadata else {}),
                    "trajectory_reward": cls._trajectory_reward,
                    "tools_used": cls._tools_used.copy(),
                    "difficulty": cls._current_task["difficulty"],
                    "task_number": cls._task_index,
                },
            )

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
