# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import contextvars
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

# Context variable to dynamically track the active environment instance during a tool call
_active_env = contextvars.ContextVar('_active_env')

KNOWLEDGE_BASE = {
    "password reset procedure": "POLICY: Password Reset\n1. Verify identity.\n2. Send link.",
    "duplicate charge refund policy": "POLICY: Refund Policy\n1. Refunds within 30 days.",
    "suspended account error": "POLICY: Account Mgmt\n1. 500 error + 'suspended' -> SECURITY.",
    "refund policy account deletion conflict": "POLICY: Conflict\n1. Refund + Deletion -> BILLING.",
    "error code x77 payment failure": "POLICY: X77\n1. If 'active', search KB for 'Shadow Ban Protocol'.",
    "shadow ban protocol security": "POLICY: Shadow Ban\n1. ACTION: Escalate to SECURITY.",
    "email change request": "DEPRECATED POLICY (v1.2): For email changes, see 'Project Aegis' Internal Memo for latest security requirements.",
    "project aegis memo": "POLICY: Aegis Security\n1. If account has 'Real UID' mapping, you MUST verify the 'Compliance Auditor'.\n2. IF ticket date is within 48 hours of last transaction: Escalation is MANDATORY.\n3. Search KB for 'Aegis Audit' for final routing.",
    "aegis audit routing": "POLICY: Aegis Audit\n1. If Compliance Auditor status is NOT 'green', escalate to SECURITY.\n2. If Compliance Auditor is 'green' but account is 'flagged', escalate to BILLING."
}

class SupportEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    # Class-level reference to recover state if HTTP session routing drops the episode_id
    _latest_instance = None

    def __init__(self):
        # Convert state back to instance variables for pure concurrency
        self._task_index: int = 0
        self._current_task: dict = {}
        self._episode_done: bool = False
        self._trajectory_reward: float = 0.0
        self._tools_used: list[str] = []
        self._db: dict = {}
        self._progress: dict = {
            "read": False, 
            "searched_kb": False, 
            "checked_db": False, 
            "distracted": False,
            "out_of_order": False,
            "milestones": set() 
        }
        self._state = State(episode_id=str(uuid4()), step_count=0)

        mcp = FastMCP("support_env")

        @mcp.tool
        def read_ticket(thought: str) -> str:
            """Read the ticket."""
            env = _active_env.get()
            env._validate_thought(thought)
            env._record_tool_use("read_ticket")
            env._progress["read"] = True
            return env._current_task.get("ticket_text", "Error: Ticket missing.")

        @mcp.tool
        def search_knowledge_base(thought: str, query: str) -> str:
            """Search policies."""
            env = _active_env.get()
            env._validate_thought(thought)
            env._record_tool_use("search_knowledge_base")
            if not env._progress["read"]: env._progress["out_of_order"] = True

            q = query.lower()
            results = [v for k, v in KNOWLEDGE_BASE.items() if any(w in q for w in k.split())]
            
            if results and env._progress["read"]:
                env._progress["searched_kb"] = True
                if "aegis" in q: env._progress["milestones"].add("found_aegis_memo")
                if "audit" in q: env._progress["milestones"].add("found_aegis_audit")
                if "shadow" in q: env._progress["milestones"].add("found_shadow_policy")
                    
            return "\n\n".join(results) if results else "No specific policies found."

        @mcp.tool
        def check_billing(thought: str, user_id: str) -> str:
            """Check account status."""
            env = _active_env.get()
            env._validate_thought(thought)
            env._record_tool_use("check_billing")
            if not env._progress["searched_kb"]: env._progress["out_of_order"] = True

            user = env._db.get(user_id)
            if not user: return f"User {user_id} not found."
            
            if env._progress["read"]:
                env._progress["checked_db"] = True
                if user_id == "USR099": env._progress["milestones"].add("checked_real_uid")
                if user_id == "USR088": env._progress["milestones"].add("checked_compliance_auditor")

            res = f"Account: {user['name']}\nStatus: {user['status']}\nLast Txn: {user.get('last_txn', 'N/A')}"
            if "real_uid" in user: res += f"\nNote: Real UID is {user['real_uid']}"
            if "compliance_auditor" in user: res += f"\nAssigned Auditor: {user['compliance_auditor']}"
            return res
            
        @mcp.tool
        def ping_human_manager(thought: str, reason: str) -> str:
            """Ask a manager for help."""
            env = _active_env.get()
            env._validate_thought(thought)
            env._record_tool_use("ping_human_manager")
            env._progress["distracted"] = True
            return "SYSTEM AUTO-REPLY: All managers are currently offline. Please follow standard SOP."

        @mcp.tool
        def escalate_ticket(thought: str, department: str) -> str:
            """Escalate to billing, engineering, or security."""
            env = _active_env.get()
            env._validate_thought(thought)
            env._finalize_episode("escalate", department=department)
            return f"Escalated to {department.upper()}."

        @mcp.tool
        def resolve_ticket(thought: str, message: str) -> str:
            """Resolve ticket."""
            env = _active_env.get()
            env._validate_thought(thought)
            env._finalize_episode("resolve")
            return "Resolved."

        super().__init__(mcp)

    def _validate_thought(self, thought: str):
        if not thought or len(thought.strip()) < 10: self._progress["distracted"] = True 

    def _generate_dynamic_database(self) -> dict:
        num_dummy_users = random.randint(50, 150)
        db = {}
        benchmark_date = datetime(2026, 4, 1)
        
        first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
        
        for i in range(10, 10 + num_dummy_users):
            random_days_ago = random.randint(3, 365)
            txn_date = (benchmark_date - timedelta(days=random_days_ago)).strftime("%Y-%m-%d")
            random_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            db[f"USR{i:03d}"] = {
                "name": random_name,
                "status": random.choice(["active", "active", "active", "suspended"]),
                "last_txn": txn_date
            }
            
        db["USR001"] = {"name": "Alice", "status": "active"} 
        db["USR002"] = {"name": "Bob", "status": "active"}   
        db["USR003"] = {"name": "Carol", "status": "active"} 
        db["USR004"] = {"name": "Dan", "status": "suspended"} 
        db["USR005"] = {"name": "Eve", "status": "active"}   
        db["USR006"] = {"name": "Frank", "status": "active", "real_uid": "USR099", "last_txn": "2026-03-31"} 
        db["USR099"] = {"name": "Frank (REAL)", "status": "active", "compliance_auditor": "USR088"} 
        db["USR088"] = {"name": "System Auditor", "status": "red-flag"} 
        return db

    def _record_tool_use(self, tool_name: str) -> None:
        if self._episode_done: return
        self._tools_used.append(tool_name)

    def _get_partial_reward(self) -> float:
        if not self._current_task: return 0.0
        r = 0.0
        if self._progress["read"]: r += 0.1
        if self._progress["searched_kb"]: r += 0.1
        
        # Safely get difficulty to avoid KeyError
        diff = self._current_task.get("difficulty")
        if diff != "easy" and self._progress["checked_db"]: r += 0.1
        
        if self._progress["distracted"]: r -= 0.2
        if self._progress["out_of_order"]: r -= 0.1
        
        for tool in set(self._tools_used):
            count = self._tools_used.count(tool)
            if count > 2:
                r -= (count - 2) * 0.05
        
        m = self._progress["milestones"]
        if diff == "ultra":
            if "found_aegis_memo" not in m: r -= 0.1
            if "checked_real_uid" not in m: r -= 0.1
            if "found_aegis_audit" not in m: r -= 0.1
            if "checked_compliance_auditor" not in m: r -= 0.1
        elif diff == "multi_hop":
            if "found_shadow_policy" not in m: r -= 0.1
            
        req = self._current_task.get("required_steps", 4)
        if self._state.step_count > req:
            r -= ((self._state.step_count - req) * 0.05)
            
        return r

    def _finalize_episode(self, action_type: str, department: str | None = None) -> None:
        if self._episode_done: return
        self._episode_done = True
        r = self._get_partial_reward()
        
        # Safe dict lookups using .get()
        task_diff = self._current_task.get("difficulty", "medium") if self._current_task else "medium"
        bonus = 0.6 if task_diff == "easy" else 0.4
        
        correct_action = self._current_task.get("correct_action") if self._current_task else None
        correct_dept = self._current_task.get("correct_dept", "") if self._current_task else ""
        
        if action_type == correct_action:
            if action_type == "escalate" and department and department.lower().strip() == correct_dept:
                r += bonus
            elif action_type == "resolve":
                r += bonus
                
        self._trajectory_reward = max(0.0, min(1.0, r))

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        self._db = self._generate_dynamic_database()
        
        forced_idx = kwargs.get("task_idx")
        if forced_idx is not None: self._task_index = int(forced_idx)

        tasks = [
            {"difficulty": "easy", "required_steps": 3, "ticket_text": "Reset password USR002.", "correct_action": "resolve", "correct_dept": ""},
            {"difficulty": "medium", "required_steps": 4, "ticket_text": "Refund duplicate for USR003.", "correct_action": "resolve", "correct_dept": ""},
            {"difficulty": "hard", "required_steps": 4, "ticket_text": "Error 500 for USR004.", "correct_action": "escalate", "correct_dept": "security"},
            {"difficulty": "trap", "required_steps": 4, "ticket_text": "Delete and refund USR001.", "correct_action": "escalate", "correct_dept": "billing"},
            {"difficulty": "multi_hop", "required_steps": 5, "ticket_text": "Error X77 for USR005.", "correct_action": "escalate", "correct_dept": "security"},
            {"difficulty": "ultra", "required_steps": 8, "ticket_text": "Change email for USR006. URGENT.", "correct_action": "escalate", "correct_dept": "security"}
        ]
        
        self._current_task = tasks[self._task_index % len(tasks)]
        self._task_index += 1
        self._episode_done = False
        self._trajectory_reward = 0.0
        self._tools_used = []
        self._progress = {"read": False, "searched_kb": False, "checked_db": False, "distracted": False, "out_of_order": False, "milestones": set()}
        
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        
        # Save this instance as the master session context
        self.__class__._latest_instance = self
        return Observation(done=False, reward=0.0, metadata={"difficulty": self._current_task["difficulty"]})

    def _step_impl(self, action: Action, **kwargs) -> Observation: 
        return Observation(done=False, reward=0.0)

    def step(self, action: Action, **kwargs) -> Observation:
        active_instance = self
        
        # If this instance is blank (due to stateless HTTP), delegate to the master session!
        if not self._current_task and getattr(self.__class__, "_latest_instance", None):
            active_instance = self.__class__._latest_instance
            
        token = _active_env.set(active_instance)
        try:
            if isinstance(action, ListToolsAction): 
                return super().step(action, **kwargs)
                
            if isinstance(action, CallToolAction):
                active_instance._state.step_count += 1
                
            obs = super().step(action, **kwargs)
            
            if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
                if not active_instance._episode_done: 
                    active_instance._trajectory_reward = max(0.0, min(1.0, active_instance._get_partial_reward()))
                obs.done = active_instance._episode_done
                obs.reward = active_instance._trajectory_reward
                return obs
                
            return obs
        finally:
            _active_env.reset(token)

    @property
    def state(self) -> State: return self._state