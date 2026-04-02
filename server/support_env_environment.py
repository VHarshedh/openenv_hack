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
    # Core Policies
    "password reset procedure": "POLICY: Password Reset\n1. Verify identity.\n2. Send link.",
    "duplicate charge refund policy": "POLICY: Refund Policy\n1. Refunds within 30 days.",
    "suspended account error": "POLICY: Account Mgmt\n1. 500 error + 'suspended' -> SECURITY.",
    "refund policy account deletion conflict": "POLICY: Conflict\n1. Refund + Deletion -> BILLING.",
    "error code x77 payment failure": "POLICY: X77\n1. If 'active', search KB for 'Shadow Ban Protocol'.",
    "shadow ban protocol security": "POLICY: Shadow Ban\n1. ACTION: Escalate to SECURITY.",
    "email change request": "DEPRECATED POLICY (v1.2): For email changes, see 'Project Aegis' Internal Memo for latest security requirements.",
    "project aegis memo": "POLICY: Aegis Security\n1. If account has 'Real UID' mapping, you MUST verify the 'Compliance Auditor'.\n2. IF ticket date is within 48 hours of last transaction: Escalation is MANDATORY.\n3. Search KB for 'Aegis Audit' for final routing.",
    "aegis audit routing": "POLICY: Aegis Audit\n1. If Compliance Auditor status is NOT 'green', escalate to SECURITY.\n2. If Compliance Auditor is 'green' but account is 'flagged', escalate to BILLING.",
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
        self._active_noise_policies: list[dict] = []
        
        # Dynamic Target Placeholders
        self._target_easy = ""
        self._target_medium = ""
        self._target_hard = ""
        self._target_trap = ""
        self._target_multi = ""
        self._target_ultra = ""
        self._real_uid = ""
        self._auditor_uid = ""

        self._progress: dict = {
            "read": False, 
            "searched_kb": False, 
            "checked_db": False, 
            "distracted": False,
            "out_of_order": False,
            "god_query": False,
            "teleported": False,
            "lazy_resolution": False,
            "milestones": set(),
            "queried_users": set(),
            # Tracking Instruction Following (40% of grade)
            "thought_identified_request": False,
            "thought_noted_state": False,
            "thought_verified_kb": False
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
            
            # Anti-Cheat 1: God Query (Milestone Spoofing)
            trigger_words = ["password", "refund", "delete", "account", "500", "suspended", "x77", "email", "aegis", "audit", "shadow"]
            for p in env._active_noise_policies:
                trigger_words.extend(p["keywords"])
                
            hits = sum(1 for k in trigger_words if k in q)
            if hits > 3:
                env._progress["god_query"] = True
                return "SYSTEM ERROR: Query too broad. Please refine your search to specific keywords."

            results = []
            
            # Core Routes
            if "password" in q: results.append(KNOWLEDGE_BASE["password reset procedure"])
            if "refund" in q and not ("delete" in q or "account" in q or "conflict" in q): 
                results.append(KNOWLEDGE_BASE["duplicate charge refund policy"])
            if ("refund" in q and ("delete" in q or "account" in q)) or "conflict" in q:
                results.append(KNOWLEDGE_BASE["refund policy account deletion conflict"])
            if "500" in q or "suspended" in q:
                results.append(KNOWLEDGE_BASE["suspended account error"])
            if "x77" in q:
                results.append(KNOWLEDGE_BASE["error code x77 payment failure"])
            if "shadow" in q:
                results.append(KNOWLEDGE_BASE["shadow ban protocol security"])
            if "email" in q:
                results.append(KNOWLEDGE_BASE["email change request"])
            if "aegis" in q and "audit" not in q:
                results.append(KNOWLEDGE_BASE["project aegis memo"])
            if "audit" in q:
                results.append(KNOWLEDGE_BASE["aegis audit routing"])
                
            # Noise Routes (Dynamic)
            for p in env._active_noise_policies:
                if any(k in q for k in p["keywords"]):
                    results.append(p["text"])
            
            if results and env._progress["read"]:
                env._progress["searched_kb"] = True
                # Milestones
                if "password" in q: env._progress["milestones"].add("found_password_policy")
                if "refund" in q or "duplicate" in q: env._progress["milestones"].add("found_refund_policy")
                if "account" in q or "delete" in q or "deletion" in q: env._progress["milestones"].add("found_account_policy")
                if "500" in q or "suspended" in q: env._progress["milestones"].add("found_500_policy")
                if "x77" in q: env._progress["milestones"].add("found_x77_policy")
                if "email" in q: env._progress["milestones"].add("found_email_policy")
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
            
            if not env._progress["read"]: env._progress["out_of_order"] = True

            # Anti-Cheat: Teleportation prevents mind-reading dynamically generated inner IDs
            if user_id == env._real_uid and "found_aegis_memo" not in env._progress["milestones"]:
                env._progress["teleported"] = True
            if user_id == env._auditor_uid and ("checked_real_uid" not in env._progress["milestones"] or "found_aegis_audit" not in env._progress["milestones"]):
                env._progress["teleported"] = True

            env._progress["queried_users"].add(user_id)
            user = env._db.get(user_id)
            if not user: return f"User {user_id} not found."
            
            if env._progress["read"]:
                env._progress["checked_db"] = True
                if user_id == env._real_uid: env._progress["milestones"].add("checked_real_uid")
                if user_id == env._auditor_uid: env._progress["milestones"].add("checked_compliance_auditor")

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
            
            # Anti-Cheat: Lazy Resolution
            if len(message.strip().split()) < 5:
                env._progress["lazy_resolution"] = True
                
                if any(w in thought_lower for w in ["policy", "knowledge base", "kb", "escalation", "verify", "rule", "protocol"]):
                    self._progress["thought_verified_kb"] = True

    def _generate_random_noise_policies(self, count: int) -> list[dict]:
        """Procedurally generates completely random noise policies."""
        topics = [
            (["dress", "code", "attire"], "Business attire", "is required on level", str(random.randint(1, 5))),
            (["lunch", "reimbursement", "food"], "Lunch expenses", "are capped at $", str(random.randint(15, 50))),
            (["holiday", "vacation", "schedule"], "Vacation requests", "must be submitted", f"{random.randint(7, 30)} days in advance"),
            (["travel", "flight", "hotel"], "Flight bookings", "must use airline code", f"{random.choice(['AA', 'DL', 'UA'])}{random.randint(100, 999)}"),
            (["remote", "wfh", "home"], "Remote work", "is limited to", f"{random.randint(1, 4)} days per week"),
            (["expense", "report", "receipt"], "Expense reports", "must be approved by", random.choice(["Finance", "HR", "your manager"])),
            (["benefits", "health", "insurance"], "Health enrollment", "closes on", f"November {random.randint(1, 30)}"),
            (["pet", "dog", "office"], "Office pets", "must weigh under", f"{random.randint(10, 50)} lbs"),
            (["parking", "garage", "permit"], "Parking permits", "expire on", f"December {random.randint(1, 31)}"),
            (["security", "badge", "access"], "Guest badges", "must be returned to", random.choice(["Reception", "Security", "Lobby"])),
            (["hardware", "laptop", "monitor"], "Hardware requests", "take approximately", f"{random.randint(2, 8)} weeks to process")
        ]
        
        policies = []
        sampled_topics = random.sample(topics, min(count, len(topics)))
        
        for keywords, subject, action, value in sampled_topics:
            # Construct the randomized text
            text = f"POLICY: {subject} Guidelines\n1. {subject} {action} {value}."
            
            # 50% chance to add a randomized secondary rule
            if random.random() > 0.5:
                text += f"\n2. Violations will be reported to {random.choice(['HR', 'Management', 'Facilities', 'Legal'])}."
            
            policies.append({"keywords": keywords, "text": text})
            
        return policies

    def _generate_dynamic_database(self) -> dict:
        num_dummy_users = random.randint(50, 150)
        db = {}
        benchmark_date = datetime(2026, 4, 1)
        
        first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
        
        # Generate the pool
        for i in range(1, num_dummy_users + 10):
            random_days_ago = random.randint(3, 365)
            txn_date = (benchmark_date - timedelta(days=random_days_ago)).strftime("%Y-%m-%d")
            random_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            db[f"USR{i:03d}"] = {
                "name": random_name,
                "status": random.choice(["active", "active", "active", "suspended"]),
                "last_txn": txn_date
            }
            
        # Procedurally map target tasks to random IDs from the pool
        keys = list(db.keys())
        random.shuffle(keys)
        
        self._target_easy = keys[0]
        self._target_medium = keys[1]
        self._target_hard = keys[2]
        self._target_trap = keys[3]
        self._target_multi = keys[4]
        self._target_ultra = keys[5]
        self._real_uid = keys[6]
        self._auditor_uid = keys[7]

        # Inject necessary task state conditions into those specific random IDs
        db[self._target_easy]["status"] = "active"
        db[self._target_medium]["status"] = "active"
        db[self._target_hard]["status"] = "suspended"
        db[self._target_trap]["status"] = "active"
        db[self._target_multi]["status"] = "active"
        
        db[self._target_ultra]["status"] = "active"
        db[self._target_ultra]["real_uid"] = self._real_uid
        db[self._target_ultra]["last_txn"] = "2026-03-31" # recent txn triggers Aegis
        
        db[self._real_uid]["status"] = "active"
        db[self._real_uid]["compliance_auditor"] = self._auditor_uid
        
        db[self._auditor_uid]["status"] = "red-flag"
        
        return db

    def _record_tool_use(self, tool_name: str) -> None:
        if self._episode_done: return
        self._tools_used.append(tool_name)

    def _get_partial_reward(self) -> float:
        if not self._current_task: return 0.0
        
        # 1. Instruction Following Reward (Max 0.4 / 40%)
        thought_r = 0.0
        if self._progress["thought_identified_request"]: thought_r += 0.1
        if self._progress["thought_noted_state"]: thought_r += 0.1
        if self._progress["thought_verified_kb"]: thought_r += 0.2
        
        # 2. Task Accuracy/Progression Reward (Max 0.3 here, leaving 0.3 for the final action)
        task_r = 0.0
        diff = self._current_task.get("difficulty")
        m = self._progress["milestones"]
        u = self._progress["queried_users"]
        target_user = self._current_task.get("target_user")
        
        if diff == "easy":
            if self._progress["read"]: task_r += 0.1
            if "found_password_policy" in m: task_r += 0.2
        elif diff == "medium":
            if self._progress["read"]: task_r += 0.1
            if target_user in u: task_r += 0.1
            if "found_refund_policy" in m: task_r += 0.1
        elif diff == "hard":
            if self._progress["read"]: task_r += 0.1
            if target_user in u: task_r += 0.1
            if "found_500_policy" in m: task_r += 0.1
        elif diff == "trap":
            if self._progress["read"]: task_r += 0.1
            if "found_refund_policy" in m: task_r += 0.1
            if "found_account_policy" in m: task_r += 0.1
        elif diff == "multi_hop":
            if self._progress["read"]: task_r += 0.1
            if "found_x77_policy" in m: task_r += 0.1
            if "found_shadow_policy" in m: task_r += 0.1
        elif diff == "ultra":
            if self._progress["read"]: task_r += 0.1
            if "checked_real_uid" in m: task_r += 0.1
            if "found_aegis_audit" in m: task_r += 0.1
            
        r = thought_r + task_r

        # Global Penalties
        if self._progress["distracted"]: r -= 0.2
        if self._progress["out_of_order"]: r -= 0.1
        if self._progress["god_query"]: r -= 0.2
        if self._progress["teleported"]: r -= 0.3
        if self._progress["lazy_resolution"]: r -= 0.2
        
        for tool in set(self._tools_used):
            if self._tools_used.count(tool) > 2: r -= (self._tools_used.count(tool) - 2) * 0.05
            
        return max(0.0, r)

    def _finalize_episode(self, action_type: str, department: str | None = None) -> None:
        if self._episode_done: return
        self._episode_done = True
        r = self._get_partial_reward()
        
        task_diff = self._current_task.get("difficulty", "medium") if self._current_task else "medium"
        target_user = self._current_task.get("target_user")
        m = self._progress["milestones"]
        u = self._progress["queried_users"]
        correct_action = self._current_task.get("correct_action")
        correct_dept = self._current_task.get("correct_dept", "")
        
        # Verify if final terminal action strings match
        is_action_correct = False
        if action_type == correct_action:
            if action_type == "escalate" and department and department.lower().strip() == correct_dept:
                is_action_correct = True
            elif action_type == "resolve":
                is_action_correct = True
                
        # 3. Final Task Accuracy Bonus (Max 0.3 to complete the 60% accuracy split)
        bonus = 0.0
        cheated = self._progress["god_query"] or self._progress["teleported"] or self._progress["lazy_resolution"]
        
        # Only award the bonus if they actually navigated the task properly and didn't cheat
        if is_action_correct and not cheated:
            valid_progression = False
            if task_diff == "easy" and self._progress["read"] and "found_password_policy" in m: valid_progression = True
            elif task_diff == "medium" and self._progress["read"] and target_user in u and "found_refund_policy" in m: valid_progression = True
            elif task_diff == "hard" and self._progress["read"] and target_user in u and "found_500_policy" in m: valid_progression = True
            elif task_diff == "trap" and self._progress["read"] and "found_refund_policy" in m and "found_account_policy" in m: valid_progression = True
            elif task_diff == "multi_hop" and self._progress["read"] and "found_x77_policy" in m and "found_shadow_policy" in m: valid_progression = True
            elif task_diff == "ultra" and self._progress["read"] and "checked_real_uid" in m and "found_aegis_audit" in m: valid_progression = True
            
            if valid_progression:
                bonus = 0.3

        r += bonus
        
        # Task 4 Conflict Awareness Penalty: Dock -0.2 if they resolved but only searched ONE of the two required policies.
        if task_diff == "trap":
            has_ref = "found_refund_policy" in m
            has_acc = "found_account_policy" in m
            if (has_ref != has_acc) and (has_ref or has_acc):
                r -= 0.2

        # Step Count Efficiency Penalty
        req = self._current_task.get("required_steps", 4) if self._current_task else 4
        if self._state.step_count > req:
            r -= ((self._state.step_count - req) * 0.05)

        # Strict Clamping ensures the reward mathematically cannot exit 0.0 - 1.0 bounds
        self._trajectory_reward = max(0.0, min(1.0, r))

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        self._db = self._generate_dynamic_database()
        
        # Procedurally generate 3-5 entirely random noise policies for this episode
        self._active_noise_policies = self._generate_random_noise_policies(random.randint(3, 5))
        
        forced_idx = kwargs.get("task_idx")
        if forced_idx is not None: self._task_index = int(forced_idx)

        # Generate Procedural Ticket Text variations
        easy_texts = [f"Reset password for {self._target_easy}.", f"User {self._target_easy} is locked out, needs password reset.", f"Forgot password on account {self._target_easy}."]
        medium_texts = [f"Refund duplicate charge for {self._target_medium}.", f"Double billed user {self._target_medium}, please refund.", f"Process duplicate refund for {self._target_medium}."]
        hard_texts = [f"Deep Diagnosis: Error 500 for {self._target_hard}.", f"{self._target_hard} keeps getting Error 500.", f"Fix 500 internal error for {self._target_hard}."]
        trap_texts = [f"Conflict Awareness: Delete account and refund {self._target_trap}.", f"User {self._target_trap} wants a refund and account deletion.", f"Please refund my last order and delete my account ({self._target_trap})."]
        multi_texts = [f"Error X77 for {self._target_multi}.", f"Payment failed with X77 for {self._target_multi}.", f"Getting error code X77 on checkout, user {self._target_multi}."]
        ultra_texts = [f"Change email for {self._target_ultra}. URGENT.", f"Urgent email update requested for {self._target_ultra}.", f"Update email address for account {self._target_ultra} immediately."]

        tasks = [
            {"difficulty": "easy", "required_steps": 3, "target_user": self._target_easy, "ticket_text": random.choice(easy_texts), "correct_action": "resolve", "correct_dept": ""},
            {"difficulty": "medium", "required_steps": 4, "target_user": self._target_medium, "ticket_text": random.choice(medium_texts), "correct_action": "resolve", "correct_dept": ""},
            {"difficulty": "hard", "required_steps": 4, "target_user": self._target_hard, "ticket_text": random.choice(hard_texts), "correct_action": "escalate", "correct_dept": "security"},
            {"difficulty": "trap", "required_steps": 5, "target_user": self._target_trap, "ticket_text": random.choice(trap_texts), "correct_action": "escalate", "correct_dept": "billing"},
            {"difficulty": "multi_hop", "required_steps": 5, "target_user": self._target_multi, "ticket_text": random.choice(multi_texts), "correct_action": "escalate", "correct_dept": "security"},
            {"difficulty": "ultra", "required_steps": 8, "target_user": self._target_ultra, "ticket_text": random.choice(ultra_texts), "correct_action": "escalate", "correct_dept": "security"}
        ]
        
        self._current_task = tasks[self._task_index % len(tasks)]
        self._task_index += 1
        self._episode_done = False
        self._trajectory_reward = 0.0
        self._tools_used = []
        
        self._progress = {
            "read": False, "searched_kb": False, "checked_db": False, 
            "distracted": False, "out_of_order": False, 
            "god_query": False, "teleported": False, "lazy_resolution": False, 
            "milestones": set(), "queried_users": set(),
            # Tracking Instruction Following
            "thought_identified_request": False,
            "thought_noted_state": False,
            "thought_verified_kb": False
        }
        
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.__class__._latest_instance = self
        return Observation(done=False, reward=0.0, metadata={"difficulty": self._current_task["difficulty"]})

    def _step_impl(self, action: Action, **kwargs) -> Observation: 
        return Observation(done=False, reward=0.0)

    def step(self, action: Action, **kwargs) -> Observation:
        active_instance = self
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