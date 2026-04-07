# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
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
_active_env: contextvars.ContextVar[Optional['SupportEnvironment']] = contextvars.ContextVar('_active_env', default=None)
_current_episode_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('_current_episode_id', default=None)

KNOWLEDGE_BASE = {
    # Core Policies — Tasks 1-6
    "password reset procedure":               "POLICY: Password Reset\n1. Verify identity.\n2. Send link.",
    "duplicate charge refund policy":         "POLICY: Refund Policy\n1. Refunds within 30 days.",
    "suspended account error":                "POLICY: Account Mgmt\n1. 500 error + 'suspended' -> SECURITY.",
    "refund policy account deletion conflict": "POLICY: Conflict\n1. Refund + Deletion -> BILLING.",
    "error code x77 payment failure":         "POLICY: X77\n1. If 'active', search KB for 'Shadow Ban Protocol'.",
    "shadow ban protocol security":           "POLICY: Shadow Ban\n1. ACTION: Escalate to SECURITY.",
    "gdpr deletion request":          "POLICY: GDPR Deletion\n1. If 'gdpr_flag' is active, standard deletion is blocked.\n2. Search KB for 'Privacy Officer'.",
    "privacy officer routing":        "POLICY: Privacy Officer\n1. Verify status of the assigned Privacy Officer UID.\n2. If status is 'red-flag', escalate to SECURITY immediately.",
    "email change request":                   "DEPRECATED POLICY (v1.2): For email changes, see 'Project Aegis' Internal Memo for latest security requirements.",
    "project aegis memo":                     "POLICY: Aegis Security\n1. If account has 'Real UID' mapping, you MUST verify the 'Compliance Auditor'.\n2. IF ticket date is within 48 hours of last transaction: Escalation is MANDATORY.\n3. Search KB for 'Aegis Audit' for final routing.",
    "aegis audit routing":                    "POLICY: Aegis Audit\n1. If Compliance Auditor status is NOT 'green', escalate to SECURITY.\n2. If Compliance Auditor is 'green' but account is 'flagged', escalate to BILLING.",
    # Task 7 — VIP Payment Update
    "vip account policy":    "POLICY: VIP Account Mgmt\n1. Accounts with 'vip_flag' = true MUST be escalated to BILLING, not Security.\n2. Do NOT resolve directly.",
    "payment method update": "POLICY: Payment Update\n1. Standard accounts: resolve directly.\n2. VIP accounts: see 'VIP Account Policy'.",
    # Task 8 — Security Breach
    "unauthorized access":   "POLICY: Unauthorized Login\n1. Collect breach details.\n2. Search KB for 'Data Breach Protocol'.",
    "data breach protocol":  "POLICY: Security Incident\n1. Verify with assigned Security Auditor.\n2. If Auditor status is 'active': escalate to ENGINEERING.\n3. If Auditor is NOT 'active': escalate to SECURITY immediately.",
    # Task 9 — Mega Chain
    "compliance hold policy":        "POLICY: Compliance Hold\n1. Accounts with 'compliance_hold' = true must NOT be modified.\n2. Verify with Compliance Auditor UID before any action.",
    "compliance resolution":         "POLICY: Compliance Resolution\n1. Auditor 'green': resolve account action.\n2. Auditor NOT 'green': escalate to SECURITY.\n3. If BOTH compliance_hold AND shadow_ban are active: escalate to ENGINEERING.",
    "shadow compliance intersection": "POLICY: Shadow + Compliance\n1. A Shadow Ban and Compliance Hold active simultaneously require ENGINEERING escalation.",
    # Decoy deprecated articles
    "legacy password policy v1":     "DEPRECATED (v0.9): Password resets via phone call are no longer supported. Refer to the current 'password reset procedure'.",
    "general escalation guidelines": "DEPRECATED: All escalations were previously routed to engineering by default. This policy is superseded by department-specific routing rules.",
}

# ---------------------------------------------------------------------------
# Module-level helpers and routing constants
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Return a set of lowercase word-boundary tokens. Prevents substring false positives."""
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


# Canonical trigger words for god-query detection
_CANONICAL_TRIGGERS: frozenset = frozenset({
    "password", "refund", "delete", "account", "500",
    "suspended", "x77", "email", "aegis", "audit", "shadow",
    "vip", "breach", "compliance", "gdpr", "officer",
})

# KB routing table: synonym-expanded, word-boundary safe.
# Each entry: synonyms (set), milestone (str|None), KB key (str), guard (lambda tokens->bool)
KB_ROUTES: list = [
    # --- Tasks 1-6 ---
    {"synonyms": {"password","reset","locked","lock","login","credential","lockout"},
     "milestone": "found_password_policy", "key": "password reset procedure",
     "guard": lambda t: True},
    {"synonyms": {"refund","duplicate","double","charged","overcharged","billed"},
     "milestone": "found_refund_policy", "key": "duplicate charge refund policy",
     "guard": lambda t: not (t & {"delete","account","removal","cancel","conflict"})},
    {"synonyms": {"refund","delete","account","removal","cancel","conflict","deletion"},
     "milestone": "found_account_policy", "key": "refund policy account deletion conflict",
     "guard": lambda t: ("refund" in t and bool(t & {"delete","account","removal","cancel","conflict","deletion"})) or "conflict" in t},
    {"synonyms": {"500","suspended","suspension"},
     "milestone": "found_500_policy", "key": "suspended account error",
     "guard": lambda t: bool(t & {"500","suspended","suspension"})},
    {"synonyms": {"x77"},
     "milestone": "found_x77_policy", "key": "error code x77 payment failure",
     "guard": lambda t: "x77" in t},
    {"synonyms": {"shadow","shadowban"},
     "milestone": "found_shadow_policy", "key": "shadow ban protocol security",
     "guard": lambda t: "shadow" in t},
    {"synonyms": {"email","mail"},
     "milestone": "found_email_policy", "key": "email change request",
     "guard": lambda t: bool(t & {"email","mail"})},
    {"synonyms": {"aegis","memo"},
     "milestone": "found_aegis_memo", "key": "project aegis memo",
     "guard": lambda t: "aegis" in t and "audit" not in t},
    {"synonyms": {"audit","auditor"},
     "milestone": "found_aegis_audit", "key": "aegis audit routing",
     "guard": lambda t: "audit" in t},
    # --- Task 7 ---
    {"synonyms": {"vip","premium","priority"},
     "milestone": "found_vip_policy", "key": "vip account policy",
     "guard": lambda t: "vip" in t},
    {"synonyms": {"payment","method","update","card"},
     "milestone": "found_payment_policy", "key": "payment method update",
     "guard": lambda t: bool(t & {"payment","method","update","card"})},
    # --- Task 8 ---
    {"synonyms": {"unauthorized","suspicious","hacked","intrusion"},
     "milestone": "found_unauthorized_policy", "key": "unauthorized access",
     "guard": lambda t: bool(t & {"unauthorized","suspicious","hacked","intrusion"})},
    {"synonyms": {"breach","incident"},
     "milestone": "found_breach_policy", "key": "data breach protocol",
     "guard": lambda t: bool(t & {"breach","incident"})},
    {"synonyms": {"gdpr", "eu", "privacy", "data"},
     "milestone": "found_gdpr_policy", "key": "gdpr deletion request",
     "guard": lambda t: "gdpr" in t or "data" in t},
    {"synonyms": {"officer", "privacy", "routing"},
     "milestone": "found_privacy_officer", "key": "privacy officer routing",
     "guard": lambda t: "officer" in t},
    # --- Task 9 ---
    {"synonyms": {"compliance","hold","frozen"},
     "milestone": "found_compliance_hold", "key": "compliance hold policy",
     "guard": lambda t: "compliance" in t and "resolution" not in t and "shadow" not in t},
    {"synonyms": {"compliance","resolution"},
     "milestone": "found_compliance_resolution", "key": "compliance resolution",
     "guard": lambda t: "compliance" in t and "resolution" in t},
    {"synonyms": {"shadow","compliance"},
     "milestone": "found_shadow_compliance", "key": "shadow compliance intersection",
     "guard": lambda t: "shadow" in t and "compliance" in t},
    # --- Decoys (no milestone) ---
    {"synonyms": {"phone","legacy"},
     "milestone": None, "key": "legacy password policy v1",
     "guard": lambda t: "phone" in t},
    {"synonyms": {"escalation","general"},
     "milestone": None, "key": "general escalation guidelines",
     "guard": lambda t: "escalation" in t and "audit" not in t and "aegis" not in t},
]

# Maps difficulty string to task index (used by reset() difficulty kwarg)
_DIFFICULTY_MAP: dict = {
    "easy": 0, "medium": 1, "hard": 2, "trap": 3,
    "vip": 4, "multi_hop": 5, "breach": 6, "privacy": 7, "ultra": 8, "mega": 9,
}

# Historical CRM noise injected into each task user's DB record on reset()
CRM_NOISE: list = [
    "NOTE (6 months ago): User inquired about a refund. Case closed.",
    "NOTE (3 months ago): Password reset completed successfully.",
    "NOTE (90 days ago): Billing dispute raised. No action taken.",
    "NOTE (1 year ago): Account suspension requested. Reinstated.",
    "NOTE (2 months ago): Flagged for suspicious login. Security cleared.",
    "NOTE (8 months ago): Duplicate charge reported. Refunded.",
    "NOTE (4 months ago): User requested feature upgrade. Forwarded to product.",
]

class SupportEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    # Class-level reference to recover state if HTTP session routing drops the episode_id
    _latest_instance = None
    _instances: dict[str, 'SupportEnvironment'] = {}

    def __init__(self):
        # Convert state back to instance variables for pure concurrency
        self._task_index: int = 0
        self._current_task: dict = {}
        self._episode_done: bool = False
        self._trajectory_reward: float = 0.0
        self._tools_used: list[str] = []
        self._db: dict = {}
        self._active_noise_policies: list[dict] = []
        
        # Dynamic Target Placeholders — Tasks 1-6
        self._target_easy = ""
        self._target_medium = ""
        self._target_hard = ""
        self._target_trap = ""
        self._target_multi = ""
        self._target_ultra = ""
        self._real_uid = ""
        self._auditor_uid = ""
        # Dynamic Target Placeholders — Tasks 7-9
        self._target_vip = ""
        self._target_breach = ""
        self._target_mega = ""
        self._target_privacy = ""
        self._privacy_officer_uid = ""
        self._breach_auditor_uid = ""
        self._mega_compliance_uid = ""

        self._progress: dict = {
            "read": False, 
            "searched_kb": False, 
            "checked_db": False, 
            "distracted": False,
            "pinged_manager": False,
            "out_of_order": False,
            "god_query": False,
            "teleported": False,
            "lazy_resolution": False,
            "sla_breached": False,
            "sop_violations": 0,
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
            env = _active_env.get() or SupportEnvironment._latest_instance
            env._validate_thought(thought, "read_ticket")
            env._progress["read"] = True
            
            ticket_text = env._current_task.get("ticket_text", "Error: Ticket missing.")
            optimal_steps = env._current_task.get("required_steps", 4)
            sla_limit = optimal_steps + 3
            
            return f"{ticket_text}\n\n[METADATA: SLA Limit = {sla_limit} actions. Efficiency penalties apply if exceeded.]"

        @mcp.tool
        def search_knowledge_base(thought: str, query: str) -> str:
            """Search the standard operating procedure knowledge base."""
            env = _active_env.get() or SupportEnvironment._latest_instance
            env._validate_thought(thought, "search_knowledge_base")
            if not env._progress["read"]: env._progress["out_of_order"] = True

            tokens = _tokenize(query)

            # Anti-Cheat: God Query — count canonical + noise hits (word-boundary safe)
            noise_kws: set = set()
            for p in env._active_noise_policies:
                noise_kws.update(p["keywords"])
            total_hits = len(tokens & _CANONICAL_TRIGGERS) + len(tokens & noise_kws)
            if total_hits > 3:
                env._progress["god_query"] = True
                return "SYSTEM ERROR: Query too broad. Please refine your search to specific keywords."

            results = []

            # Route via KB_ROUTES (synonym-expanded, word-boundary safe)
            for route in KB_ROUTES:
                if tokens & route["synonyms"] and route["guard"](tokens):
                    text = KNOWLEDGE_BASE.get(route["key"], "")
                    if text and text not in results:
                        results.append(text)
                        if env._progress["read"] and route["milestone"]:
                            env._progress["milestones"].add(route["milestone"])

            # Noise Routes (Dynamic — unchanged logic)
            q = query.lower()
            for p in env._active_noise_policies:
                if any(k in q for k in p["keywords"]):
                    results.append(p["text"])

            if results and env._progress["read"]:
                env._progress["searched_kb"] = True

            res = "\n\n".join(results) if results else "No specific policies found."
            
            # [NEW] SLA Breach Warning
            optimal_steps = env._current_task.get("required_steps", 4)
            sla_limit = optimal_steps + 3
            if env._state.step_count > sla_limit:
                res += f"\n\n⚠️ SYSTEM_WARNING: SLA Breached ({env._state.step_count}/{sla_limit} steps). Efficiency penalty of -0.05 active."
                
            return res

        @mcp.tool
        def check_billing(thought: str, user_id: str) -> str:
            """Check account status."""
            env = _active_env.get() or SupportEnvironment._latest_instance
            env._validate_thought(thought, "check_billing")
            if not env._progress["read"]: env._progress["out_of_order"] = True

            # Anti-Cheat: Teleportation — Tasks 1-6
            if user_id == env._real_uid and "found_aegis_memo" not in env._progress["milestones"]:
                env._progress["teleported"] = True
            if user_id == env._auditor_uid and (
                "checked_real_uid" not in env._progress["milestones"]
                or "found_aegis_audit" not in env._progress["milestones"]
            ):
                env._progress["teleported"] = True
            # Anti-Cheat: Teleportation — Tasks 7-9
            if user_id == env._breach_auditor_uid and "found_breach_policy" not in env._progress["milestones"]:
                if env._target_breach in env._progress["queried_users"]:
                    env._progress["teleported"] = True
            if user_id == env._privacy_officer_uid and "found_gdpr_policy" not in env._progress["milestones"]:
                if env._target_privacy in env._progress["queried_users"]:
                    env._progress["teleported"] = True
            if user_id == env._mega_compliance_uid and env._target_mega not in env._progress["queried_users"]:
                env._progress["teleported"] = True

            env._progress["queried_users"].add(user_id)
            user = env._db.get(user_id)
            if not user: return f"User {user_id} not found."

            if env._progress["read"]:
                env._progress["checked_db"] = True
                # Task 6 milestones
                if user_id == env._real_uid:          env._progress["milestones"].add("checked_real_uid")
                if user_id == env._auditor_uid:       env._progress["milestones"].add("checked_compliance_auditor")
                # Task 8 milestones
                if user_id == env._breach_auditor_uid: env._progress["milestones"].add("checked_breach_auditor")
                if user_id == env._privacy_officer_uid: env._progress["milestones"].add("checked_privacy_officer")
                # Task 9 milestones
                if user_id == env._mega_compliance_uid: env._progress["milestones"].add("checked_mega_auditor")

            res = f"Account: {user['name']}\nStatus: {user['status']}\nLast Txn: {user.get('last_txn', 'N/A')}"
            if "real_uid" in user:              res += f"\nNote: Real UID is {user['real_uid']}"
            if "compliance_auditor" in user:    res += f"\nAssigned Auditor: {user['compliance_auditor']}"
            if "security_auditor" in user:      res += f"\nSecurity Auditor: {user['security_auditor']}"
            if user.get("vip_flag"):            res += "\nAccount Type: VIP"
            if user.get("security_incident"):   res += "\nAlert: Security Incident Reported"
            if user.get("gdpr_flag"):           res += "\nFlag: GDPR Privacy Hold Active"
            if "privacy_officer" in user:       res += f"\nPrivacy Officer: {user['privacy_officer']}"
            if user.get("compliance_hold"):     res += "\nFlag: Compliance Hold Active"
            if user.get("shadow_ban"):          res += "\nFlag: Shadow Ban Active"
            if user.get("x77_flag"):            res += "\nError: X77 Payment Failure Detected"
            if "crm_notes" in user:
                res += f"\nCRM History: {' | '.join(user['crm_notes'])}"

            # [NEW] SLA Breach Warning
            optimal_steps = env._current_task.get("required_steps", 4)
            sla_limit = optimal_steps + 3
            if env._state.step_count > sla_limit:
                res += f"\n\n⚠️ SYSTEM_WARNING: SLA Breached ({env._state.step_count}/{sla_limit} steps). Efficiency penalty of -0.05 active."
                
            return res
            
        @mcp.tool
        def ping_human_manager(thought: str, reason: str) -> str:
            """Ask a manager for help."""
            env = _active_env.get() or SupportEnvironment._latest_instance
            env._validate_thought(thought, "ping_human_manager")
            env._progress["pinged_manager"] = True
            return "SYSTEM AUTO-REPLY: All managers are currently offline. Please follow standard SOP."

        @mcp.tool
        def escalate_ticket(thought: str, department: str) -> str:
            """Escalate to another department (billing, security, engineering)."""
            env = _active_env.get() or SupportEnvironment._latest_instance
            env._validate_thought(thought, "escalate_ticket")
            # Process Supervision: SOP soft-block (easy tasks exempt — no DB check required)
            diff = env._current_task.get("difficulty", "")
            if not env._progress["checked_db"] and diff != "easy":
                if env._progress["sop_violations"] < 3:
                    env._progress["sop_violations"] += 1
                    warn_n = env._progress["sop_violations"]
                    return (
                        f"⚠️ SYSTEM_REJECT: SOP VIOLATION — You must verify the user's "
                        f"account status in the billing CRM before escalating. "
                        f"Call check_billing first, then retry. [Warning {warn_n}/3]"
                    )
                # 4th+ attempt after 3 warnings — fall through and finalise
            env._finalize_episode("escalate", department=department)
            return f"Escalated to {department.upper()}."

        @mcp.tool
        def resolve_ticket(thought: str, message: str) -> str:
            """Resolve the issue and message the user."""
            env = _active_env.get() or SupportEnvironment._latest_instance
            env._validate_thought(thought, "resolve_ticket")

            # Process Supervision: must read ticket first
            if not env._progress["read"]:
                if env._progress["sop_violations"] < 3:
                    env._progress["sop_violations"] += 1
                    warn_n = env._progress["sop_violations"]
                    return (
                        f"⚠️ SYSTEM_REJECT: SOP VIOLATION — You must read the ticket "
                        f"using read_ticket before attempting to resolve it. "
                        f"[Warning {warn_n}/3]"
                    )

            msg_lower = message.lower()
            diff = env._current_task.get("difficulty")

            # Anti-Cheat 1: Hallucinated placeholder detection.
            if re.search(r"\[[^\]]+\]", message) or re.search(r"<[^>]+>", message):
                env._progress["lazy_resolution"] = True

            # Anti-Cheat 2: Name/ID grounding check (only if the agent has queried the DB,
            # so they had the opportunity to discover the name).
            if diff in ("easy", "medium") and env._progress.get("checked_db", False):
                target_user = env._current_task.get("target_user", "")
                db_record = env._db.get(target_user, {})
                user_name = db_record.get("name", "").lower()
                name_parts = [p.strip() for p in user_name.split() if len(p.strip()) > 2]
                id_present = target_user.lower() in msg_lower
                name_present = any(part in msg_lower for part in name_parts) if name_parts else False
                if not (id_present or name_present):
                    env._progress["lazy_resolution"] = True

            # Anti-Cheat 3: Difficulty-specific content check.
            if diff == "easy" and not any(w in msg_lower for w in ["password", "reset", "link", "email"]):
                env._progress["lazy_resolution"] = True
            if diff == "medium" and not any(w in msg_lower for w in ["refund", "process", "duplicate", "charge"]):
                env._progress["lazy_resolution"] = True

            env._finalize_episode("resolve")
            return "Resolved."

        # Pass mcp as a positional argument
        super().__init__(mcp)
        
        # Hard fail-safe to guarantee the missing property from the Traceback always exists
        if not hasattr(self, "_mode_tools"):
            self._mode_tools = []

    def _validate_thought(self, thought: str, tool_name: str | None = None):
        """Scans the thought parameter contextually to prove step-by-step reasoning."""
        if not thought:
            self._progress["thought_missing"] = True
            return
            
        thought_lower = thought.lower()
        
        # Step 1: Identify all distinct requests (Can happen anytime)
        if any(w in thought_lower for w in [
            "request", "distinct", "issue", "problem", "ticket",
            "user wants", "inquiry", "wants", "needs", "asking", "complaint"
        ]):
            self._progress["thought_identified_request"] = True
            
        # Step 2: Note specific states (Context-Aware: Only valid if they have actually queried the DB OR are querying now)
        if self._progress.get("checked_db", False) or tool_name == "check_billing" or self._current_task.get("difficulty") == "easy":
            if any(w in thought_lower for w in [
                "status", "active", "suspended", "state", "red-flag",
                "banned", "disabled", "status is", "found in db"
            ]):
                self._progress["thought_noted_state"] = True
                
        # Step 3/4: Cross-reference & Verify against KB (Context-Aware: Only valid if they searched KB OR are searching now)
        if self._progress.get("searched_kb", False) or tool_name == "search_knowledge_base":
            if any(w in thought_lower for w in [
                "policy", "knowledge base", "kb", "escalation", "verify", "rule", "protocol",
                "guidelines", "rules", "instructions", "sop", "standard operating procedure"
            ]):
                self._progress["thought_verified_kb"] = True

    def _generate_dynamic_noise_policies(self, count: int) -> list[dict]:
        """Generates completely randomized corporate policies using built-in random."""
        subjects = [
            ("parking", "garage", "permit", "vehicle"),
            ("lunch", "catering", "food", "cafeteria"),
            ("travel", "flight", "hotel", "expenses"),
            ("hardware", "laptop", "monitor", "keyboard"),
            ("software", "license", "adobe", "installation"),
            ("dress", "attire", "shoes", "casual"),
            ("pets", "dog", "office", "animals")
        ]
        
        verbs = ["must be authorized by", "are strictly prohibited by", "require a 14-day notice to", "will be audited by", "should be reported to"]
        departments = ["Human Resources", "Facilities Management", "Legal", "the VP of Operations", "IT Support", "Corporate Compliance"]
        penalties = ["immediate termination.", "a formal warning.", "loss of privileges.", "a $50 fee.", "mandatory retraining."]
        
        policies = []
        # Pick 'count' random subjects without replacement
        sampled_subjects = random.sample(subjects, min(count, len(subjects)))
        
        for word_tuple in sampled_subjects:
            # The first word is the main subject
            main_subject = word_tuple[0].capitalize()
            
            # Build a highly randomized policy string
            text = (
                f"POLICY: {main_subject} Guidelines\n"
                f"1. Requests regarding {random.choice(word_tuple)} {random.choice(verbs)} {random.choice(departments)}.\n"
                f"2. Failure to comply will result in {random.choice(penalties)}"
            )
            
            policies.append({
                "keywords": list(word_tuple),  # The fuzzy search will look for any of these
                "text": text
            })
            
        return policies

    def _generate_dynamic_database(self) -> dict:
        num_dummy_users = random.randint(50, 150)
        db = {}
        benchmark_date = datetime(2026, 4, 1)
        
        first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
        
        # Generate the pool — needs at least 63 entries to guarantee keys[0..12] exist
        for i in range(1, num_dummy_users + 16):
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

        # Tasks 1-6
        self._target_easy   = keys[0]
        self._target_medium = keys[1]
        self._target_hard   = keys[2]
        self._target_trap   = keys[3]
        self._target_multi  = keys[4]
        self._target_ultra  = keys[5]
        self._real_uid      = keys[6]
        self._auditor_uid   = keys[7]
        # Tasks 7-9
        self._target_vip          = keys[8]
        self._target_breach       = keys[9]
        self._target_mega         = keys[10]
        self._breach_auditor_uid  = keys[11]
        self._mega_compliance_uid = keys[12]
        self._target_privacy      = keys[13]
        self._privacy_officer_uid = keys[14]

        # Inject task state conditions — Tasks 1-6
        db[self._target_easy]["status"]   = "active"
        db[self._target_medium]["status"] = "active"
        db[self._target_hard]["status"]   = "suspended"
        db[self._target_trap]["status"]   = "active"
        db[self._target_multi]["status"]  = "active"
        db[self._target_multi]["x77_flag"] = True

        db[self._target_ultra]["status"]   = "active"
        db[self._target_ultra]["real_uid"] = self._real_uid
        db[self._target_ultra]["last_txn"] = "2026-03-31"

        db[self._real_uid]["status"]             = "active"
        db[self._real_uid]["compliance_auditor"] = self._auditor_uid
        db[self._auditor_uid]["status"]           = "red-flag"

        # Inject task state conditions — Task 7 (VIP Payment)
        db[self._target_vip]["status"]   = "active"
        db[self._target_vip]["vip_flag"] = True

        # Inject task state conditions — Task 8 (Security Breach)
        db[self._target_breach]["status"]            = "active"
        db[self._target_breach]["security_incident"] = True
        db[self._target_breach]["security_auditor"]  = self._breach_auditor_uid
        db[self._breach_auditor_uid]["status"]        = "active"  # → ENGINEERING

        # Inject task state conditions — GDPR Privacy Hold
        db[self._target_privacy]["status"] = "active"
        db[self._target_privacy]["gdpr_flag"] = True
        db[self._target_privacy]["privacy_officer"] = self._privacy_officer_uid
        db[self._privacy_officer_uid]["status"] = "red-flag"

        # Inject task state conditions — Task 9 (Mega Chain)
        db[self._target_mega]["status"]             = "active"
        db[self._target_mega]["compliance_hold"]    = True
        db[self._target_mega]["shadow_ban"]         = True
        db[self._target_mega]["compliance_auditor"] = self._mega_compliance_uid
        db[self._target_mega]["x77_flag"]           = True
        db[self._mega_compliance_uid]["status"]      = "red-flag"  # + both flags → ENGINEERING

        # Inject CRM historical noise into all task target users
        for uid in [
            self._target_easy, self._target_medium, self._target_hard,
            self._target_trap, self._target_multi, self._target_ultra,
            self._target_vip, self._target_breach, self._target_mega,
            self._target_privacy,
        ]:
            db[uid]["crm_notes"] = random.sample(CRM_NOISE, k=random.randint(1, 2))

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
        # thought_verified_kb: award if agent demonstrated KB reasoning AND actually found a policy
        # (searched_kb set by successful route match + milestones non-empty = no free exploit)
        if self._progress.get("searched_kb") and bool(self._progress["milestones"]):
            thought_r += 0.2
        elif self._progress["thought_verified_kb"]:
            thought_r += 0.2
        
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
            if self._progress["read"]: task_r += 0.05
            if target_user in u: task_r += 0.05
            if "found_refund_policy" in m: task_r += 0.1
            if "found_account_policy" in m: task_r += 0.1
        elif diff == "multi_hop":
            if self._progress["read"]: task_r += 0.05
            if target_user in u: task_r += 0.05
            if "found_x77_policy" in m: task_r += 0.1
            if "found_shadow_policy" in m: task_r += 0.1
        elif diff == "ultra":
            if self._progress["read"]: task_r += 0.05
            if target_user in u: task_r += 0.05
            if "found_aegis_memo" in m: task_r += 0.05
            if "checked_real_uid" in m: task_r += 0.05
            if "found_aegis_audit" in m: task_r += 0.05
            if "checked_compliance_auditor" in m: task_r += 0.05
        elif diff == "vip":
            if self._progress["read"]: task_r += 0.1
            if target_user in u: task_r += 0.1
            if "found_vip_policy" in m: task_r += 0.1
        elif diff == "breach":
            if self._progress["read"]: task_r += 0.1
            if target_user in u: task_r += 0.1
            if "found_breach_policy" in m and "checked_breach_auditor" in m: task_r += 0.1
        elif diff == "privacy":
            if self._progress["read"]: task_r += 0.05
            if target_user in u: task_r += 0.05
            if "found_gdpr_policy" in m: task_r += 0.1
            if "checked_privacy_officer" in m: task_r += 0.1
        elif diff == "mega":
            if self._progress["read"]:                task_r += 0.05
            if target_user in u:                      task_r += 0.05
            if "found_compliance_resolution" in m:    task_r += 0.1
            if "found_shadow_compliance" in m:        task_r += 0.1
            
        r = thought_r + task_r

        # Global Penalties
        if self._progress.get("thought_missing"): r -= 0.1
        if self._progress["pinged_manager"]: r -= 0.3
        if self._progress["distracted"]:    r -= 0.2
        if self._progress["out_of_order"]:  r -= 0.1
        if self._progress["god_query"]:     r -= 0.2
        if self._progress["teleported"]:    r -= 0.3
        if self._progress["lazy_resolution"]: r -= 0.2
        if self._progress["sop_violations"] > 0:
            r -= self._progress["sop_violations"] * 0.05

        # [NEW] SLA Draining Timer (Padded)
        optimal_steps = self._current_task.get("required_steps", 4) if self._current_task else 4
        sla_limit = optimal_steps + 3
        
        if self._state.step_count > sla_limit:
            self._progress["sla_breached"] = True
            # Drain 0.05 for every step past the padded SLA
            r -= ((self._state.step_count - sla_limit) * 0.05)

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
            elif task_diff == "trap" and self._progress["read"] and target_user in u and "found_refund_policy" in m and "found_account_policy" in m: valid_progression = True
            elif task_diff == "multi_hop" and self._progress["read"] and target_user in u and "found_x77_policy" in m and "found_shadow_policy" in m: valid_progression = True
            elif task_diff == "ultra" and self._progress["read"] and target_user in u and "found_aegis_memo" in m and "checked_real_uid" in m and "found_aegis_audit" in m and "checked_compliance_auditor" in m: valid_progression = True
            elif task_diff == "vip" and self._progress["read"] and target_user in u and "found_vip_policy" in m: valid_progression = True
            elif task_diff == "breach" and self._progress["read"] and target_user in u and "found_breach_policy" in m and "checked_breach_auditor" in m: valid_progression = True
            elif task_diff == "privacy" and self._progress["read"] and target_user in u and "found_gdpr_policy" in m and "checked_privacy_officer" in m: valid_progression = True
            elif task_diff == "mega" and self._progress["read"] and target_user in u and "found_compliance_resolution" in m and "found_shadow_compliance" in m and "checked_mega_auditor" in m: valid_progression = True
            
            if valid_progression:
                bonus = 0.3

        r += bonus
        
        # Task 4 Conflict Awareness Penalty: Dock -0.2 if they resolved but only searched ONE of the two required policies.
        if task_diff == "trap":
            has_ref = "found_refund_policy" in m
            has_acc = "found_account_policy" in m
            if (has_ref != has_acc) and (has_ref or has_acc):
                r -= 0.2

        # Strict Clamping ensures the reward mathematically cannot exit 0.0 - 1.0 bounds
        self._trajectory_reward = max(0.0, min(1.0, r))

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        self._db = self._generate_dynamic_database()
        
        # Procedurally generate 3-5 entirely random noise policies for this episode
        self._active_noise_policies = self._generate_dynamic_noise_policies(random.randint(3, 5))
        
        options = kwargs.get("options", {})
        forced_idx = options.get("task_idx")
        forced_diff = options.get("difficulty")
        if forced_diff and forced_diff in _DIFFICULTY_MAP:
            self._task_index = _DIFFICULTY_MAP[forced_diff]
        elif forced_idx is not None:
            self._task_index = int(forced_idx)
        elif kwargs.get("task_idx") is not None:
             self._task_index = int(kwargs.get("task_idx"))
        elif kwargs.get("difficulty") is not None:
             self._task_index = _DIFFICULTY_MAP[kwargs.get("difficulty")]

        # Generate Procedural Ticket Text variations
        easy_texts = [f"Reset password for {self._target_easy}.", f"User {self._target_easy} is locked out, needs password reset.", f"Forgot password on account {self._target_easy}."]
        medium_texts = [f"Refund duplicate charge for {self._target_medium}.", f"Double billed user {self._target_medium}, please refund.", f"Process duplicate refund for {self._target_medium}."]
        hard_texts = [f"Deep Diagnosis: Error 500 for {self._target_hard}.", f"{self._target_hard} keeps getting Error 500.", f"Fix 500 internal error for {self._target_hard}."]
        trap_texts = [f"Conflict Awareness: Delete account and refund {self._target_trap}.", f"User {self._target_trap} wants a refund and account deletion.", f"Please refund my last order and delete my account ({self._target_trap})."]
        multi_texts = [f"Error X77 for {self._target_multi}.", f"Payment failed with X77 for {self._target_multi}.", f"Getting error code X77 on checkout, user {self._target_multi}."]
        ultra_texts = [f"Change email for {self._target_ultra}. URGENT.", f"Urgent email update requested for {self._target_ultra}.", f"Update email address for account {self._target_ultra} immediately."]

        vip_texts   = [f"Update payment method for {self._target_vip}.", f"Payment info change request for {self._target_vip}.", f"{self._target_vip} needs to update card on file."]
        breach_texts = [f"Unauthorized login attempt on {self._target_breach}.", f"{self._target_breach} reports suspicious account access.", f"Possible breach detected on account {self._target_breach}."]
        mega_texts  = [f"Account {self._target_mega} has a payment error and compliance issue. URGENT.", f"{self._target_mega}: X77 error, compliance hold active.", f"Multiple issues on {self._target_mega}: payment failure and compliance flag."]

        tasks = [
            {"difficulty": "easy",      "required_steps": 3,  "target_user": self._target_easy,   "ticket_text": random.choice(easy_texts),   "correct_action": "resolve",   "correct_dept": ""},
            {"difficulty": "medium",    "required_steps": 4,  "target_user": self._target_medium, "ticket_text": random.choice(medium_texts), "correct_action": "resolve",   "correct_dept": ""},
            {"difficulty": "hard",      "required_steps": 4,  "target_user": self._target_hard,   "ticket_text": random.choice(hard_texts),   "correct_action": "escalate",  "correct_dept": "security"},
            {"difficulty": "trap",      "required_steps": 5,  "target_user": self._target_trap,   "ticket_text": random.choice(trap_texts),   "correct_action": "escalate",  "correct_dept": "billing"},
            {"difficulty": "vip",       "required_steps": 5,  "target_user": self._target_vip,    "ticket_text": random.choice(vip_texts),    "correct_action": "escalate",  "correct_dept": "billing"},
            {"difficulty": "multi_hop", "required_steps": 5,  "target_user": self._target_multi,  "ticket_text": random.choice(multi_texts),  "correct_action": "escalate",  "correct_dept": "security"},
            {"difficulty": "breach",    "required_steps": 6,  "target_user": self._target_breach, "ticket_text": random.choice(breach_texts), "correct_action": "escalate",  "correct_dept": "engineering"},
            {"difficulty": "privacy",   "required_steps": 7,  "target_user": self._target_privacy, "ticket_text": f"Urgent: I demand immediate deletion of my data under GDPR ({self._target_privacy}).", "correct_action": "escalate",  "correct_dept": "security"},
            {"difficulty": "ultra",     "required_steps": 8,  "target_user": self._target_ultra,  "ticket_text": random.choice(ultra_texts),  "correct_action": "escalate",  "correct_dept": "security"},
            {"difficulty": "mega",      "required_steps": 10, "target_user": self._target_mega,   "ticket_text": random.choice(mega_texts),   "correct_action": "escalate",  "correct_dept": "engineering"},
        ]

        
        self._current_task = tasks[self._task_index % len(tasks)]
        self._task_index += 1
        self._episode_done = False
        self._trajectory_reward = 0.0
        self._tools_used = []
        
        self._progress = {
            "read": False, "searched_kb": False, "checked_db": False,
            "distracted": False, "pinged_manager": False, "out_of_order": False,
            "god_query": False, "teleported": False, "lazy_resolution": False,
            "sla_breached": False,
            "sop_violations": 0,
            "milestones": set(), "queried_users": set(),
            # Tracking Instruction Following
            "thought_identified_request": False,
            "thought_noted_state": False,
            "thought_verified_kb": False
        }
        
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.__class__._latest_instance = self
        self.__class__._instances[self._state.episode_id] = self
        return Observation(done=False, reward=0.0, metadata={"difficulty": self._current_task["difficulty"]})

    def _step_impl(self, action: Action, **kwargs) -> Observation: 
        return Observation(done=False, reward=0.0)

    def step(self, action: Action, **kwargs) -> Observation:
        active_instance = self
        
        req_episode_id = kwargs.get("episode_id") or _current_episode_id.get()
        if req_episode_id and req_episode_id in self.__class__._instances:
            active_instance = self.__class__._instances[req_episode_id]
        # Safely check if self has an empty task (FastAPI dummy instance check)
        elif not getattr(active_instance, "_current_task", None) and getattr(self.__class__, "_latest_instance", None):
            active_instance = self.__class__._latest_instance
            
        token = _active_env.set(active_instance)
        try:
            if isinstance(action, ListToolsAction): 
                return super().step(action, **kwargs)
                
            if isinstance(action, CallToolAction):
                active_instance._state.step_count += 1
                active_instance._tools_used.append(action.tool_name)
                
            # Execute on the CURRENT thread's instance so FastMCP doesn't silently fail cross-thread!
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