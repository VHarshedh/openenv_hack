"""
Integration test for support_env using the WebSocket client.

Verifies that trajectory rewards accumulate correctly across steps
for all 4 task difficulty levels, including strict Chain of Thought constraints.

Usage (with server already running on :8000):
    PYTHONPATH=../src:../envs python test_ws.py
"""
import sys
import os

# Ensure we can import from the repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))

from client import SupportEnv
from openenv.core.env_server.mcp_types import CallToolAction

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")


def extract_text(step_result) -> str:
    """Get the text result from a step."""
    obs = step_result.observation
    result = getattr(obs, "result", None)
    if result is None:
        return "(no result)"
    if hasattr(result, "data"):
        return str(result.data)
    if isinstance(result, dict):
        if "data" in result:
            return str(result["data"])
        if "content" in result:
            c = result["content"]
            if isinstance(c, list) and c:
                return c[0].get("text", str(c[0])) if isinstance(c[0], dict) else str(c[0])
            return str(c)
    return str(result)


def run_all_tests():
    results = []

    with SupportEnv(base_url=ENV_URL).sync() as env:

        # ═══════════════════════════════════════════════
        # TASK 1 — Easy: Password Reset (correct: resolve)
        # ═══════════════════════════════════════════════
        print("═" * 50)
        print("TASK 1: Easy — Password Reset")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        # Step 1: read_ticket (+0.1 required + 0.1 first = 0.2)
        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={"thought": "Read the new ticket."}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")
        print(f"    Text: {extract_text(sr)[:80]}...")

        # Step 2: search_knowledge_base (+0.1 required = 0.3)
        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"thought": "Look up password reset.", "query": "password reset"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        # Step 3: resolve_ticket (+0.3 correct action + 0.2*ratio keywords = ~0.8)
        sr = env.step(CallToolAction(
            tool_name="resolve_ticket",
            arguments={
                "thought": "Resolving with all required keywords.",
                "message": "I have sent a password reset link to your email. Please check your spam folder."
            },
        ))
        print(f"  resolve → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 1 (easy)", sr.reward, sr.done))

        # ═══════════════════════════════════════════════
        # TASK 2 — Medium: Duplicate Charge (correct: resolve)
        # ═══════════════════════════════════════════════
        print(f"\n{'═' * 50}")
        print("TASK 2: Medium — Duplicate Charge Identification")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={"thought": "Read the new ticket."}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"thought": "Look up duplicate charges.", "query": "duplicate charge"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="check_billing", arguments={"thought": "Check array for duplicate.", "user_id": "USR003"}
        ))
        print(f"  check_billing → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="resolve_ticket",
            arguments={
                "thought": "Found the duplicate, processing refund.",
                "message": "I have processed a refund for the extra duplicate charge that was issued."
            },
        ))
        print(f"  resolve → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 2 (medium)", sr.reward, sr.done))

        # ═══════════════════════════════════════════════
        # TASK 3 — Hard: Suspension (correct: escalate to security)
        # ═══════════════════════════════════════════════
        print(f"\n{'═' * 50}")
        print("TASK 3: Hard — Multi-Step Suspension Diagnosis")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={"thought": "Read the new ticket."}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="check_billing", arguments={"thought": "Check if user has billing issues.", "user_id": "USR004"}
        ))
        print(f"  check_billing → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"thought": "Look up suspended accounts.", "query": "suspended account error"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="escalate_ticket", arguments={"thought": "Suspension found, routing to security.", "department": "security"}
        ))
        print(f"  escalate(security) → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 3 (hard)", sr.reward, sr.done))

        # ═══════════════════════════════════════════════
        # TASK 4 — Trap: Policy Conflict (correct: escalate to billing)
        # ═══════════════════════════════════════════════
        print(f"\n{'═' * 50}")
        print("TASK 4: Trap — Cross-Policy Conflict")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={"thought": "Read the new ticket."}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="check_billing", arguments={"thought": "Check transaction dates.", "user_id": "USR001"}
        ))
        print(f"  check_billing → reward={sr.reward}, done={sr.done}")

        # Trap logic: Must search both "refund" and "account" to avoid the -0.2 penalty
        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"thought": "Check refund rule.", "query": "refund policy"},
        ))
        print(f"  search_kb(refund) → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"thought": "Check account rule.", "query": "account deletion conflict"},
        ))
        print(f"  search_kb(account) → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="escalate_ticket", arguments={"thought": "Conflict detected, escalating to billing.", "department": "billing"}
        ))
        print(f"  escalate(billing) → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 4 (trap)", sr.reward, sr.done))

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'═' * 50}")
    print("SUMMARY")
    print("═" * 50)
    all_pass = True
    for name, reward, done in results:
        # Require a strict > 0.8 reward for a "pass" to validate perfect trajectories
        if not done or reward < 0.8:
            status = "❌"
            all_pass = False
        else:
            status = "✅"
        print(f"  {status} {name}: reward={reward:.2f}, done={done}")

    avg = sum(r[1] for r in results) / len(results) if results else 0
    print(f"\n  Average reward: {avg:.2f}")

    if all_pass:
        print("\n  🎉 All tasks completed successfully with perfect trajectories!")
    else:
        print("\n  ⚠ Some tasks had issues or sub-optimal trajectories.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()