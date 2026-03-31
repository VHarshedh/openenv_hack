"""
Integration test for support_env using the WebSocket client.

Verifies that trajectory rewards accumulate correctly across steps
for all 4 task difficulty levels.

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
        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")
        print(f"    Text: {extract_text(sr)[:80]}...")

        # Step 2: search_knowledge_base (+0.1 required = 0.3)
        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"query": "password reset"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        # Step 3: resolve_ticket (+0.3 correct action + 0.2*ratio keywords = ~0.8)
        sr = env.step(CallToolAction(
            tool_name="resolve_ticket",
            arguments={
                "message": "I've sent a password reset link to your email. "
                "Please check your spam folder. The link will expire in 24 hours."
            },
        ))
        print(f"  resolve → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 1 (easy)", sr.reward, sr.done))

        # ═══════════════════════════════════════════════
        # TASK 2 — Medium: Refund (correct: escalate to billing)
        # ═══════════════════════════════════════════════
        print(f"\n{'═' * 50}")
        print("TASK 2: Medium — Refund Past 30-Day Policy")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="check_billing", arguments={"user_id": "USR001"}
        ))
        print(f"  check_billing → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"query": "refund policy"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="escalate_ticket", arguments={"department": "billing"}
        ))
        print(f"  escalate(billing) → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 2 (medium)", sr.reward, sr.done))

        # ═══════════════════════════════════════════════
        # TASK 3 — Hard: Server Outage (correct: escalate to engineering)
        # ═══════════════════════════════════════════════
        print(f"\n{'═' * 50}")
        print("TASK 3: Hard — Server 500 Outage")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"query": "outage server error"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="escalate_ticket", arguments={"department": "engineering"}
        ))
        print(f"  escalate(engineering) → reward={sr.reward:.2f}, done={sr.done}")
        meta = getattr(sr.observation, "metadata", {}) or {}
        print(f"    tools_used: {meta.get('tools_used', [])}")
        results.append(("Task 3 (hard)", sr.reward, sr.done))

        # ═══════════════════════════════════════════════
        # TASK 4 — Trap: Unsolvable (correct: escalate)
        # ═══════════════════════════════════════════════
        print(f"\n{'═' * 50}")
        print("TASK 4: Trap — Unsolvable Request")
        print("═" * 50)

        r = env.reset()
        meta = getattr(r.observation, "metadata", {}) or {}
        print(f"  Difficulty: {meta.get('difficulty', '?')}")

        sr = env.step(CallToolAction(tool_name="read_ticket", arguments={}))
        print(f"  read_ticket → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="search_knowledge_base",
            arguments={"query": "data export"},
        ))
        print(f"  search_kb → reward={sr.reward}, done={sr.done}")

        sr = env.step(CallToolAction(
            tool_name="escalate_ticket", arguments={"department": "engineering"}
        ))
        print(f"  escalate(engineering) → reward={sr.reward:.2f}, done={sr.done}")
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
        if not done or reward < 0.3:
            status = "❌"
            all_pass = False
        else:
            status = "✅"
        print(f"  {status} {name}: reward={reward:.2f}, done={done}")

    avg = sum(r[1] for r in results) / len(results) if results else 0
    print(f"\n  Average reward: {avg:.2f}")

    if all_pass:
        print("\n  🎉 All tasks completed successfully!")
    else:
        print("\n  ⚠ Some tasks had issues.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
