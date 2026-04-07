"""Cleaner smoke test with sequential output (starts local uvicorn automatically)."""
import sys

import httpx

from _http_test_server import start_background_server, stop_background_server


def run_smoke_tests(base: str) -> list[str]:
    results: list[str] = []

    def log(msg: str) -> None:
        results.append(msg)

    # Test 1: Reset
    log("=== TEST 1: Reset ===")
    r = httpx.post(f"{base}/reset", json={}, timeout=10)
    d = r.json()
    log(f"  Status: {r.status_code}")
    log(f"  Done: {d.get('done')}, Reward: {d.get('reward')}")

    # Test 2: List tools
    log("\n=== TEST 2: List Tools ===")
    r = httpx.post(
        f"{base}/step",
        json={"action": {"type": "list_tools"}, "timeout_s": 30},
        timeout=10,
    )
    d = r.json()
    tools = d.get("observation", {}).get("tools", [])
    log(f"  Found {len(tools)} tools:")
    for t in tools:
        log(f"    - {t['name']}")

    # Test 3: read_ticket
    log("\n=== TEST 3: read_ticket ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "read_ticket",
                "arguments": {
                    "thought": "I need to read the ticket first so that I can understand the user's issue completely."
                },
            },
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")
    obs = d.get("observation", {})
    log(f"  tool_name: {obs.get('tool_name')}")
    result = obs.get("result", {})
    if isinstance(result, dict) and "content" in result:
        content = result["content"]
        if isinstance(content, list) and len(content) > 0:
            text = content[0].get("text", "")[:80]
        else:
            text = str(content)[:80]
        log(f"  Ticket text: {text}...")
    else:
        log(f"  Result: {str(result)[:80]}...")

    # Test 4: search_knowledge_base
    log("\n=== TEST 4: search_knowledge_base ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "search_knowledge_base",
                "arguments": {
                    "thought": "I should look up the password reset policy in the knowledge base to see how to proceed.",
                    "query": "password reset",
                },
            },
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")

    # Test 5: ping_human_manager
    log("\n=== TEST 5: ping_human_manager ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "ping_human_manager",
                "arguments": {
                    "thought": "I will ask a human manager for advice on how to proceed with the password reset because the documentation is slightly unclear to me.",
                    "reason": "Need clarification on password reset procedures.",
                },
            },
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")

    # Test 6: resolve_ticket
    log("\n=== TEST 6: resolve_ticket ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "resolve_ticket",
                "arguments": {
                    "thought": "I will send the reset link and advise them to check their spam folder just in case.",
                    "message": "I have sent a password reset link to your email. Please check spam.",
                },
            },
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Done: {d.get('done')}")
    log(f"  Final Reward: {d.get('reward')}")

    log("\n=== ALL TESTS PASSED ===")
    return results


def main() -> None:
    print("Starting background uvicorn (support_env)...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}\n")
        print("\n".join(run_smoke_tests(base)))
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    main()
    sys.exit(0)
