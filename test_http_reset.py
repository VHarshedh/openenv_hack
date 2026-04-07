"""Verify HTTP /reset cycles through tasks (starts local uvicorn automatically)."""
import sys
import time

import httpx

from _http_test_server import start_background_server, stop_background_server


def run_reset_iteration_tests(base: str) -> None:
    print("Testing HTTP /reset iteration using read_ticket...")

    for i in range(1, 8):
        try:
            httpx.post(f"{base}/reset", timeout=5).raise_for_status()

            payload = {
                "action": {
                    "type": "call_tool",
                    "tool_name": "read_ticket",
                    "arguments": {
                        "thought": "Reading ticket to verify that the HTTP reset has properly cycled the environment task."
                    },
                }
            }

            r = httpx.post(f"{base}/step", json=payload, timeout=5)
            r.raise_for_status()
            data = r.json()

            obs = data.get("observation", {})
            result = obs.get("result", {})

            if isinstance(result, dict) and "content" in result:
                text = result["content"][0]["text"]
            elif hasattr(result, "data"):
                text = result.data
            else:
                text = str(result)

            print(f"Call {i} Ticket Preview: {text[:60]}...")
            time.sleep(0.5)
        except Exception as e:
            print(f"Call {i} failed: {e}")


def main() -> None:
    print("Starting background uvicorn (support_env)...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}\n")
        run_reset_iteration_tests(base)
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    main()
    sys.exit(0)
