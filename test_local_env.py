"""In-process SupportEnvironment smoke test; also pings HTTP /health via background server."""
import sys
import traceback
from pathlib import Path

import httpx

# Package root on sys.path for `server.*` and optional `openenv` from repo
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _http_test_server import start_background_server, stop_background_server
from openenv.core.env_server.mcp_types import CallToolAction
from server.support_env_environment import SupportEnvironment


def run_inprocess_smoke() -> None:
    env = SupportEnvironment()
    env.reset()
    print("Initialized (in-process)")

    obs = env.step(
        CallToolAction(
            tool_name="read_ticket",
            arguments={"thought": "12345 67890 12345"},
        )
    )
    print("Step returned:", obs)


def main() -> None:
    print("Starting background uvicorn for /health check...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}")
        httpx.get(f"{base}/health", timeout=5.0).raise_for_status()
        print("HTTP /health OK\n")

        run_inprocess_smoke()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    main()
