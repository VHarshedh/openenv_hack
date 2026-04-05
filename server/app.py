# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Customer Support Triage Environment.

This module creates an HTTP server that exposes the SupportEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .support_env_environment import SupportEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.support_env_environment import SupportEnvironment

# Create the app with MCP types for action/observation
# Pass the class (factory) for WebSocket session support
app = create_app(
    SupportEnvironment, CallToolAction, CallToolObservation, env_name="support_env"
)

# Explicit /health endpoint — guarantees 200 OK for Docker HEALTHCHECK and
# the Phase 1 automated validator even if create_app's auto-route is not
# the first route resolved by the ASGI router.
from fastapi import Response  # noqa: E402

@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe: returns 200 OK with a healthy status payload."""
    return {"status": "healthy"}


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m support_env.server.app
        openenv serve support_env

    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
