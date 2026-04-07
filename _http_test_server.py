# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Shared helper: start/stop support_env FastAPI in a subprocess for standalone tests.

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Directory containing server.app (this package root)
_SUPPORT_ENV_ROOT = Path(__file__).resolve().parent


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_until_ready(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    health = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(health, timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Server at {base_url} did not become ready within {timeout_s}s")


def start_background_server(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> tuple[subprocess.Popen, str]:
    """Spawn uvicorn for ``server.app:app``; return (process, base URL)."""
    p = port if port is not None else pick_free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            host,
            "--port",
            str(p),
        ],
        cwd=str(_SUPPORT_ENV_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://{host}:{p}"
    wait_until_ready(base)
    return proc, base


def stop_background_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
