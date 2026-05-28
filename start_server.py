"""Persistent local launcher for the LitVerify AI Streamlit app.

Some desktop shells close detached Streamlit processes as soon as the launching
shell exits. This tiny supervisor keeps the child process alive for local demos
and restarts it if it exits unexpectedly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "streamlit.supervisor.log"
PID_PATH = ROOT / "streamlit.supervisor.pid"
STOP_PATH = ROOT / "streamlit.stop"

if os.name == "nt":
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    VENV_PY = ROOT / ".venv" / "bin" / "python"


def resolve_python() -> str:
    """Always prefer the project-local .venv interpreter."""
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def streamlit_command(port: int) -> list[str]:
    """Build the Streamlit command using the project's .venv Python."""
    return [
        resolve_python(),
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]


def start_child(port: int, log_file) -> subprocess.Popen:
    """Start Streamlit while keeping stdin open."""
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.Popen(
        streamlit_command(port),
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=log_file,
        stderr=log_file,
        creationflags=creationflags,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--restart-delay", type=float, default=2.0)
    args = parser.parse_args()

    STOP_PATH.unlink(missing_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")

    if not VENV_PY.exists():
        print(
            f"[LitVerify AI] WARNING: project .venv not found at {VENV_PY}. "
            "Falling back to current interpreter. Run setup.bat (Windows) or "
            "setup.sh first for a fully isolated environment.",
            file=sys.stderr,
        )

    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] supervisor started on "
            f"port {args.port} using {resolve_python()}\n"
        )
        while not STOP_PATH.exists():
            child = start_child(args.port, log_file)
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] streamlit pid={child.pid}\n")
            while child.poll() is None and not STOP_PATH.exists():
                time.sleep(1)
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
            code = child.returncode
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] streamlit exited code={code}\n")
            if not STOP_PATH.exists():
                time.sleep(args.restart_delay)

    PID_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
