#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# First-time setup: build a local .venv and install dependencies.
# After this, run.sh is enough to start the app.

if command -v python3.12 >/dev/null 2>&1; then
    BOOT_PY="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    BOOT_PY="python3"
else
    BOOT_PY="python"
fi

echo "[DataWise AI] Creating .venv with ${BOOT_PY} ..."
"${BOOT_PY}" -m venv .venv

if [ -x ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
else
    VENV_PY=".venv/Scripts/python.exe"
fi

echo "[DataWise AI] Upgrading pip / wheel / setuptools ..."
"${VENV_PY}" -m pip install --upgrade pip wheel setuptools

echo "[DataWise AI] Installing requirements ..."
"${VENV_PY}" -m pip install -r requirements.txt

echo
echo "[DataWise AI] Setup complete. Launch with ./run.sh."
