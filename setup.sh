#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# First-time setup: build a local .venv and install dependencies.
# After this, run.sh is enough to start the app.

# Try several interpreter names — few machines have exactly python3.12 —
# and verify each candidate actually runs and is 3.10+.
BOOT_PY=""
for cand in python3.12 python3.11 python3.13 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
        && "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        BOOT_PY="$cand"
        break
    fi
done
if [ -z "$BOOT_PY" ]; then
    echo "[DataWise AI] No usable Python 3.10+ found. Install Python 3.12 first." >&2
    exit 1
fi

# Downloads from pypi.org often stall on mainland-China networks. Default to
# the Tsinghua mirror; pre-set PIP_INDEX_URL to override.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
echo "[DataWise AI] Using PyPI index: ${PIP_INDEX_URL}"

echo "[DataWise AI] Creating .venv with ${BOOT_PY} ..."
"${BOOT_PY}" -m venv .venv

if [ -x ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
else
    VENV_PY=".venv/Scripts/python.exe"
fi

echo "[DataWise AI] Upgrading pip / wheel / setuptools ..."
"${VENV_PY}" -m pip install --upgrade pip wheel setuptools

echo "[DataWise AI] Installing runtime requirements ..."
"${VENV_PY}" -m pip install -r requirements.txt

echo
echo "[DataWise AI] Setup complete. Launch with ./run.sh."
echo "[DataWise AI] For tests, install dev tools with:"
echo "    ${VENV_PY} -m pip install -r requirements-dev.txt"
