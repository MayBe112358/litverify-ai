#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8501}"

if [ -x ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    VENV_PY=".venv/Scripts/python.exe"
else
    echo "[LitVerify AI] .venv not found. Creating it now..."
    ./setup.sh
    if [ -x ".venv/bin/python" ]; then
        VENV_PY=".venv/bin/python"
    else
        VENV_PY=".venv/Scripts/python.exe"
    fi
fi

echo "[LitVerify AI] Starting Streamlit on http://localhost:${PORT} ..."

exec "$VENV_PY" -m streamlit run app.py \
    --server.port "${PORT}" \
    --server.headless false \
    --browser.gatherUsageStats false \
    "$@"
