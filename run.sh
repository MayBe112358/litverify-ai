#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8501}"

if [ -x ".venv/bin/python" ]; then
    VENV_PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    VENV_PY=".venv/Scripts/python.exe"
else
    echo "[LitVerify AI] .venv not found." >&2
    echo "Run setup.sh once, or:" >&2
    echo "    python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

echo "[LitVerify AI] Starting Streamlit on http://localhost:${PORT} ..."

exec "$VENV_PY" -m streamlit run app.py \
    --server.port "${PORT}" \
    --server.headless false \
    --browser.gatherUsageStats false \
    "$@"
