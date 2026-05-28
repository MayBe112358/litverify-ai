@echo off
setlocal
cd /d %~dp0

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PORT=8501"

if not exist "%VENV_PY%" (
    echo [LitVerify AI] .venv not found.
    echo Run setup.bat once to create it, or:
    echo     python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [LitVerify AI] Starting Streamlit on http://localhost:%PORT% ...
echo [LitVerify AI] The browser will pop up in a few seconds.

"%VENV_PY%" -m streamlit run app.py --server.port %PORT% --server.headless false --browser.gatherUsageStats false %*

endlocal
