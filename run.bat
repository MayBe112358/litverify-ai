@echo off
setlocal
cd /d %~dp0

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PORT=8501"

if not exist "%VENV_PY%" (
    echo [LitVerify AI] .venv not found. Creating it now...
    call setup.bat
    if errorlevel 1 (
        echo [LitVerify AI] Setup failed.
        pause
        exit /b 1
    )
)

REM Probe the venv before launching: a .venv copied from another computer
REM keeps that machine's Python path inside pyvenv.cfg, so python.exe exists
REM but dies instantly with "No Python at ..." - which used to close this
REM window before anyone could read the error. Rebuild it on this machine.
"%VENV_PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [LitVerify AI] .venv is broken - it was probably copied from another
    echo computer ^(a Python virtual env is not portable^). Rebuilding here...
    rmdir /s /q .venv
    call setup.bat
    if errorlevel 1 (
        echo [LitVerify AI] Setup failed.
        pause
        exit /b 1
    )
)

echo [LitVerify AI] Starting Streamlit on http://localhost:%PORT% ...
echo [LitVerify AI] The browser will pop up in a few seconds.

"%VENV_PY%" -m streamlit run app.py --server.port %PORT% --server.headless false --browser.gatherUsageStats false %*

if errorlevel 1 (
    echo.
    echo [LitVerify AI] Streamlit exited with an error - see the message above.
    echo Common causes: port %PORT% already in use, or broken dependencies
    echo ^(delete the .venv folder and run setup.bat again^).
    pause
    exit /b 1
)
endlocal
