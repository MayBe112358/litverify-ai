@echo off
setlocal
cd /d %~dp0

REM First-time setup: builds a local .venv next to this script and installs
REM all dependencies. After this, run.bat is enough to start the app.

REM ---- Pick a working Python 3.10+ ----------------------------------------
REM Try the py launcher with several versions first (a machine rarely has
REM exactly 3.12), then fall back to whatever "python" is on PATH. Each
REM candidate is PROBED with a real run: on fresh Windows "python" is a
REM Microsoft Store stub that prints an ad and exits non-zero, and py may
REM list versions that are not actually installed.
set "BOOT_PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined BOOT_PY (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "BOOT_PY=py -%%V"
    )
)
if not defined BOOT_PY (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "BOOT_PY=python"
)
if not defined BOOT_PY (
    echo [DataWise AI] No usable Python 3.10+ found on this machine.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add python.exe to PATH" in the installer,
    echo then run this script again.
    pause
    exit /b 1
)

REM ---- PyPI mirror ----------------------------------------------------------
REM Downloads from pypi.org often stall on mainland-China networks, which
REM made this script hang forever on other computers. Default to the
REM Tsinghua mirror; set PIP_INDEX_URL beforehand to override (e.g. back to
REM https://pypi.org/simple).
if not defined PIP_INDEX_URL set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
if not defined PIP_DEFAULT_TIMEOUT set "PIP_DEFAULT_TIMEOUT=60"
echo [DataWise AI] Using PyPI index: %PIP_INDEX_URL%

echo [DataWise AI] Creating .venv with %BOOT_PY% ...
%BOOT_PY% -m venv .venv
if errorlevel 1 (
    echo Failed to create .venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo [DataWise AI] Upgrading pip / wheel / setuptools ...
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
    echo pip upgrade failed - check your network connection. See output above.
    pause
    exit /b 1
)

echo [DataWise AI] Installing runtime requirements ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed. See output above.
    pause
    exit /b 1
)

echo.
echo [DataWise AI] Setup complete. Launch with run.bat.
echo [DataWise AI] For tests, install dev tools with:
echo     .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
endlocal
