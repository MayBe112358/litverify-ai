@echo off
setlocal
cd /d %~dp0

REM First-time setup: builds a local .venv next to this script and installs
REM all dependencies. After this, run.bat is enough to start the app.

REM Prefer the Windows py launcher with an explicit 3.12 target, falling back
REM to whichever "python" is on PATH.
where py >nul 2>nul
if %errorlevel%==0 (
    set "BOOT_PY=py -3.12"
) else (
    set "BOOT_PY=python"
)

echo [DataWise AI] Creating .venv with %BOOT_PY% ...
%BOOT_PY% -m venv .venv
if errorlevel 1 (
    echo Failed to create .venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo [DataWise AI] Upgrading pip / wheel / setuptools ...
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools

echo [DataWise AI] Installing requirements ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed. See output above.
    pause
    exit /b 1
)

echo.
echo [DataWise AI] Setup complete. Launch with run.bat.
endlocal
