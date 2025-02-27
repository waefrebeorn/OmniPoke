@echo off
cd /d "%~dp0"
echo [INFO] Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate the virtual environment. Ensure it exists.
    exit /b 1
)
echo [INFO] Virtual environment activated.
cmd
