@echo off
setlocal

:: Set the .whl path
set "WHL=C:\Projects\OmniPoke\flash_attn-2.7.4+cu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"



:: Pause for debugging
echo [INFO] FlashAttention2 Installation Script Started.
pause

:: Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment detected.
    call venv\Scripts\activate
    echo [INFO] Virtual environment activated.
) else (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)
pause

:: Check if the .whl file exists
if not exist "%WHL%" (
    echo [ERROR] FlashAttention2 .whl file not found at %WHL%
    echo [INFO] Please manually download it and place it there.
    pause
    exit /b 1
)
pause

:: Install FlashAttention2
echo [INFO] Installing FlashAttention2 from "%WHL%"...
pip install "%WHL%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install FlashAttention2.
    pause
    exit /b 1
)

echo [INFO] FlashAttention2 installed successfully.
pause
