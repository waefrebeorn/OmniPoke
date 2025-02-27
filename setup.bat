@echo off
echo [Setup] Creating virtual environment...
python -m venv venv

echo [Setup] Activating virtual environment...
call venv\Scripts\activate

echo [Setup] Upgrading pip...
python -m pip install --upgrade pip

echo [Setup] Installing dependencies...
pip install -r requirements.txt

echo [Setup] Checking Git installation...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [Setup] Git not found! Installing Git...
    winget install --id Git.Git -e --source winget
) else (
    echo [Setup] Git is already installed.
)

echo [Setup] Checking Hugging Face CLI...
where huggingface-cli >nul 2>nul
if %errorlevel% neq 0 (
    echo [Setup] Installing Hugging Face CLI...
    pip install huggingface_hub
) else (
    echo [Setup] Hugging Face CLI is already installed.
)

echo [Setup] Login to Hugging Face (required for model access)...
huggingface-cli login

echo [Setup] Setup complete! Run `run.bat` to start the AI.
pause
