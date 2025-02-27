@echo off
echo [Run] Activating virtual environment...
call venv\Scripts\activate

echo [Run] Starting Pokémon Blue AI...
python src\main.py

echo [Run] AI process finished.
pause
