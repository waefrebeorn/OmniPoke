@echo off
setlocal enabledelayedexpansion

:: Define the output file
set OUTPUT_FILE=combinedpy.txt

:: Delete the output file if it already exists
if exist "%OUTPUT_FILE%" del "%OUTPUT_FILE%"

:: Loop through all .py files in the current directory
for %%f in (*.py) do (
    echo ======================== >> "%OUTPUT_FILE%"
    echo %%f >> "%OUTPUT_FILE%"
    echo ======================== >> "%OUTPUT_FILE%"
    type "%%f" >> "%OUTPUT_FILE%"
    echo. >> "%OUTPUT_FILE%"
    echo. >> "%OUTPUT_FILE%"
)

echo Combined all .py files into %OUTPUT_FILE%
