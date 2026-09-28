@echo off
REM AirControl Windows Launcher
cd /d "%~dp0"
title AirControl V1
echo ======================================================
echo Launching AirControl V1...
echo ======================================================
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)
.venv\Scripts\python.exe main.py 2>nul
if errorlevel 1 (
    echo.
    echo [AirControl has exited]
    pause
)
