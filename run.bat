@echo off
REM AirControl V2 Windows Launcher (100% Offline Two-Hand Gesture Control)
cd /d "%~dp0"
title AirControl V2 - Gesture Control
echo ======================================================
echo   Launching AirControl V2 (Two-Hand Gesture Control)
echo ======================================================
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)
.venv\Scripts\python.exe main.py 2>nul
if errorlevel 1 (
    echo.
    echo [AirControl V2 has exited]
    pause
)
