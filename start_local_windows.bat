@echo off
setlocal
cd /d "%~dp0"
title LawPak Local Server

echo [LawPak] Project root: %CD%

where python >nul 2>nul
if errorlevel 1 (
  echo [LawPak] Python is not available in PATH.
  echo [LawPak] Install Python 3.10+ and retry.
  pause
  exit /b 1
)

echo [LawPak] Checking Ollama...
ollama list
if errorlevel 1 (
  echo [LawPak] Warning: Ollama CLI is not responding.
  echo [LawPak] Start Ollama first, then rerun this script.
  pause
  exit /b 1
)

echo [LawPak] Starting local server on http://127.0.0.1:5001
echo [LawPak] Press Ctrl+C in this window to stop the server.
python Web\api.py
set EXIT_CODE=%ERRORLEVEL%
echo.
echo [LawPak] Server process exited with code %EXIT_CODE%.
echo [LawPak] If you saw the command prompt come back, the backend is no longer running.
pause
exit /b %EXIT_CODE%
