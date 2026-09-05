@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run setup.bat once before using run.bat.
  pause
  exit /b 1
)

if not exist "launcher.py" (
  echo ERROR: launcher.py not found in %cd%
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "launcher.py"
if errorlevel 1 (
  echo.
  echo Launcher exited with an error.
  pause
  exit /b 1
)

endlocal
