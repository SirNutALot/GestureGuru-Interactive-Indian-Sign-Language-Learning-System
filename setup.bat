@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  GestureGuru - First-time Setup
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Install Python 3.10 or 3.11 from https://www.python.org/downloads/
  echo Make sure "Add python.exe to PATH" is checked, then re-run setup.bat.
  pause
  exit /b 1
)

echo Using:
python --version
echo.

if not exist "models\model.h5" (
  echo WARNING: models\model.h5 is missing.
  echo ISL Detection / Practice / Keyboard Mapper need this file.
  echo Place your trained model at models\model.h5 before using those modes.
  echo.
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
  )
) else (
  echo Virtual environment already exists.
)

echo.
echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed.
  pause
  exit /b 1
)

echo.
echo Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Dependency install failed.
  echo Tip: Use Python 3.10 or 3.11. TensorFlow 2.11 may not support newer Python.
  pause
  exit /b 1
)

echo.
echo ============================================
echo  Setup complete.
echo  Next: double-click run.bat to open the app.
echo ============================================
pause
endlocal
