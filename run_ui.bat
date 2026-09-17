@echo off
echo ==========================================
echo   ML Performance Predictor - Starting UI
echo ==========================================
echo.

REM Check if ui_app.py exists
if not exist "ui_app.py" (
    echo ERROR: ui_app.py not found in current directory
    echo Please make sure you are in the correct folder
    pause
    exit /b 1
)

REM Check if complete_predictor.py exists
if not exist "complete_predictor.py" (
    echo ERROR: complete_predictor.py not found
    echo Please ensure the predictor module is in the same folder
    pause
    exit /b 1
)

echo Starting application...
echo.
echo Please wait while models are being trained...
echo This may take 10-30 seconds on first run.
echo.

python ui_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo.
    echo Common issues:
    echo   1. Missing dependencies - run setup_ui.bat
    echo   2. Python not installed - install Python 3.8+
    echo   3. Wrong directory - navigate to project folder
    echo.
    pause
)

exit /b 0