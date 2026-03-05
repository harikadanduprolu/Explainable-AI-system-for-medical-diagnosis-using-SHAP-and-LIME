@echo off
REM ==============================================================================
REM Start Explainable Medical AI Web Application (Windows)
REM ==============================================================================

echo.
echo ========================================================================
echo  Explainable Medical AI - Web Application Launcher
echo ========================================================================
echo.

REM Check if virtual environment exists
if exist .venv\Scripts\activate.ps1 (
    echo [1/3] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

echo [2/3] Starting FastAPI server...
echo.
echo Once started, open your browser to:
echo    Web App:  http://localhost:8000/app
echo    API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================================================
echo.

REM Start the application
python start_web_application.py

echo.
echo Application stopped.
pause
