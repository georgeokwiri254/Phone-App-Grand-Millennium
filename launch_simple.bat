@echo off
echo Setting up Grand Millennium Revenue Analytics Dashboard...

REM Set UTF-8 encoding environment variables
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Change console code page to UTF-8
chcp 65001 > nul

REM Change to project directory
cd /d "%~dp0"

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import streamlit, pandas, plotly" > nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install -r requirements.txt
)

REM Find available port
set PORT=8501
netstat -ano | findstr :8501 > nul
if not errorlevel 1 (
    echo Port 8501 is busy, trying 8502...
    set PORT=8502
)

REM Launch Streamlit directly
echo Starting Dashboard on port %PORT%...
echo Browser will open automatically at http://localhost:%PORT%
echo Press Ctrl+C to stop the dashboard

start "" "http://localhost:%PORT%"
python -m streamlit run app/streamlit_app.py --server.port=%PORT% --server.headless=false --browser.gatherUsageStats=false

pause