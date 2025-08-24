@echo off
title Grand Millennium Revenue Analytics - Enhanced Launcher
cd /d "%~dp0"

echo ====================================================================
echo  🏨 Grand Millennium Revenue Analytics - Enhanced Launcher
echo ====================================================================
echo.
echo  🚀 Logo-enabled launcher available: Grand_Millennium_Analytics_Launcher.exe
echo  📊 This launcher includes automatic port conflict resolution:
echo  - Detects if port 8511 is in use
echo  - Offers to kill the conflicting process
echo  - Can automatically find the next available port
echo.
echo ====================================================================
echo.

REM Try the enhanced launcher first
echo Launching with port management...
python launch_with_port_management.py

REM If that fails, fall back to basic launcher
if errorlevel 1 (
    echo.
    echo ====================================================================
    echo  Enhanced launcher failed - using fallback method
    echo ====================================================================
    echo.
    python -m streamlit run app/streamlit_app_simple.py --server.port 8511
)

echo.
echo ====================================================================
echo  Server has stopped
echo ====================================================================
pause