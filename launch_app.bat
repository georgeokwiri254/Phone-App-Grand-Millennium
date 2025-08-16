@echo off
cd /d "%~dp0"
echo Starting Grand Millennium Revenue Analytics...
echo.
python -m streamlit run app/streamlit_app_simple.py --server.port 8511
pause
