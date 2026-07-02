@echo off
color 0A
echo =======================================================
echo     ⚡ Schematic to Netlist AI - UI Launcher ⚡
echo =======================================================
echo.
echo Checking for required Python packages...
pip install streamlit opencv-python ultralytics Pillow numpy >nul 2>&1
echo.
echo Starting the Streamlit Web UI...
echo (A browser window should open automatically in a moment)
python -m streamlit run app.py
pause
