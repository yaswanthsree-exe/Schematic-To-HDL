@echo off
color 0B
echo =======================================================
echo     ⚡ Schematic to Netlist AI - CLI Tool ⚡
echo =======================================================
echo.

if "%~1"=="" (
    echo [ERROR] No input image provided.
    echo.
    echo Usage: launch_cli.bat ^<path_to_schematic_image.png^>
    echo.
    echo Example: 
    echo   launch_cli.bat test.png
    echo   launch_cli.bat 20-combinational_circuit.png
    echo.
    pause
    exit /b
)

echo [INFO] Running AI Pipeline on: %~1
echo -------------------------------------------------------
python predict.py "%~1"
echo -------------------------------------------------------
echo [INFO] Execution finished. Check the console output above and look for the "_result.png" image.
pause
