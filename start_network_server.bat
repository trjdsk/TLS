@echo off
echo Touchless Lock System - Network Server Starter
echo =============================================

echo.
echo 🔍 Auto-detecting local IP address...
echo.

REM Get local IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip

REM Remove leading spaces
set LOCAL_IP=%LOCAL_IP: =%

echo ✅ Detected local IP: %LOCAL_IP%
echo.
echo Choose startup mode:
echo 1. Local mode (localhost only)
echo 2. Network mode (accessible from other devices)
echo 3. Show network configuration
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo Starting in LOCAL mode...
    python launcher.py web
) else if "%choice%"=="2" (
    echo Starting in NETWORK mode...
    echo Server will be accessible at: http://%LOCAL_IP%:8000
    python launcher.py web --network --host %LOCAL_IP%
) else if "%choice%"=="3" (
    echo Running network configuration helper...
    python network_config.py
    pause
) else (
    echo Invalid choice. Please run the script again.
    pause
)

pause
