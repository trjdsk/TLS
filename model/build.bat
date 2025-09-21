@echo off
setlocal enabledelayedexpansion

REM === CONFIG ===
set BUILD_DIR=build
set GENERATOR="Visual Studio 17 2022"  REM Or use "MinGW Makefiles" if you have MinGW
set CONFIG=Release

REM === CLEAN BUILD ===
if exist %BUILD_DIR% (
    echo Removing old build directory...
    rmdir /s /q %BUILD_DIR%
)

mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM === RUN CMAKE ===
echo Configuring project with CMake...
cmake .. -G %GENERATOR%

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

echo Building Edge Impulse model library...
cmake --build . --config %CONFIG% -- /maxcpucount

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo ✅ Build finished. DLL is in %BUILD_DIR%\%CONFIG%\ or %BUILD_DIR%\lib\
cd ..
endlocal
