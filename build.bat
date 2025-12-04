@echo off
echo Installing Nuitka...
pip install nuitka

echo Building Touchless Lock System...
python build_nuitka.py

echo.
echo Build complete! Check the 'dist' directory.
pause






