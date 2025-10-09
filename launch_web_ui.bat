@echo off
setlocal
echo Starting Touchless Lock System Web Server with UI...
start "" "http://localhost:8000/ui"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
endlocal
