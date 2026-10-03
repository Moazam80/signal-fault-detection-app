@echo off
echo Starting Signal Fault Diagnostic System...
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
start "" http://localhost:8000/app
uvicorn main:app --host 0.0.0.0 --port 8000
pause
