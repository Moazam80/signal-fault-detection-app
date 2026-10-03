@echo off
echo ============================================================
echo  Building Signal Fault Diagnostic System EXE
echo ============================================================
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
pip install pyinstaller --quiet
pyinstaller ^
  --onefile ^
  --name "SignalFaultDiagnostic" ^
  --add-data "models;models" ^
  --add-data "..\frontend;frontend" ^
  --hidden-import sklearn ^
  --hidden-import sklearn.neighbors ^
  --hidden-import sklearn.preprocessing ^
  --hidden-import scipy.io ^
  --hidden-import scipy.fft ^
  --hidden-import uvicorn.logging ^
  --hidden-import uvicorn.loops ^
  --hidden-import uvicorn.loops.auto ^
  --hidden-import uvicorn.protocols ^
  --hidden-import uvicorn.protocols.http ^
  --hidden-import uvicorn.protocols.http.auto ^
  --hidden-import uvicorn.lifespan ^
  --hidden-import uvicorn.lifespan.on ^
  launcher.py
echo.
echo Done! EXE at: backend\dist\SignalFaultDiagnostic.exe
pause
