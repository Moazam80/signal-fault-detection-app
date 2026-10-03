import os, sys, time, threading, webbrowser
from pathlib import Path
import uvicorn
from main import app

PORT = 8000
URL = f"http://localhost:{PORT}/app"

if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)

def _open_browser():
    time.sleep(2.0)
    webbrowser.open(URL)

def main():
    print(f"Signal Fault Diagnostic System running at {URL}")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

if __name__ == "__main__":
    main()
