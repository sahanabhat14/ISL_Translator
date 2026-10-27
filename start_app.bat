@echo off
cd /d "%~dp0"
if not exist ".venv-train\Scripts\python.exe" (
    echo The training environment was not found in this project folder.
    echo Follow the setup instructions in README.md first.
    pause
    exit /b 1
)
".venv-train\Scripts\python.exe" -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2); assert r.status==200" >nul 2>&1
if not errorlevel 1 (
    echo ISL Translator is already running. Opening it in your browser.
    start "" "http://127.0.0.1:8501"
    exit /b 0
)
echo Starting ISL Translator...
echo Keep this window open while using the app. Press Ctrl+C here to stop it.
echo.
".venv-train\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none
if errorlevel 1 (
    echo.
    echo The app could not start. Copy the error above when asking for help.
    pause
)
