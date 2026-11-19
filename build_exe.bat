@echo off
rem ============================================================
rem  Build ForexBot.exe on this PC (Windows, Python 3.10-3.12)
rem  Result: dist\ForexBot\ForexBot.exe  and  dist\ForexBot-<version>-windows-x64.zip
rem  Takes 5-15 minutes the first time (downloads the packages).
rem ============================================================
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

if not exist .build-venv (
    echo Creating a clean build environment in .build-venv ...
    %PY% -m venv .build-venv || goto :fail
)
call .build-venv\Scripts\activate.bat || goto :fail

echo Installing the bot's packages and PyInstaller ...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt pyinstaller || goto :fail

echo Building ForexBot.exe ...
python build_app.py || goto :fail

echo.
echo Done. Open the dist folder: ForexBot\ (the app) and the .zip to share.
explorer dist
exit /b 0

:fail
echo.
echo Build FAILED - see the messages above.
pause
exit /b 1
