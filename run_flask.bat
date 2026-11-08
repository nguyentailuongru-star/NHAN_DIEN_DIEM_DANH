@echo off
chcp 65001 >nul
echo ============================================================
echo    KHOI DONG - Flask Backend
echo ============================================================

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
	set "PY_EXE=python"
)

cd /d "%~dp0flask_app"
"%PY_EXE%" app.py
pause
