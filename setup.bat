@echo off
chcp 65001 >nul
echo ============================================================
echo    SETUP - Hệ thống Nhận diện Điểm danh AI
echo ============================================================
echo.

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%PY_EXE%" (
    echo [INFO] Su dung Python trong .venv: %PY_EXE%
) else (
    set "PY_EXE=python"
    echo [INFO] Khong tim thay .venv, su dung Python he thong.
)

:: Kiểm tra Python
"%PY_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Chua cai Python! Tai tai: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Cai thu vien AI Service...
echo -----------------------------------------------------------
cd /d "%~dp0ai_service"
"%PY_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [LOI] Cai thu vien AI Service that bai.
    echo Goi y:
    echo   - Neu may chua cai CUDA/cuDNN: mo ai_service\requirements.txt va thay onnxruntime-gpu bang onnxruntime
    echo   - Cai Visual C++ Redistributable (x64)
    pause
    exit /b 1
)

echo.
echo [2/4] Cai thu vien Flask Backend...
echo -----------------------------------------------------------
cd /d "%~dp0flask_app"
"%PY_EXE%" -m pip install -r requirements.txt

echo.
echo [3/4] Cai model AI vao may (%%LOCALAPPDATA%%\NhanDienDiemDanh\models)...
echo -----------------------------------------------------------
cd /d "%~dp0ai_service"
"%PY_EXE%" download_model.py

echo.
echo [4/4] Tao thu muc can thiet...
echo -----------------------------------------------------------
if not exist "%~dp0ai_service\captures" mkdir "%~dp0ai_service\captures"
if not exist "%~dp0flask_app\uploads" mkdir "%~dp0flask_app\uploads"
if not exist "%~dp0flask_app\uploads\avatars" mkdir "%~dp0flask_app\uploads\avatars"

echo.
echo ============================================================
echo    SETUP HOAN TAT!
echo ============================================================
echo.
echo Buoc tiep theo:
echo   1. Database SQLite tu tao tai database\diem_danh.db
echo      (Chuyen du lieu MySQL cu: cd flask_app ^&^& python migrate_to_sqlite.py)
echo.
echo   2. Khong can cai MySQL.
echo.
echo   3. Dang ky khuon mat:
echo      cd ai_service
echo      python register_face.py "Nguyen Van A" NV001 "Phong IT"
echo.
echo   4. Chay he thong: bam dup KHOI_DONG.bat
echo.
pause
