@echo off
chcp 65001 >nul
setlocal
title He thong nhan dien diem danh

rem Bam dup file nay: chay Flask + AI Service, tu mo dashboard. Nhan Q de dung.
rem Lenh khac: manage.bat menu ^| start ^| stop [--force] ^| restart ^| status

set "ROOT=%~dp0"
set "PS1=%ROOT%manage_services.ps1"

if not exist "%PS1%" (
  echo Khong tim thay %PS1%
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
if errorlevel 1 pause
