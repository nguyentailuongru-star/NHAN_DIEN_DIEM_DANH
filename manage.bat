@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PS1=%ROOT%manage_services.ps1"

if not exist "%PS1%" (
  echo Khong tim thay %PS1%
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
