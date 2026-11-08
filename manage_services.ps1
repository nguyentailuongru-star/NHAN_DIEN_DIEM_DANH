param(
  [Parameter(Position=0)]
  [ValidateSet('start','stop','restart','status','menu','help')]
  [string]$Command = 'menu',

  [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Get-Root {
  $inv = (Get-Item -LiteralPath $PSCommandPath)
  return $inv.Directory.FullName
}

function Get-PythonExe([string]$root) {
  $venvPy = Join-Path $root '.venv\Scripts\python.exe'
  if (Test-Path -LiteralPath $venvPy) { return $venvPy }
  return 'python'
}

function Ensure-Dir([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path | Out-Null
  }
}

function Read-Pid([string]$pidFile) {
  if (-not (Test-Path -LiteralPath $pidFile)) { return $null }
  try {
    $txt = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($txt)) { return $null }
    return [int]$txt
  } catch { return $null }
}

function Write-Pid([string]$pidFile, [int]$processId) {
  Set-Content -LiteralPath $pidFile -Value $processId -Encoding ASCII
}

function Is-Running([int]$processId) {
  if (-not $processId) { return $false }
  try {
    $p = Get-Process -Id $processId -ErrorAction Stop
    return ($null -ne $p)
  } catch {
    return $false
  }
}

function Start-ManagedProcess(
  [string]$name,
  [string]$workDir,
  [string]$pyExe,
  [string[]]$argList,
  [string]$pidFile,
  [string]$logOut,
  [string]$logErr
) {
  $existingPid = Read-Pid $pidFile
  if ($existingPid -and (Is-Running $existingPid)) {
    Write-Host "$name dang chay (PID=$existingPid)" -ForegroundColor Yellow
    return
  }

  Ensure-Dir (Split-Path -Parent $pidFile)
  Ensure-Dir (Split-Path -Parent $logOut)

  $env:PYTHONUNBUFFERED = '1'

  $p = Start-Process -FilePath $pyExe -WorkingDirectory $workDir -ArgumentList $argList -PassThru -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr
  Write-Pid $pidFile $p.Id
  Write-Host "$name da khoi dong (PID=$($p.Id))" -ForegroundColor Green
}

function Stop-ByPidFile([string]$name, [string]$pidFile) {
  $procId = Read-Pid $pidFile
  if (-not $procId) {
    Write-Host "${name}: khong co PID file" -ForegroundColor DarkYellow
    return $false
  }
  if (-not (Is-Running $procId)) {
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    Write-Host "${name}: PID $procId khong con chay" -ForegroundColor DarkYellow
    return $true
  }

  try {
    Stop-Process -Id $procId -Force
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    Write-Host "$name da dung (PID=$procId)" -ForegroundColor Green
    return $true
  } catch {
    Write-Host "${name}: dung PID=$procId that bai: $($_.Exception.Message)" -ForegroundColor Red
    return $false
  }
}

function Stop-ByCommandLineMatch([string]$label, [string[]]$patterns, [string]$exeMustContain = $null) {
  $killed = 0
  $procs = Get-CimInstance Win32_Process | Where-Object {
    $cl = $_.CommandLine
    if (-not $cl) { return $false }
    foreach ($pat in $patterns) {
      if ($cl -like $pat) {
        if ($exeMustContain) {
          $ep = $_.ExecutablePath
          if (-not $ep) { return $false }
          return ($ep -like "*$exeMustContain*")
        }
        return $true
      }
    }
    return $false
  }

  foreach ($proc in $procs) {
    try {
      $null = Invoke-CimMethod -InputObject $proc -MethodName Terminate
      $killed++
    } catch {
      # ignore
    }
  }

  if ($killed -gt 0) {
    Write-Host "${label}: da dung $killed tien trinh cu" -ForegroundColor Green
  } else {
    Write-Host "${label}: khong thay tien trinh cu" -ForegroundColor DarkYellow
  }
}

function Show-Status([string]$name, [string]$pidFile) {
  $procId = Read-Pid $pidFile
  if ($procId -and (Is-Running $procId)) {
    Write-Host "${name}: RUNNING (PID=$procId)" -ForegroundColor Green
  } elseif ($procId) {
    Write-Host "${name}: STOPPED (PID file=$procId, process not running)" -ForegroundColor DarkYellow
  } else {
    Write-Host "${name}: STOPPED" -ForegroundColor DarkYellow
  }
}

$root = Get-Root
$py = Get-PythonExe $root

$pidDir = Join-Path $root '.pids'
$logDir = Join-Path $root 'logs'
Ensure-Dir $pidDir
Ensure-Dir $logDir

$flaskPid = Join-Path $pidDir 'flask.pid'
$aiPid = Join-Path $pidDir 'ai_service.pid'

$flaskWorkDir = Join-Path $root 'flask_app'
$aiWorkDir = Join-Path $root 'ai_service'

$flaskOut = Join-Path $logDir 'flask.out.log'
$flaskErr = Join-Path $logDir 'flask.err.log'
$aiOut = Join-Path $logDir 'ai_service.out.log'
$aiErr = Join-Path $logDir 'ai_service.err.log'

function Do-Start {
  Start-ManagedProcess -name 'Flask' -workDir $flaskWorkDir -pyExe $py -argList @('app.py') -pidFile $flaskPid -logOut $flaskOut -logErr $flaskErr
  Start-ManagedProcess -name 'AI Service' -workDir $aiWorkDir -pyExe $py -argList @('main.py') -pidFile $aiPid -logOut $aiOut -logErr $aiErr
}

function Do-Stop([switch]$force) {
  $ok1 = Stop-ByPidFile -name 'AI Service' -pidFile $aiPid
  $ok2 = Stop-ByPidFile -name 'Flask' -pidFile $flaskPid

  if ($force) {
    # Kill any leftover python processes that run these entrypoints
    $rootNeedle = (Split-Path -Path $root -Leaf)
    # Most python processes show only 'app.py' / 'main.py' in CommandLine.
    Stop-ByCommandLineMatch -label 'Force-stop Flask' -patterns @('* app.py*', '*\\app.py*') -exeMustContain $rootNeedle
    Stop-ByCommandLineMatch -label 'Force-stop AI Service' -patterns @('* main.py*', '*\\main.py*') -exeMustContain $rootNeedle
  }
}

function Do-Status {
  Show-Status -name 'Flask' -pidFile $flaskPid
  Show-Status -name 'AI Service' -pidFile $aiPid
  Write-Host "Logs: $logDir" -ForegroundColor Cyan
}

function Do-Menu {
  while ($true) {
    Clear-Host
    Write-Host '==============================='
    Write-Host '  QUAN LY HE THONG DIEM DANH'
    Write-Host '==============================='
    Do-Status
    Write-Host ''
    Write-Host '1) Start (Flask + AI Service)'
    Write-Host '2) Stop'
    Write-Host '3) Restart'
    Write-Host '4) Status (refresh)'
    Write-Host '5) Stop --force (diet tien trinh cu)'
    Write-Host '0) Thoat (tu dong Stop)'
    $choice = Read-Host 'Chon'

    switch ($choice) {
      '1' { Do-Start }
      '2' { Do-Stop }
      '3' { Do-Stop; Start-Sleep -Milliseconds 300; Do-Start }
      '4' { }
      '5' { Do-Stop -force }
      '0' {
        Do-Stop
        break
      }
      default {
        Write-Host 'Lua chon khong hop le.' -ForegroundColor Yellow
      }
    }

    if ($choice -ne '0') {
      Write-Host ''
      Read-Host 'Nhan Enter de tiep tuc'
    }
  }
}

function Do-Help {
  Write-Host 'Dung:'
  Write-Host '  manage.bat start'
  Write-Host '  manage.bat stop'
  Write-Host '  manage.bat stop --force'
  Write-Host '  manage.bat restart'
  Write-Host '  manage.bat status'
}

# Accept bat-style --force
if ($args -contains '--force') { $Force = $true }

switch ($Command) {
  'start'   { Do-Start }
  'stop'    { Do-Stop -force:$Force }
  'restart' { Do-Stop -force:$Force; Start-Sleep -Milliseconds 300; Do-Start }
  'status'  { Do-Status }
  'help'    { Do-Help }
  default   { Do-Menu }
}
