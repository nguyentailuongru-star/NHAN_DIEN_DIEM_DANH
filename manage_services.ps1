param(
  [Parameter(Position=0)]
  [ValidateSet('run','start','stop','restart','status','menu','help')]
  [string]$Command = 'run',

  [switch]$Force
)

$ErrorActionPreference = 'Stop'

$FlaskPort = 5000
$FlaskUrl = "http://localhost:$FlaskPort"

function Get-Root {
  $inv = (Get-Item -LiteralPath $PSCommandPath)
  return $inv.Directory.FullName
}

function Get-PythonExe([string]$root) {
  $venvPy = Join-Path $root '.venv\Scripts\python.exe'
  if (Test-Path -LiteralPath $venvPy) { return $venvPy }
  # Bo qua ban "python.exe" gia cua Microsoft Store (WindowsApps)
  $cmd = Get-Command python.exe -All -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -notlike '*\WindowsApps\*' } | Select-Object -First 1
  if ($cmd) { return $cmd.Source }
  throw 'Khong tim thay Python. Hay cai Python 3.11 va tick "Add python.exe to PATH".'
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
  return ($null -ne (Get-Process -Id $processId -ErrorAction SilentlyContinue))
}

function Stop-Tree([int]$processId) {
  # /T: dung ca tien trinh con (Flask debug reloader sinh them 1 tien trinh con giu cong 5000)
  & taskkill.exe /PID $processId /T /F 2>&1 | Out-Null
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
  # Ghi log ra file bang UTF-8, tranh loi UnicodeEncodeError khi in tieng Viet (cp1252)
  $env:PYTHONIOENCODING = 'utf-8'

  $p = Start-Process -FilePath $pyExe -WorkingDirectory $workDir -ArgumentList $argList -PassThru -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr
  Write-Pid $pidFile $p.Id
  Write-Host "$name da khoi dong (PID=$($p.Id))" -ForegroundColor Green
}

function Stop-ByPidFile([string]$name, [string]$pidFile) {
  $procId = Read-Pid $pidFile
  if (-not $procId) {
    Write-Host "${name}: khong chay" -ForegroundColor DarkYellow
    return
  }
  if (Is-Running $procId) {
    Stop-Tree $procId
    Write-Host "$name da dung (PID=$procId)" -ForegroundColor Green
  } else {
    Write-Host "${name}: PID $procId khong con chay" -ForegroundColor DarkYellow
  }
  Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
}

function Stop-Leftovers {
  # Tien trinh python dang giu cong Flask
  $conns = Get-NetTCPConnection -LocalPort $FlaskPort -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -like 'python*') {
      Stop-Tree $p.Id
      Write-Host "Da dung python giu cong $FlaskPort (PID=$($p.Id))" -ForegroundColor Green
    }
  }
  # AI service con sot lai: python ... main.py
  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match '\smain\.py\s*$' }
  foreach ($proc in $procs) {
    Stop-Tree ([int]$proc.ProcessId)
    Write-Host "Da dung AI service cu (PID=$($proc.ProcessId))" -ForegroundColor Green
  }
}

function Show-Status([string]$name, [string]$pidFile) {
  $procId = Read-Pid $pidFile
  if ($procId -and (Is-Running $procId)) {
    Write-Host "${name}: RUNNING (PID=$procId)" -ForegroundColor Green
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
  Stop-ByPidFile -name 'AI Service' -pidFile $aiPid
  Stop-ByPidFile -name 'Flask' -pidFile $flaskPid
  if ($force) { Stop-Leftovers }
}

function Do-Status {
  Show-Status -name 'Flask' -pidFile $flaskPid
  Show-Status -name 'AI Service' -pidFile $aiPid
  Write-Host "Logs: $logDir" -ForegroundColor Cyan
}

function Show-LogTail([string]$path, [int]$lines = 15) {
  if (Test-Path -LiteralPath $path) {
    Get-Content -LiteralPath $path -Tail $lines -Encoding UTF8 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
  }
}

function Wait-FlaskReady([int]$timeoutSec = 60) {
  $deadline = (Get-Date).AddSeconds($timeoutSec)
  while ((Get-Date) -lt $deadline) {
    $flaskProc = Read-Pid $flaskPid
    if (-not (Is-Running $flaskProc)) { return $false }
    try {
      $null = Invoke-WebRequest -Uri "$FlaskUrl/login" -UseBasicParsing -TimeoutSec 2
      return $true
    } catch {
      Start-Sleep -Milliseconds 700
    }
  }
  return $false
}

function Open-Shared([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return $null }
  $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
  return New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
}

# Bam 1 lan: chay Flask + AI Service, mo trinh duyet, hien log AI; nhan Q de dung tat ca.
function Do-Run {
  Clear-Host
  Write-Host '=============================================' -ForegroundColor Cyan
  Write-Host '   HE THONG NHAN DIEN DIEM DANH - KHOI DONG' -ForegroundColor Cyan
  Write-Host '=============================================' -ForegroundColor Cyan
  Write-Host "Python: $py"
  Write-Host ''

  try {
    Do-Start
    Write-Host ''
    Write-Host 'Dang cho Flask san sang...' -NoNewline
    if (Wait-FlaskReady) {
      Write-Host ' OK' -ForegroundColor Green
      Write-Host "Dashboard: $FlaskUrl" -ForegroundColor Green
      Start-Process $FlaskUrl
    } else {
      Write-Host ' LOI' -ForegroundColor Red
      Write-Host "Flask khong khoi dong duoc. Log loi ($flaskErr):" -ForegroundColor Red
      Show-LogTail $flaskErr 25
      Write-Host ''
      Read-Host 'Nhan Enter de dung he thong'
      return
    }

    Write-Host ''
    Write-Host '---------------------------------------------'
    Write-Host ' Log AI Service ben duoi. Nhan  Q  de DUNG he thong.' -ForegroundColor Yellow
    Write-Host ' (Dong cua so bang dau X se KHONG dung service - hay dung phim Q)' -ForegroundColor DarkYellow
    Write-Host '---------------------------------------------'

    $readers = @{}
    while ($true) {
      foreach ($log in @($aiOut, $aiErr)) {
        if (-not $readers.ContainsKey($log)) {
          $r = Open-Shared $log
          if ($r) { $readers[$log] = $r }
        }
        if ($readers.ContainsKey($log)) {
          while ($null -ne ($line = $readers[$log].ReadLine())) { Write-Host $line }
        }
      }

      if ([Console]::KeyAvailable) {
        $key = [Console]::ReadKey($true)
        if ($key.Key -eq 'Q') { break }
      }

      $aiAlive = Is-Running (Read-Pid $aiPid)
      $flaskAlive = Is-Running (Read-Pid $flaskPid)
      if (-not $flaskAlive) {
        Write-Host ''
        Write-Host 'Flask da dung bat ngo. Log loi:' -ForegroundColor Red
        Show-LogTail $flaskErr 25
        Read-Host 'Nhan Enter de dung he thong'
        break
      }
      if (-not $aiAlive -and -not $script:aiDeadShown) {
        $script:aiDeadShown = $true
        Write-Host ''
        Write-Host 'AI Service da dung (xem log o tren). Flask van chay, nhan Q de dung.' -ForegroundColor Red
      }
      Start-Sleep -Milliseconds 400
    }
  } finally {
    foreach ($r in $readers.Values) { $r.Dispose() }
    Write-Host ''
    Write-Host 'Dang dung he thong...' -ForegroundColor Yellow
    Do-Stop
  }
}

function Do-Menu {
  while ($true) {
    Clear-Host
    Write-Host '==============================='
    Write-Host '  QUAN LY HE THONG DIEM DANH'
    Write-Host '==============================='
    Do-Status
    Write-Host ''
    Write-Host '1) Start (Flask + AI Service, chay nen)'
    Write-Host '2) Stop'
    Write-Host '3) Restart'
    Write-Host '4) Status (refresh)'
    Write-Host '5) Stop --force (diet ca tien trinh cu con sot)'
    Write-Host '0) Thoat (tu dong Stop)'
    $choice = Read-Host 'Chon'

    switch ($choice) {
      '1' { Do-Start }
      '2' { Do-Stop }
      '3' { Do-Stop; Start-Sleep -Milliseconds 500; Do-Start }
      '4' { }
      '5' { Do-Stop -force }
      '0' { Do-Stop; return }
      default { Write-Host 'Lua chon khong hop le.' -ForegroundColor Yellow }
    }

    Write-Host ''
    Read-Host 'Nhan Enter de tiep tuc'
  }
}

function Do-Help {
  Write-Host 'Dung:'
  Write-Host '  manage.bat              Bam 1 lan: chay tat ca + mo dashboard (nhan Q de dung)'
  Write-Host '  manage.bat menu         Menu quan ly'
  Write-Host '  manage.bat start        Chay nen Flask + AI Service'
  Write-Host '  manage.bat stop [--force]'
  Write-Host '  manage.bat restart'
  Write-Host '  manage.bat status'
}

# Accept bat-style --force
if ($args -contains '--force') { $Force = $true }

switch ($Command) {
  'run'     { Do-Run }
  'start'   { Do-Start }
  'stop'    { Do-Stop -force:$Force }
  'restart' { Do-Stop -force:$Force; Start-Sleep -Milliseconds 500; Do-Start }
  'status'  { Do-Status }
  'help'    { Do-Help }
  default   { Do-Menu }
}
