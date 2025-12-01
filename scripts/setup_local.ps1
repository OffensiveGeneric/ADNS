#!/usr/bin/env pwsh
# Bootstrap a local ADNS dev environment on Windows/PowerShell.
# Usage: pwsh ./scripts/setup_local.ps1

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

function Test-Command {
  param([string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "==> Using repository root: $root"

if (-not (Test-Command $pythonBin)) {
  if (Test-Command "py") {
    $pythonBin = "py"
  } else {
    Write-Error "Python was not found in PATH. Install Python 3.9+ and retry."
    exit 1
  }
}

if (-not (Test-Command "npm")) {
  Write-Error "npm is required to build the frontend. Install Node.js 18+."
  exit 1
}

$venvPath = Join-Path $root ".venv"
if (-not (Test-Path $venvPath)) {
  Write-Host "==> Creating virtualenv at $venvPath"
  & $pythonBin -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts/python.exe"

Write-Host "==> Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Host "==> Installing Python dependencies (api + agent)"
& $venvPython -m pip install -r (Join-Path $root "api/requirements.txt") -r (Join-Path $root "agent/requirements.txt")

Write-Host "==> Installing frontend dependencies (npm install)"
Push-Location (Join-Path $root "frontend/adns-frontend")
npm install
Pop-Location

if (-not (Test-Path (Join-Path $root ".env")) -and (Test-Path (Join-Path $root ".env.example"))) {
  Write-Host "==> Copying .env.example to .env (edit as needed)"
  Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env")
}

Write-Host "==> Setup complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host " 1) Install tshark (Wireshark installer includes it) and ensure your user can capture packets."
Write-Host " 2) Activate the virtualenv: `& $venvPath\Scripts\Activate.ps1`"
Write-Host " 3) Start services in separate terminals:"
Write-Host "    # API"
Write-Host "    $env:FLASK_APP='app.py'; .\\.venv\\Scripts\\python.exe -m flask --app api/app.py run"
Write-Host "    # Worker (optional if using inline scoring)"
Write-Host "    $env:FLASK_APP='app.py'; .\\.venv\\Scripts\\python.exe api/worker.py"
Write-Host "    # Agent (needs tshark + capture privileges)"
Write-Host "    $env:API_URL='http://127.0.0.1:5000/ingest'; .\\.venv\\Scripts\\python.exe agent/capture.py"
Write-Host "    # Frontend"
Write-Host "    pushd frontend/adns-frontend; npm run dev -- --host; popd"
