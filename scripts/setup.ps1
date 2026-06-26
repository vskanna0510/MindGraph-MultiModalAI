# MindGraph++ Setup Script (Windows PowerShell)
# Usage: .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

Write-Host "==> MindGraph++ Setup (Windows)"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.11+ is required"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example"
}

$dirs = @("backend\logs", "assets\uploads", "ml_pipeline\weights", "datasets\raw")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

if ((Get-Command flutter -ErrorAction SilentlyContinue) -and (Test-Path "flutter_app")) {
    Set-Location flutter_app
    flutter pub get
    Set-Location $RootDir
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose up -d
} else {
    Write-Warning "Docker not found. Install Docker Desktop and run: docker compose up -d"
}

$env:PYTHONPATH = "backend"
python -c "from app.config.settings import get_settings; print('Settings OK')"

Set-Location backend
..\.venv\Scripts\pytest.exe tests\unit\test_health.py -q
Set-Location $RootDir

Write-Host "==> Setup complete. Run: make backend (or uvicorn from backend/)"
