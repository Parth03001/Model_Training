# =============================================================================
# DataVision Platform - Start Backend (Windows PowerShell + Conda)
# =============================================================================
# Usage: .\scripts\start_backend.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectRoot "backend"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  DataVision Platform - Backend Setup"     -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $BackendDir

# --- Create conda environment if it doesn't exist ---
$EnvName = "datavision"
$EnvExists = conda env list | Select-String -Pattern "^$EnvName\s"

if (-not $EnvExists) {
    Write-Host "[1/5] Creating Conda environment '$EnvName' with Python 3.11..." -ForegroundColor Yellow
    conda create -n $EnvName python=3.11 -y
} else {
    Write-Host "[1/5] Conda environment '$EnvName' already exists." -ForegroundColor Green
}

# --- Activate conda environment ---
Write-Host "[2/5] Activating Conda environment '$EnvName'..." -ForegroundColor Yellow
conda activate $EnvName

# --- Install Python dependencies ---
Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements-local.txt --quiet

# --- Install Node.js via conda if not available ---
$NodeCheck = $null
try { $NodeCheck = Get-Command node -ErrorAction SilentlyContinue } catch {}

if (-not $NodeCheck) {
    Write-Host "[4/5] Installing Node.js via Conda..." -ForegroundColor Yellow
    conda install -c conda-forge nodejs=20 -y
} else {
    $NodeVersion = node --version
    Write-Host "[4/5] Node.js already installed ($NodeVersion)." -ForegroundColor Green
}

# --- Create data directories ---
$DataDirs = @("data\uploads", "data\models", "data\exports", "data\faiss_indices")
foreach ($dir in $DataDirs) {
    $fullPath = Join-Path $BackendDir $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}
Write-Host "[5/5] Data directories ready." -ForegroundColor Green

# --- Start the server ---
Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Cyan
Write-Host "  API:    http://localhost:8000" -ForegroundColor White
Write-Host "  Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health: http://localhost:8000/health" -ForegroundColor White
Write-Host ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
