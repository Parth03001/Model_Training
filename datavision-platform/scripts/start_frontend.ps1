# =============================================================================
# DataVision Platform — Start Frontend (Windows PowerShell + Conda)
# =============================================================================
# Usage: .\scripts\start_frontend.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $ProjectRoot "frontend"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  DataVision Platform — Frontend Setup"   -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $FrontendDir

# --- Ensure Node.js is available (install via conda if needed) ---
$NodeCheck = $null
try { $NodeCheck = Get-Command node -ErrorAction SilentlyContinue } catch {}

if (-not $NodeCheck) {
    Write-Host "[0/2] Node.js not found. Installing via Conda..." -ForegroundColor Yellow
    conda install -c conda-forge nodejs=20 -y
}

$NodeVersion = node --version
$NpmVersion = npm --version
Write-Host "Using Node.js $NodeVersion, npm $NpmVersion" -ForegroundColor Green
Write-Host ""

# --- Install dependencies ---
if (-not (Test-Path "node_modules")) {
    Write-Host "[1/2] Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
} else {
    Write-Host "[1/2] Node modules already installed. Run 'npm install' manually to update." -ForegroundColor Green
}

# --- Start dev server ---
Write-Host "[2/2] Starting Vite dev server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  App: http://localhost:5173" -ForegroundColor White
Write-Host ""

npm run dev
