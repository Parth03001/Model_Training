# =============================================================================
# DataVision Platform - Start Celery Worker (Windows PowerShell)
# =============================================================================
# This is the "Engine" that runs the AI models (CLIP, SAM2, Grounding DINO).
# Usage: .\scripts\start_worker.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectRoot "backend"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  DataVision Platform - AI Worker"         -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $BackendDir

# --- Activate conda environment ---
$EnvName = "datavision"
Write-Host "Activating Conda environment '$EnvName'..." -ForegroundColor Yellow
# We use 'cmd /c' to ensure conda activation works correctly in scripts
conda activate $EnvName

Write-Host "Starting Celery Worker (GPU/CPU)..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

# --pool=solo is required for Windows compatibility with AI models
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo -Q auto_annotate,training,export
