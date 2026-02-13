# =============================================================================
# DataVision Platform - Run Locally (Windows PowerShell)
# =============================================================================
#
# Starts both backend and frontend in parallel as background jobs.
# Press Ctrl+C then run the cleanup to stop both.
#
# Prerequisites:
#   - Conda (Miniconda / Anaconda)
#   - Python 3.10+ (installed via Conda)
#   - Node.js 18+ (installed via Conda or manually)
#
# Usage:
#   .\run_local.ps1
#
# Or run separately in two PowerShell windows:
#   .\scripts\start_backend.ps1   (Window 1)
#   .\scripts\start_frontend.ps1  (Window 2)
# =============================================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DataVision Platform - Local Development"    -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend  (FastAPI) -> http://localhost:8000"  -ForegroundColor White
Write-Host "Frontend (React)   -> http://localhost:5173"  -ForegroundColor White
Write-Host ""

# --- Start backend as a new PowerShell process ---
Write-Host "Launching backend (API)..." -ForegroundColor Yellow
$BackendProc = Start-Process powershell -ArgumentList "-NoExit", "-File", "$ScriptDir\scripts\start_backend.ps1" -PassThru

# --- Start worker as a new PowerShell process ---
Write-Host "Launching worker (AI Engine)..." -ForegroundColor Yellow
$WorkerProc = Start-Process powershell -ArgumentList "-NoExit", "-File", "$ScriptDir\scripts\start_worker.ps1" -PassThru

# Give processes a head start
Start-Sleep -Seconds 5

# --- Start frontend as a new PowerShell process ---
Write-Host "Launching frontend (UI)..." -ForegroundColor Yellow
$FrontendProc = Start-Process powershell -ArgumentList "-NoExit", "-File", "$ScriptDir\scripts\start_frontend.ps1" -PassThru

Write-Host ""
Write-Host "All servers are running in separate windows." -ForegroundColor Green
Write-Host ""
Write-Host "To stop: close the PowerShell windows, or run:" -ForegroundColor Yellow
Write-Host "  Stop-Process -Id $($BackendProc.Id)" -ForegroundColor Gray
Write-Host "  Stop-Process -Id $($WorkerProc.Id)"  -ForegroundColor Gray
Write-Host "  Stop-Process -Id $($FrontendProc.Id)" -ForegroundColor Gray
Write-Host ""

# Wait for user to press a key to shut down
Write-Host "Press Enter here to stop all servers..." -ForegroundColor Yellow
Read-Host

Write-Host "Shutting down..." -ForegroundColor Red
try { Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id $WorkerProc.Id -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue } catch {}
Write-Host "Done." -ForegroundColor Green
