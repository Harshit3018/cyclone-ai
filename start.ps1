# ==============================================================
# CYCLONE-AI Development Start Script (Windows)
# ==============================================================
# Usage: .\start.ps1
#   -Mode dev     → Start both backend + frontend dev servers
#   -Mode prod    → Build frontend, start backend serving everything
#   -Mode backend → Start backend only
# ==============================================================

param(
    [ValidateSet("dev", "prod", "backend")]
    [string]$Mode = "dev"
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  ====================================" -ForegroundColor Cyan
Write-Host "  CYCLONE-AI Intelligence Platform" -ForegroundColor Cyan
Write-Host "  ====================================" -ForegroundColor Cyan
Write-Host ""

if ($Mode -eq "dev") {
    Write-Host "[*] Starting in DEVELOPMENT mode..." -ForegroundColor Yellow
    Write-Host "    Backend:  http://localhost:8000" -ForegroundColor Gray
    Write-Host "    Frontend: http://localhost:5173" -ForegroundColor Gray
    Write-Host "    API Docs: http://localhost:8000/docs" -ForegroundColor Gray
    Write-Host ""

    # Start backend in background
    $backend = Start-Process -NoNewWindow -PassThru powershell -ArgumentList "-Command", "cd '$ProjectRoot'; python -m uvicorn backend.app.main:app --reload --port 8000"
    
    # Start frontend
    Push-Location "$ProjectRoot\frontend"
    npm run dev
    Pop-Location

    # Cleanup
    if ($backend -and !$backend.HasExited) { Stop-Process $backend }

} elseif ($Mode -eq "prod") {
    Write-Host "[*] Building for PRODUCTION..." -ForegroundColor Yellow
    
    # Build frontend
    Push-Location "$ProjectRoot\frontend"
    Write-Host "[1/2] Building frontend..." -ForegroundColor Gray
    npm run build
    Pop-Location

    # Start backend with frontend serving
    Write-Host "[2/2] Starting production server..." -ForegroundColor Gray
    Write-Host "    App: http://localhost:8000" -ForegroundColor Gray
    Write-Host ""
    
    $env:SERVE_FRONTEND = "true"
    $env:FRONTEND_DIR = "frontend/dist"
    python "$ProjectRoot\backend\run.py"

} elseif ($Mode -eq "backend") {
    Write-Host "[*] Starting BACKEND only..." -ForegroundColor Yellow
    Write-Host "    API:  http://localhost:8000/api" -ForegroundColor Gray
    Write-Host "    Docs: http://localhost:8000/docs" -ForegroundColor Gray
    Write-Host ""
    
    python -m uvicorn backend.app.main:app --reload --port 8000
}
