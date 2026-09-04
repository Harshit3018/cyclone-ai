#!/bin/bash
# ==============================================================
# CYCLONE-AI Development Start Script (Linux/macOS)
# ==============================================================
# Usage: ./start.sh [dev|prod|backend]
# ==============================================================

set -e

MODE="${1:-dev}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ===================================="
echo "  CYCLONE-AI Intelligence Platform"
echo "  ===================================="
echo ""

case "$MODE" in
  dev)
    echo "[*] Starting in DEVELOPMENT mode..."
    echo "    Backend:  http://localhost:8000"
    echo "    Frontend: http://localhost:5173"
    echo "    API Docs: http://localhost:8000/docs"
    echo ""

    # Start backend in background
    cd "$PROJECT_ROOT"
    python -m uvicorn backend.app.main:app --reload --port 8000 &
    BACKEND_PID=$!

    # Start frontend
    cd "$PROJECT_ROOT/frontend"
    npm run dev

    # Cleanup
    kill $BACKEND_PID 2>/dev/null || true
    ;;

  prod)
    echo "[*] Building for PRODUCTION..."

    # Build frontend
    cd "$PROJECT_ROOT/frontend"
    echo "[1/2] Building frontend..."
    npm run build

    # Start backend with frontend serving
    cd "$PROJECT_ROOT"
    echo "[2/2] Starting production server..."
    echo "    App: http://localhost:8000"
    echo ""

    SERVE_FRONTEND=true FRONTEND_DIR=frontend/dist python backend/run.py
    ;;

  backend)
    echo "[*] Starting BACKEND only..."
    echo "    API:  http://localhost:8000/api"
    echo "    Docs: http://localhost:8000/docs"
    echo ""

    cd "$PROJECT_ROOT"
    python -m uvicorn backend.app.main:app --reload --port 8000
    ;;

  *)
    echo "Usage: $0 [dev|prod|backend]"
    exit 1
    ;;
esac
