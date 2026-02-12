#!/usr/bin/env bash
# =============================================================================
# DataVision Platform — Start Backend (Local Development)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "========================================="
echo "  DataVision Platform — Backend Setup"
echo "========================================="

cd "$BACKEND_DIR"

# --- Create virtual environment if it doesn't exist ---
if [ ! -d "venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

# --- Activate virtual environment ---
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# --- Install dependencies ---
echo "[3/4] Installing dependencies (this may take a while on first run)..."
pip install -r requirements-local.txt --quiet

# --- Create data directories ---
mkdir -p data/uploads data/models data/exports data/faiss_indices

# --- Start the server ---
echo "[4/4] Starting FastAPI server on http://localhost:8000 ..."
echo ""
echo "  API docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/health"
echo ""
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
