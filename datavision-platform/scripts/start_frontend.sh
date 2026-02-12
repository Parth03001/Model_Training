#!/usr/bin/env bash
# =============================================================================
# DataVision Platform — Start Frontend (Local Development)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "========================================="
echo "  DataVision Platform — Frontend Setup"
echo "========================================="

cd "$FRONTEND_DIR"

# --- Install dependencies ---
if [ ! -d "node_modules" ]; then
    echo "[1/2] Installing Node.js dependencies..."
    npm install
else
    echo "[1/2] Node modules already installed. Run 'npm install' manually to update."
fi

# --- Start dev server ---
echo "[2/2] Starting Vite dev server on http://localhost:5173 ..."
echo ""
echo "  App:  http://localhost:5173"
echo ""
npm run dev
