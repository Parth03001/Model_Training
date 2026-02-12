#!/usr/bin/env bash
# =============================================================================
# DataVision Platform — Run Locally (Backend + Frontend)
# =============================================================================
#
# This script starts both the backend and frontend in parallel.
# Press Ctrl+C to stop both servers.
#
# Prerequisites:
#   - Python 3.10+
#   - Node.js 18+
#   - npm
#
# Usage:
#   chmod +x run_local.sh
#   ./run_local.sh
#
# Or run them separately:
#   ./scripts/start_backend.sh   (Terminal 1)
#   ./scripts/start_frontend.sh  (Terminal 2)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  DataVision Platform — Local Development"
echo "============================================"
echo ""
echo "Starting backend (FastAPI)  → http://localhost:8000"
echo "Starting frontend (React)   → http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Trap Ctrl+C to kill both background processes
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend in background
bash "$SCRIPT_DIR/scripts/start_backend.sh" &
BACKEND_PID=$!

# Give backend a head start
sleep 3

# Start frontend in background
bash "$SCRIPT_DIR/scripts/start_frontend.sh" &
FRONTEND_PID=$!

# Wait for both
wait $BACKEND_PID $FRONTEND_PID
