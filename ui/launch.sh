#!/bin/bash
# Launch LeRobot UI — backend (FastAPI) + frontend (Svelte/Vite)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Python dependencies ----
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# ---- Node dependencies ----
if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi

# ---- Start backend ----
echo "Starting backend server on :8000 ..."
python server.py &
BACKEND_PID=$!

# Give the backend a moment to bind its port
sleep 1

# ---- Start frontend ----
echo "Starting frontend dev server on :5173 ..."
npm run dev &
FRONTEND_PID=$!

# ---- Cleanup on exit ----
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$BACKEND_PID"  2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "================================================"
echo "  LeRobot UI   →  http://localhost:5173"
echo "  Backend API  →  http://localhost:8000"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""

wait
