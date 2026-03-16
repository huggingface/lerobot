#!/bin/bash
# =============================================================================
# Async Inference Startup Script
# =============================================================================
#
# Starts the async inference components in the correct order:
# 1. Policy server (includes trajectory visualization via HTTP/WebSocket)
# 2. Robot client (connects to policy server)
#
# The trajectory visualization runs inside the policy server, so you can
# view it at http://localhost:8088 once the policy server starts.
#
# Usage:
#   ./scripts/start_async_inference.sh                  # Normal mode
#
# Environment variables:
#   POLICY_SERVER_DELAY_S   - Seconds to wait after starting policy server (default: 3)
#   LEROBOT_DEBUG           - Set to 1 for debug logging
#
# =============================================================================

set -e

# Optional debug tracing for this script
if [ "${LEROBOT_DEBUG:-0}" = "1" ]; then
    set -x
fi

# Configuration
POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/policy_server.log"

# PIDs for cleanup
POLICY_SERVER_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down async inference components..."

    if [ -n "$POLICY_SERVER_PID" ] && kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
        echo "Stopping policy server (PID: $POLICY_SERVER_PID)..."
        kill -TERM "$POLICY_SERVER_PID" 2>/dev/null || true
        wait "$POLICY_SERVER_PID" 2>/dev/null || true
    fi

    echo "Cleanup complete."
    exit 0
}

# Register signal handlers
trap cleanup SIGINT SIGTERM

# Change to project root
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Async Inference Startup Script"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Start Policy Server (includes trajectory visualization)
# -----------------------------------------------------------------------------
echo "[1/2] Starting policy server..."
# Ensure log directory exists and capture server output for debugging.
mkdir -p "$LOG_DIR"
echo "      Policy server logs: $LOG_FILE"
# Use --no-sync to skip dependency resolution (avoids grpcio version conflicts)
uv run --no-sync python examples/tutorial/async-inf/policy_server_drtc.py >"$LOG_FILE" 2>&1 &
POLICY_SERVER_PID=$!
echo "      Policy server started (PID: $POLICY_SERVER_PID)"
echo "      Trajectory visualization: http://localhost:8088"
echo "      Waiting ${POLICY_SERVER_DELAY_S}s for server to initialize..."
sleep "$POLICY_SERVER_DELAY_S"

# Verify policy server is still running
if ! kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
    echo "ERROR: Policy server failed to start!"
    echo ""
    echo "---- policy server log (last 200 lines) ----"
    tail -n 200 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi
echo "      Policy server is running."
echo ""

# Keep the policy server alive until Ctrl+C (otherwise the EXIT trap will stop it).
echo "      Press Ctrl+C to stop the policy server."

# `wait` returns the child exit code. With `set -e`, capture it explicitly so we can print logs.
set +e
wait "$POLICY_SERVER_PID"
POLICY_SERVER_EXIT_CODE=$?
set -e

echo ""
echo "Policy server exited with code: $POLICY_SERVER_EXIT_CODE"
echo "---- policy server log (last 200 lines) ----"
tail -n 200 "$LOG_FILE" 2>/dev/null || true

exit "$POLICY_SERVER_EXIT_CODE"
