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

# Configuration
POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
trap cleanup SIGINT SIGTERM EXIT

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
# Use --no-sync to skip dependency resolution (avoids grpcio version conflicts)
uv run --no-sync python examples/tutorial/async-inf/policy_server_drtc.py &
POLICY_SERVER_PID=$!
echo "      Policy server started (PID: $POLICY_SERVER_PID)"
echo "      Trajectory visualization: http://localhost:8088"
echo "      Waiting ${POLICY_SERVER_DELAY_S}s for server to initialize..."
sleep "$POLICY_SERVER_DELAY_S"

# Verify policy server is still running
if ! kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
    echo "ERROR: Policy server failed to start!"
    exit 1
fi
echo "      Policy server is running."
echo ""

# -----------------------------------------------------------------------------
# Step 2: Start Robot Client (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting robot client..."
echo "      Press Ctrl+C to stop all components."
echo ""
echo "----------------------------------------------"

# Run robot client in foreground (this blocks until Ctrl+C)
# Use --no-sync to skip dependency resolution (avoids grpcio version conflicts)
uv run --no-sync python examples/tutorial/async-inf/robot_client_drtc.py

# If robot client exits normally, cleanup will be called via trap
