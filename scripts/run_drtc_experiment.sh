#!/bin/bash
# =============================================================================
# DRTC Experiment Runner
# =============================================================================
#
# Starts the policy server (if not already running), then runs experiments
# defined in a YAML config. All arguments are forwarded to the Python
# experiment runner.
#
# Usage:
#   ./scripts/run_drtc_experiment.sh --config mixture_of_faults
#   ./scripts/run_drtc_experiment.sh --config spike --output_dir results/experiments
#   ./scripts/run_drtc_experiment.sh --config examples/experiments/configs/disconnect.yaml
#
# Environment variables:
#   POLICY_SERVER_DELAY_S   - Seconds to wait for policy server startup (default: 3)
#   POLICY_SERVER_PORT      - Port to check / bind (default: 8080)
#
# =============================================================================

set -e

POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/policy_server_${LOG_TIMESTAMP}.log"

# PIDs for cleanup
POLICY_SERVER_PID=""
STARTED_SERVER=false

cleanup() {
    echo ""
    echo "Shutting down experiment components..."

    if [ "$STARTED_SERVER" = true ] && [ -n "$POLICY_SERVER_PID" ] && kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
        echo "Stopping policy server (PID: $POLICY_SERVER_PID)..."
        kill -TERM "$POLICY_SERVER_PID" 2>/dev/null || true
        wait "$POLICY_SERVER_PID" 2>/dev/null || true
    fi

    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  DRTC Experiment Runner"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Arguments:    $*"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Start Policy Server (kill existing + start fresh)
# -----------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

if ss -tlnp 2>/dev/null | grep -q ":${POLICY_SERVER_PORT} " || \
   lsof -iTCP:"${POLICY_SERVER_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[1/2] Killing existing policy server on port ${POLICY_SERVER_PORT}..."
    # Find and kill the process listening on the port
    EXISTING_PID=$(lsof -ti TCP:"${POLICY_SERVER_PORT}" -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
        kill -TERM $EXISTING_PID 2>/dev/null || true
        sleep 1
        # Force-kill if still running
        kill -0 $EXISTING_PID 2>/dev/null && kill -9 $EXISTING_PID 2>/dev/null || true
        sleep 0.5
    fi
    echo "      Old server stopped."
fi

echo "[1/2] Starting policy server..."
echo "      Policy server logs: $LOG_FILE"
uv run --no-sync python examples/tutorial/async-inf/policy_server_drtc.py --verbose-diagnostics >"$LOG_FILE" 2>&1 &
POLICY_SERVER_PID=$!
STARTED_SERVER=true
echo "      Policy server started (PID: $POLICY_SERVER_PID)"
echo "      Trajectory visualization: http://localhost:8088"
echo "      Waiting ${POLICY_SERVER_DELAY_S}s for server to initialize..."
sleep "$POLICY_SERVER_DELAY_S"

if ! kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
    echo "ERROR: Policy server failed to start!"
    echo ""
    echo "---- policy server log (last 200 lines) ----"
    tail -n 200 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi
echo "      Policy server is running."
echo ""

# -----------------------------------------------------------------------------
# Step 2: Run Experiment (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting experiment..."
echo "      Press Ctrl+C to stop."
echo ""
echo "----------------------------------------------"

uv run --no-sync python examples/experiments/run_drtc_experiment.py "$@"

# Show server-side diagnostics from the log (if any DIAG_SERVER lines exist)
if [ -f "$LOG_FILE" ] && grep -q "DIAG_SERVER" "$LOG_FILE"; then
    echo ""
    echo "----------------------------------------------"
    echo "  Server diagnostics (from $LOG_FILE):"
    echo "----------------------------------------------"
    grep "DIAG_SERVER" "$LOG_FILE"
fi
echo ""
echo "Server log: $LOG_FILE"
