#!/bin/bash
# =============================================================================
# DRTC Experiment Runner
# =============================================================================
#
# Starts the policy server, then runs experiments defined in a YAML config.
# All arguments are forwarded to the Python experiment runner.
#
# Usage:
#   ./scripts/run_drtc_experiment.sh --config mixture_of_faults
#   ./scripts/run_drtc_experiment.sh --config spike --output_dir results/experiments
#   ./scripts/run_drtc_experiment.sh --config examples/experiments/configs/disconnect.yaml
#
# Environment variables:
#   POLICY_SERVER_DELAY_S   - Seconds to wait for policy server startup (default: 3)
#
# =============================================================================

set -e

POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/policy_server.log"

# PIDs for cleanup
POLICY_SERVER_PID=""

cleanup() {
    echo ""
    echo "Shutting down experiment components..."

    if [ -n "$POLICY_SERVER_PID" ] && kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
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
# Step 1: Start Policy Server
# -----------------------------------------------------------------------------
echo "[1/2] Starting policy server..."
mkdir -p "$LOG_DIR"
echo "      Policy server logs: $LOG_FILE"
uv run --no-sync python examples/tutorial/async-inf/policy_server_drtc.py >"$LOG_FILE" 2>&1 &
POLICY_SERVER_PID=$!
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
