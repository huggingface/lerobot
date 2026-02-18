#!/bin/bash
# =============================================================================
# DRTC Experiment Runner (Remote Policy Server)
# =============================================================================
#
# Runs DRTC experiments from a robot client machine (e.g., Raspberry Pi)
# while the policy server runs on a remote host reachable directly
# (e.g., via Tailscale).
#
# This script runs the standard experiment runner with --server_address and
# --trajectory_viz_ws_url set to the remote host.
#
# Usage:
#   ./scripts/run_drtc_experiment_with_remote_server.sh --config mixture_of_faults
#   ./scripts/run_drtc_experiment_with_remote_server.sh --config spike --output_dir results/experiments
#
# Notes:
#   - Do NOT pass --server_address; this script sets it to the remote host.
#   - You can still pass all normal run_drtc_experiment.py flags.
#
# Environment variables:
#   REMOTE_SERVER_HOST    - Remote policy server hostname (default: jackml.tail26c8db.ts.net)
#   REMOTE_GRPC_PORT      - Remote gRPC port (default: 8080)
#   REMOTE_VIZ_HTTP_PORT  - Remote viz HTTP port (default: 8088)
#   REMOTE_VIZ_WS_PORT    - Remote viz WebSocket port (default: 8089)
#
# =============================================================================

set -e

# Optional debug tracing for this script
if [ "${LEROBOT_DEBUG:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# -----------------------------------------------------------------------------
# Remote policy server (direct reachability, e.g. Tailscale)
# -----------------------------------------------------------------------------
REMOTE_SERVER_HOST="${REMOTE_SERVER_HOST:-jackml.tail26c8db.ts.net}"
REMOTE_GRPC_PORT="${REMOTE_GRPC_PORT:-8080}"
REMOTE_VIZ_HTTP_PORT="${REMOTE_VIZ_HTTP_PORT:-8088}"
REMOTE_VIZ_WS_PORT="${REMOTE_VIZ_WS_PORT:-8089}"

# -----------------------------------------------------------------------------
# Validate passthrough args
# -----------------------------------------------------------------------------
HAS_SERVER_ADDRESS_ARG=0
HAS_VIZ_WS_URL_ARG=0
for arg in "$@"; do
    case "$arg" in
        --server_address|--server_address=*)
            HAS_SERVER_ADDRESS_ARG=1
            ;;
        --trajectory_viz_ws_url|--trajectory_viz_ws_url=*)
            HAS_VIZ_WS_URL_ARG=1
            ;;
    esac
done

if [ "$HAS_SERVER_ADDRESS_ARG" = "1" ]; then
    echo "ERROR: Do not pass --server_address to this script."
    echo "       This script sets --server_address to ${REMOTE_SERVER_HOST}:${REMOTE_GRPC_PORT}."
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  DRTC Experiment Runner (Remote Server)"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Arguments:    $*"
echo ""

# -----------------------------------------------------------------------------
# Optional: fail fast if remote policy server is unreachable
# -----------------------------------------------------------------------------
tcp_probe() {
    local host="$1"
    local port="$2"
    timeout 2 bash -c "cat < /dev/null > /dev/tcp/${host}/${port}" >/dev/null 2>&1
}

if ! tcp_probe "$REMOTE_SERVER_HOST" "$REMOTE_GRPC_PORT"; then
    echo "ERROR: Policy server is not reachable at ${REMOTE_SERVER_HOST}:${REMOTE_GRPC_PORT}."
    echo "       Start/verify the server on the remote host (e.g. via scripts/start_drtc_server.sh)."
    exit 1
fi

echo "Starting experiment..."
echo "      Server address: ${REMOTE_SERVER_HOST}:${REMOTE_GRPC_PORT}"
if [ "$HAS_VIZ_WS_URL_ARG" = "0" ]; then
    echo "      Trajectory viz WS URL: ws://${REMOTE_SERVER_HOST}:${REMOTE_VIZ_WS_PORT}"
fi
echo "      Press Ctrl+C to stop."
echo ""
echo "----------------------------------------------"

EXPERIMENT_CMD=(
    uv run --no-sync python examples/experiments/run_drtc_experiment.py
    --server_address "${REMOTE_SERVER_HOST}:${REMOTE_GRPC_PORT}"
)

if [ "$HAS_VIZ_WS_URL_ARG" = "0" ]; then
    EXPERIMENT_CMD+=(--trajectory_viz_ws_url "ws://${REMOTE_SERVER_HOST}:${REMOTE_VIZ_WS_PORT}")
fi

EXPERIMENT_CMD+=("$@")
"${EXPERIMENT_CMD[@]}"
