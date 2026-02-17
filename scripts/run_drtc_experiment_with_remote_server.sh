#!/bin/bash
# =============================================================================
# DRTC Experiment Runner (Remote Policy Server)
# =============================================================================
#
# Runs DRTC experiments from a robot client machine (e.g., Raspberry Pi)
# while the policy server runs on a remote/cloud host.
#
# This script:
#   1. Starts or reuses an SSH tunnel to the remote policy server
#   2. Runs the standard experiment runner with server_address forced to the
#      tunnel's local gRPC port
#
# Usage:
#   ./scripts/run_drtc_experiment_remote_server.sh --config mixture_of_faults
#   ./scripts/run_drtc_experiment_remote_server.sh --config spike --output_dir results/experiments
#
# Notes:
#   - Do NOT pass --server_address; this script manages it through the tunnel.
#   - You can still pass all normal run_drtc_experiment.py flags.
#
# Environment variables:
#   TUNNEL_SSH_PORT           - SSH port to remote host (default: 18468)
#   TUNNEL_SSH_USER_HOST      - SSH target (default: root@103.196.86.69)
#   TUNNEL_GRPC_LOCAL_PORT    - Local forwarded gRPC port (default: 18080)
#   TUNNEL_VIZ_HTTP_LOCAL_PORT- Local forwarded viz HTTP port (default: 18088)
#   TUNNEL_VIZ_WS_LOCAL_PORT  - Local forwarded viz WS port (default: 18089)
#   TUNNEL_GRPC_REMOTE_PORT   - Remote gRPC port (default: 8080)
#   TUNNEL_VIZ_HTTP_REMOTE_PORT - Remote viz HTTP port (default: 8088)
#   TUNNEL_VIZ_WS_REMOTE_PORT - Remote viz WS port (default: 8089)
#   TUNNEL_REUSE_EXISTING     - 1 to reuse existing listeners, else fail-fast
#
# =============================================================================

set -e

# Optional debug tracing for this script
if [ "${LEROBOT_DEBUG:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TUNNEL_LOG_FILE="$LOG_DIR/ssh_tunnel_experiment_${LOG_TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Cloud tunnel configuration (Pi/robot client -> cloud policy server)
# -----------------------------------------------------------------------------
TUNNEL_SSH_PORT="${TUNNEL_SSH_PORT:-18468}"
TUNNEL_SSH_USER_HOST="${TUNNEL_SSH_USER_HOST:-root@103.196.86.69}"
TUNNEL_GRPC_LOCAL_PORT="${TUNNEL_GRPC_LOCAL_PORT:-18080}"
TUNNEL_VIZ_HTTP_LOCAL_PORT="${TUNNEL_VIZ_HTTP_LOCAL_PORT:-18088}"
TUNNEL_VIZ_WS_LOCAL_PORT="${TUNNEL_VIZ_WS_LOCAL_PORT:-18089}"
TUNNEL_GRPC_REMOTE_PORT="${TUNNEL_GRPC_REMOTE_PORT:-8080}"
TUNNEL_VIZ_HTTP_REMOTE_PORT="${TUNNEL_VIZ_HTTP_REMOTE_PORT:-8088}"
TUNNEL_VIZ_WS_REMOTE_PORT="${TUNNEL_VIZ_WS_REMOTE_PORT:-8089}"
TUNNEL_REUSE_EXISTING="${TUNNEL_REUSE_EXISTING:-0}"

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
    echo "       run_drtc_experiment_remote_server.sh sets --server_address automatically"
    echo "       to 127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT} via SSH tunnel."
    exit 1
fi

# PIDs for cleanup
SSH_TUNNEL_PID=""
STARTED_TUNNEL=false

cleanup() {
    echo ""
    echo "Shutting down remote experiment components..."

    if [ "$STARTED_TUNNEL" = true ] && [ -n "$SSH_TUNNEL_PID" ] && kill -0 "$SSH_TUNNEL_PID" 2>/dev/null; then
        echo "Stopping SSH tunnel (PID: $SSH_TUNNEL_PID)..."
        kill -TERM "$SSH_TUNNEL_PID" 2>/dev/null || true
        wait "$SSH_TUNNEL_PID" 2>/dev/null || true
    fi

    echo "Cleanup complete."
}

on_sigint() {
    exit 130
}

on_sigterm() {
    exit 143
}

trap cleanup EXIT
trap on_sigint SIGINT
trap on_sigterm SIGTERM

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  DRTC Experiment Runner (Remote Server)"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Arguments:    $*"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Start SSH tunnel to cloud policy server
# -----------------------------------------------------------------------------
echo "[1/2] Starting SSH tunnel to cloud policy server..."
echo "      Target: $TUNNEL_SSH_USER_HOST (ssh port: $TUNNEL_SSH_PORT)"
echo "      Local forwards:"
echo "        - 127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}  -> localhost:${TUNNEL_GRPC_REMOTE_PORT} (gRPC)"
echo "        - 127.0.0.1:${TUNNEL_VIZ_HTTP_LOCAL_PORT} -> localhost:${TUNNEL_VIZ_HTTP_REMOTE_PORT} (viz HTTP)"
echo "        - 127.0.0.1:${TUNNEL_VIZ_WS_LOCAL_PORT}   -> localhost:${TUNNEL_VIZ_WS_REMOTE_PORT} (viz WS)"

existing_listeners=""
for p in "$TUNNEL_GRPC_LOCAL_PORT" "$TUNNEL_VIZ_HTTP_LOCAL_PORT" "$TUNNEL_VIZ_WS_LOCAL_PORT"; do
    if ss -lnt | grep -Eq ":${p}\b"; then
        existing_listeners="1"
    fi
done

if [ -n "$existing_listeners" ]; then
    echo "      Detected existing listener(s) on one or more tunnel ports:"
    ss -lntp | grep -E ":((${TUNNEL_GRPC_LOCAL_PORT})|(${TUNNEL_VIZ_HTTP_LOCAL_PORT})|(${TUNNEL_VIZ_WS_LOCAL_PORT}))\b" || true
    if [ "$TUNNEL_REUSE_EXISTING" = "1" ]; then
        echo "      Reusing existing listener(s) (TUNNEL_REUSE_EXISTING=1)."
    else
        echo "ERROR: Tunnel ports are already in use."
        echo "       Either stop the existing tunnel/process, or re-run with:"
        echo "         TUNNEL_REUSE_EXISTING=1 ./scripts/run_drtc_experiment_remote_server.sh ..."
        exit 1
    fi
else
    : >"$TUNNEL_LOG_FILE"
    ssh -p "$TUNNEL_SSH_PORT" -N \
        -o ExitOnForwardFailure=yes \
        -o ConnectTimeout=10 \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o LogLevel=ERROR \
        -L "${TUNNEL_GRPC_LOCAL_PORT}:localhost:${TUNNEL_GRPC_REMOTE_PORT}" \
        -L "${TUNNEL_VIZ_HTTP_LOCAL_PORT}:localhost:${TUNNEL_VIZ_HTTP_REMOTE_PORT}" \
        -L "${TUNNEL_VIZ_WS_LOCAL_PORT}:localhost:${TUNNEL_VIZ_WS_REMOTE_PORT}" \
        "$TUNNEL_SSH_USER_HOST" >"$TUNNEL_LOG_FILE" 2>&1 &
    SSH_TUNNEL_PID=$!
    STARTED_TUNNEL=true
    echo "      SSH tunnel started (PID: $SSH_TUNNEL_PID)"
    echo "      Tunnel log: $TUNNEL_LOG_FILE"

    sleep 0.2
    if ! kill -0 "$SSH_TUNNEL_PID" 2>/dev/null; then
        echo "ERROR: SSH tunnel failed to start (process exited)."
        echo "---- ssh tunnel log (last 50 lines) ----"
        tail -n 50 "$TUNNEL_LOG_FILE" 2>/dev/null || true
        exit 1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Tunnel health checks (fail fast on missing remote services)
# -----------------------------------------------------------------------------
echo "Checking tunnel targets..."

tcp_probe() {
    local port="$1"
    timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

if ! tcp_probe "$TUNNEL_GRPC_LOCAL_PORT"; then
    echo "ERROR: Tunnel is up but policy server is not reachable on 127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}."
    echo "       This usually means the cloud policy server is not listening on localhost:${TUNNEL_GRPC_REMOTE_PORT}."
    echo "       Start/verify cloud server separately (for example via scripts/start_drtc_server.sh)."
    echo "---- ssh tunnel log (last 50 lines) ----"
    tail -n 50 "$TUNNEL_LOG_FILE" 2>/dev/null || true
    exit 1
fi

if ! tcp_probe "$TUNNEL_VIZ_HTTP_LOCAL_PORT"; then
    echo "WARNING: Viz HTTP not reachable on 127.0.0.1:${TUNNEL_VIZ_HTTP_LOCAL_PORT} (tunnels to :${TUNNEL_VIZ_HTTP_REMOTE_PORT})."
fi
if ! tcp_probe "$TUNNEL_VIZ_WS_LOCAL_PORT"; then
    echo "WARNING: Viz WebSocket not reachable on 127.0.0.1:${TUNNEL_VIZ_WS_LOCAL_PORT} (tunnels to :${TUNNEL_VIZ_WS_REMOTE_PORT})."
fi

echo ""

# -----------------------------------------------------------------------------
# Step 2: Run Experiment (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting experiment..."
echo "      Server address is forced to: 127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}"
if [ "$HAS_VIZ_WS_URL_ARG" = "0" ]; then
    echo "      Trajectory viz WS URL defaulted to: ws://localhost:${TUNNEL_VIZ_WS_LOCAL_PORT}"
fi
echo "      Press Ctrl+C to stop."
echo ""
echo "----------------------------------------------"

EXPERIMENT_CMD=(
    uv run --no-sync python examples/experiments/run_drtc_experiment.py
    --server_address "127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}"
)

if [ "$HAS_VIZ_WS_URL_ARG" = "0" ]; then
    EXPERIMENT_CMD+=(--trajectory_viz_ws_url "ws://localhost:${TUNNEL_VIZ_WS_LOCAL_PORT}")
fi

EXPERIMENT_CMD+=("$@")
"${EXPERIMENT_CMD[@]}"
