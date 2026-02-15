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
#   ./scripts/start_async_inference.sh 0 1              # RTC sweep: config index 0, batch 1
#   ./scripts/start_async_inference.sh 5 2              # RTC sweep: config index 5, batch 2
#   ./scripts/start_async_inference.sh 12 1             # Alex Soare sweep: n=5, batch 1
#
# RTC Sweep Config Index Mapping (15 configs):
#
#   Default sweep (sigma_d, full_traj):
#     0: sigma_d=0.1, full_traj=False    1: sigma_d=0.1, full_traj=True
#     2: sigma_d=0.2, full_traj=False    3: sigma_d=0.2, full_traj=True
#     4: sigma_d=0.4, full_traj=False    5: sigma_d=0.4, full_traj=True
#     6: sigma_d=0.6, full_traj=False    7: sigma_d=0.6, full_traj=True
#     8: sigma_d=0.8, full_traj=False    9: sigma_d=0.8, full_traj=True
#    10: sigma_d=1.0, full_traj=False   11: sigma_d=1.0, full_traj=True
#
#   Alex Soare sweep (denoising steps n, Beta=auto, sigma_d=0.2 fixed):
#    12: n=5,  Beta=auto    (faster inference, less smooth)
#    13: n=10, Beta=auto    (default, balanced)
#    14: n=20, Beta=auto    (slower inference, smoother)
#
#   Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
#
# Environment variables:
#   POLICY_SERVER_DELAY_S   - Seconds to wait after starting policy server (default: 3)
#   LEROBOT_DEBUG           - Set to 1 for debug logging
#   RTC_CONFIG_INDEX        - (auto-set) RTC sweep config index (0-11)
#   RTC_BATCH               - (auto-set) RTC sweep batch number
#
# =============================================================================

set -e

# Configuration
POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
TUNNEL_LOG_FILE="$LOG_DIR/ssh_tunnel.log"

# -----------------------------------------------------------------------------
# Cloud tunnel configuration (LAN box -> cloud policy server)
# -----------------------------------------------------------------------------
# These are the *local* ports on this machine. They forward to the cloud machine's
# standard ports (8080/8088/8089). We use 1808x to avoid conflicts with other
# port forwarders (e.g. editor/remote tooling).
TUNNEL_SSH_PORT="${TUNNEL_SSH_PORT:-18468}"
TUNNEL_SSH_USER_HOST="${TUNNEL_SSH_USER_HOST:-root@103.196.86.69}"
TUNNEL_GRPC_LOCAL_PORT="${TUNNEL_GRPC_LOCAL_PORT:-18080}"
TUNNEL_VIZ_HTTP_LOCAL_PORT="${TUNNEL_VIZ_HTTP_LOCAL_PORT:-18088}"
TUNNEL_VIZ_WS_LOCAL_PORT="${TUNNEL_VIZ_WS_LOCAL_PORT:-18089}"

# Remote ports (on the cloud host)
TUNNEL_GRPC_REMOTE_PORT="${TUNNEL_GRPC_REMOTE_PORT:-8080}"
TUNNEL_VIZ_HTTP_REMOTE_PORT="${TUNNEL_VIZ_HTTP_REMOTE_PORT:-8088}"
TUNNEL_VIZ_WS_REMOTE_PORT="${TUNNEL_VIZ_WS_REMOTE_PORT:-8089}"

# If set to 1, allow reusing an existing listener on the tunnel ports.
# Default is to fail fast (helps avoid "dangling" / stale tunnels).
TUNNEL_REUSE_EXISTING="${TUNNEL_REUSE_EXISTING:-0}"

# -----------------------------------------------------------------------------
# RTC Sweep Arguments (optional)
# -----------------------------------------------------------------------------
# If two arguments are provided, treat them as config index and batch number
if [ $# -ge 2 ]; then
    export RTC_CONFIG_INDEX="$1"
    export RTC_BATCH="$2"
    echo "RTC Sweep Mode: config_index=$RTC_CONFIG_INDEX, batch=$RTC_BATCH"
elif [ $# -eq 1 ]; then
    echo "ERROR: If providing arguments, must specify both config_index and batch"
    echo "Usage: $0 [config_index batch]"
    exit 1
fi

# PIDs for cleanup
POLICY_SERVER_PID=""
SSH_TUNNEL_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down async inference components..."
    
    if [ -n "$POLICY_SERVER_PID" ] && kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
        echo "Stopping policy server (PID: $POLICY_SERVER_PID)..."
        kill -TERM "$POLICY_SERVER_PID" 2>/dev/null || true
        wait "$POLICY_SERVER_PID" 2>/dev/null || true
    fi

    if [ -n "$SSH_TUNNEL_PID" ] && kill -0 "$SSH_TUNNEL_PID" 2>/dev/null; then
        echo "Stopping SSH tunnel (PID: $SSH_TUNNEL_PID)..."
        kill -TERM "$SSH_TUNNEL_PID" 2>/dev/null || true
        wait "$SSH_TUNNEL_PID" 2>/dev/null || true
    fi
    
    echo "Cleanup complete."
    exit 0
}

# Register signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Change to project root
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Step 1: Start SSH tunnel to cloud policy server
# -----------------------------------------------------------------------------
echo "[1/2] Starting SSH tunnel to cloud policy server..."
echo "      Target: $TUNNEL_SSH_USER_HOST (ssh port: $TUNNEL_SSH_PORT)"
echo "      Local forwards:"
echo "        - 127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}  -> localhost:${TUNNEL_GRPC_REMOTE_PORT} (gRPC)"
echo "        - 127.0.0.1:${TUNNEL_VIZ_HTTP_LOCAL_PORT} -> localhost:${TUNNEL_VIZ_HTTP_REMOTE_PORT} (viz HTTP)"
echo "        - 127.0.0.1:${TUNNEL_VIZ_WS_LOCAL_PORT}   -> localhost:${TUNNEL_VIZ_WS_REMOTE_PORT} (viz WS)"

# Check whether any of the local tunnel ports are already bound.
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
        echo "         TUNNEL_REUSE_EXISTING=1 ./scripts/start_client.sh"
        exit 1
    fi
else
    # ExitOnForwardFailure ensures we fail fast if any -L can't bind.
    # LogLevel=ERROR + redirect prevents noisy 'channel open failed' spam in your terminal.
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
    echo "      SSH tunnel started (PID: $SSH_TUNNEL_PID)"

    # Give ssh a moment to error out if something is wrong, then validate.
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
# Notes:
# - Local ports 18080/18088/18089 exist on THIS machine (LAN box), not on the cloud host.
# - Cloud services must listen on 8080/8088/8089 for the forwards to succeed.
echo "Checking tunnel targets..."

# Minimal TCP connect probe (no protocol validation).
tcp_probe() {
    local port="$1"
    # `timeout` is used to avoid hanging if something wedges.
    timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

# gRPC is required. If this fails, the client will hit "connection reset by peer".
if ! tcp_probe "$TUNNEL_GRPC_LOCAL_PORT"; then
    echo "ERROR: Tunnel is up but policy server is not reachable on 127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}."
    echo "       This usually means the cloud policy server is not listening on localhost:${TUNNEL_GRPC_REMOTE_PORT}."
    echo "       On the cloud machine, confirm with:"
    echo "         ss -lntp | egrep ':(8080|8088|8089)\\b' || true"
    echo "       If needed, start it (cloud):"
    echo "         uv run --no-sync python examples/tutorial/async-inf/policy_server_drtc.py --host 127.0.0.1 --port 8080"
    echo "---- ssh tunnel log (last 50 lines) ----"
    tail -n 50 "$TUNNEL_LOG_FILE" 2>/dev/null || true
    exit 1
fi

# Viz endpoints are optional (warn only). HTTP typically works even if WS doesn't.
if ! tcp_probe "$TUNNEL_VIZ_HTTP_LOCAL_PORT"; then
    echo "WARNING: Viz HTTP not reachable on 127.0.0.1:${TUNNEL_VIZ_HTTP_LOCAL_PORT} (tunnels to :${TUNNEL_VIZ_HTTP_REMOTE_PORT})."
fi
if ! tcp_probe "$TUNNEL_VIZ_WS_LOCAL_PORT"; then
    echo "WARNING: Viz WebSocket not reachable on 127.0.0.1:${TUNNEL_VIZ_WS_LOCAL_PORT} (tunnels to :${TUNNEL_VIZ_WS_REMOTE_PORT})."
    echo "         If 8088 is listening but 8089 is not, install websockets on the cloud env and restart the policy server."
fi

echo ""

# -----------------------------------------------------------------------------
# Step 2: Start Robot Client (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting robot client..."
echo "      Press Ctrl+C to stop all components."
echo ""
echo "----------------------------------------------"

# Configure the example robot client to use the tunnel's local ports.
export LEROBOT_SERVER_ADDRESS="127.0.0.1:${TUNNEL_GRPC_LOCAL_PORT}"
export LEROBOT_TRAJECTORY_VIZ_WS_URL="ws://localhost:${TUNNEL_VIZ_WS_LOCAL_PORT}"

# Run robot client in foreground (this blocks until Ctrl+C)
# Use --no-sync to skip dependency resolution (avoids grpcio version conflicts)
uv run --no-sync python examples/tutorial/async-inf/robot_client_drtc.py

# If robot client exits normally, cleanup will be called via trap
