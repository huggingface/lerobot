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
# -----------------------------------------------------------------------------
# Step 2: Start Robot Client (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting robot client..."
echo "      Press Ctrl+C to stop all components."
echo ""
echo "----------------------------------------------"

# Run robot client in foreground (this blocks until Ctrl+C)
# Use --no-sync to skip dependency resolution (avoids grpcio version conflicts)
uv run --no-sync python examples/tutorial/async-inf/robot_client_improved.py

# If robot client exits normally, cleanup will be called via trap
