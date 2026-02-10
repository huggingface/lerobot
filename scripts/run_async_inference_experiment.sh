#!/bin/bash
# =============================================================================
# Async Inference Experiment Runner Script
# =============================================================================
#
# Runs latency-adaptive async inference experiments with configurable parameters.
# Starts the policy server, then runs the experiment client.
#
# Usage:
#   ./scripts/run_async_inference_experiment.sh
#
# Configure the experiment by editing the variables below.
#
# =============================================================================

# =============================================================================
# Experiment Configuration (edit these)
# =============================================================================

# Experiment config mode: set to an experiment config name to run a predefined
# config, or leave empty for single experiment mode.
#
# Available experiment configs:
#   estimator_comparison  - Compare jk vs max_last_10 estimators
#   k_parameter           - Test K values (0.5, 1.0, 1.5, 2.0, 4.0)
#   epsilon               - Test epsilon values (0, 1, 2, 3, 5)
#   quick_test            - Quick comparison of cooldown on/off (30s each)
#   obs_drop              - Observation drop scenarios
#   action_drop           - Action drop scenarios
#   drop_recovery         - Compare drop recovery with/without cooldown
#   spike                 - Spike injection tests (jk vs max_last_10)
#   spike_estimator       - Compare estimators under spike conditions
#   latency_estimator_spike_recovery - Test estimator recovery after spikes
#
# EXPERIMENT_CONFIG_NAME="spike"
EXPERIMENT_CONFIG_NAME=""


# Single experiment settings (used when EXPERIMENT_CONFIG_NAME is empty)
ESTIMATOR="jk"                   # "jk" or "max_last_10"
COOLDOWN="on"                    # "on" or "off"
LATENCY_K="1.5"                  # K parameter for Jacobson-Karels
EPSILON="1"                      # Cooldown buffer
DURATION_S="30.0"                # Experiment duration in seconds

# Spike injection (JSON array, empty string = no spikes)
# Example: '[{"start_s": 5.0, "delay_ms": 2000}, {"start_s": 15.0, "delay_ms": 1000}]'
SPIKES=''

# Drop injection (JSON arrays, empty string = no drops)
# Example: '[{"start_s": 5.0, "duration_s": 1.0}]'
# DROP_OBS='[{"start_s": 10.0, "duration_s": 2.0}]'
# DROP_ACTION=''

# Duplicate injection (JSON arrays, empty string = no duplicates)
# Example: '[{"start_s": 5.0, "duration_s": 1.0}]'
DUP_OBS=''
DUP_ACTION=''

# Reorder injection (JSON arrays, empty string = no reordering)
# Example: '[{"start_s": 5.0, "duration_s": 2.0}]'
REORDER_OBS='[{"start_s": 5.0, "duration_s": 2.0}]'
REORDER_ACTION=''

# Output settings
OUTPUT_DIR="results/experiments"
PAUSE_BETWEEN_S="5.0"           # Pause between experiments in config

# Server settings
SERVER_ADDRESS="192.168.4.37:8080"

# =============================================================================
# Script Configuration (usually don't need to change)
# =============================================================================

POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"

# =============================================================================
# Implementation (don't edit below unless you know what you're doing)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# PIDs for cleanup
POLICY_SERVER_PID=""

# Cleanup function
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

# Register signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Change to project root
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Async Inference Experiment Runner"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Show experiment configuration
if [ -n "$EXPERIMENT_CONFIG_NAME" ]; then
    echo "Mode: Experiment config"
    echo "  Config name: $EXPERIMENT_CONFIG_NAME"
    echo "  Pause between experiments: ${PAUSE_BETWEEN_S}s"
else
    echo "Mode: Single experiment"
    echo "  Estimator: $ESTIMATOR"
    echo "  Cooldown: $COOLDOWN"
    echo "  Latency K: $LATENCY_K"
    echo "  Epsilon: $EPSILON"
    echo "  Duration: ${DURATION_S}s"
    if [ -n "$SPIKES" ]; then
        echo "  Spikes: $SPIKES"
    fi
    if [ -n "$DROP_OBS" ]; then
        echo "  Drop obs: $DROP_OBS"
    fi
    if [ -n "$DROP_ACTION" ]; then
        echo "  Drop action: $DROP_ACTION"
    fi
    if [ -n "$DUP_OBS" ]; then
        echo "  Dup obs: $DUP_OBS"
    fi
    if [ -n "$DUP_ACTION" ]; then
        echo "  Dup action: $DUP_ACTION"
    fi
    if [ -n "$REORDER_OBS" ]; then
        echo "  Reorder obs: $REORDER_OBS"
    fi
    if [ -n "$REORDER_ACTION" ]; then
        echo "  Reorder action: $REORDER_ACTION"
    fi
fi
echo "  Output dir: $OUTPUT_DIR"
echo "  Server address: $SERVER_ADDRESS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Step 1: Start Policy Server (includes trajectory visualization)
# -----------------------------------------------------------------------------
echo "[1/2] Starting policy server..."
# Use --no-sync to skip dependency resolution (avoids grpcio version conflicts)
uv run --no-sync python examples/tutorial/async-inf/policy_server_improved.py &
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
# Step 2: Run Experiment (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting experiment..."
echo "      Press Ctrl+C to stop."
echo ""
echo "----------------------------------------------"

# Build experiment command
EXPERIMENT_CMD="uv run --no-sync python examples/experiments/run_async_inference_experiment.py"
EXPERIMENT_CMD="$EXPERIMENT_CMD --output_dir $OUTPUT_DIR"
EXPERIMENT_CMD="$EXPERIMENT_CMD --server_address $SERVER_ADDRESS"

if [ -n "$EXPERIMENT_CONFIG_NAME" ]; then
    # Experiment config mode
    EXPERIMENT_CMD="$EXPERIMENT_CMD --experiment_config $EXPERIMENT_CONFIG_NAME"
    EXPERIMENT_CMD="$EXPERIMENT_CMD --pause_between_s $PAUSE_BETWEEN_S"
else
    # Single experiment mode
    EXPERIMENT_CMD="$EXPERIMENT_CMD --estimator $ESTIMATOR"
    EXPERIMENT_CMD="$EXPERIMENT_CMD --cooldown $COOLDOWN"
    EXPERIMENT_CMD="$EXPERIMENT_CMD --latency_k $LATENCY_K"
    EXPERIMENT_CMD="$EXPERIMENT_CMD --epsilon $EPSILON"
    EXPERIMENT_CMD="$EXPERIMENT_CMD --duration_s $DURATION_S"
    
    if [ -n "$SPIKES" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --spikes '$SPIKES'"
    fi
    if [ -n "$DROP_OBS" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --drop_obs '$DROP_OBS'"
    fi
    if [ -n "$DROP_ACTION" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --drop_action '$DROP_ACTION'"
    fi
    if [ -n "$DUP_OBS" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --dup_obs '$DUP_OBS'"
    fi
    if [ -n "$DUP_ACTION" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --dup_action '$DUP_ACTION'"
    fi
    if [ -n "$REORDER_OBS" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --reorder_obs '$REORDER_OBS'"
    fi
    if [ -n "$REORDER_ACTION" ]; then
        EXPERIMENT_CMD="$EXPERIMENT_CMD --reorder_action '$REORDER_ACTION'"
    fi
fi

# Run the experiment
eval "$EXPERIMENT_CMD"

# If experiment exits normally, cleanup will be called via trap
