#!/bin/bash
# =============================================================================
# Plot Experiment Results
# =============================================================================
#
# Plots results from async inference experiments.
#
# Usage:
#   ./scripts/plot_experiment_results.sh <experiment_name> [mode]
#
# Arguments:
#   experiment_name  - Name of the experiment directory in results/
#   mode             - Plot mode: basic (default), detailed, estimator_comparison
#
# Examples:
#   ./scripts/plot_experiment_results.sh experiments
#   ./scripts/plot_experiment_results.sh experiments estimator_comparison
#   ./scripts/plot_experiment_results.sh spike_test detailed
#
# =============================================================================

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [mode]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name  - Name of the experiment directory in results/"
    echo "  mode             - Plot mode: basic (default), detailed, estimator_comparison"
    echo ""
    echo "Examples:"
    echo "  $0 experiments"
    echo "  $0 experiments estimator_comparison"
    exit 1
fi

EXPERIMENT_NAME="$1"
MODE="${2:-basic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR="results/${EXPERIMENT_NAME}/"
OUTPUT_FILE="results/${EXPERIMENT_NAME}_${MODE}.png"

# Check if input directory exists
if [ ! -d "$PROJECT_ROOT/$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "Available directories in results/:"
    ls -d "$PROJECT_ROOT/results"/*/ 2>/dev/null | xargs -n1 basename || echo "  (none)"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Plotting experiment results..."
echo "  Input:  $INPUT_DIR"
echo "  Mode:   $MODE"
echo "  Output: $OUTPUT_FILE"
echo ""

uv run --no-sync python examples/experiments/plot_results.py \
    --input "$INPUT_DIR" \
    --mode "$MODE" \
    --output "$OUTPUT_FILE"

echo ""
echo "Plot saved to: $OUTPUT_FILE"
