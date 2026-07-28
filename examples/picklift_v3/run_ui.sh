#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/configs/pilot.template.json}"

exec uv run --with 'opencv-python==4.10.0.84' \
  python -m examples.picklift_v3.record --config "$CONFIG"
