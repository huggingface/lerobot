#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG="${PICKLIFT_PRACTICE_CONFIG:-$CONFIG_HOME/lerobot/picklift-practice.json}"
STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
LOG_DIR="$STATE_HOME/lerobot"
LOG_FILE="$LOG_DIR/picklift_practice.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

if [[ ! -f "$CONFIG" ]]; then
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="SO-101 练习配置缺失" \
      --text="请从 practice.template.json 创建本地配置：$CONFIG"
  fi
  exit 1
fi

if ! uv run --with 'opencv-python>=4.10' \
  python -m examples.picklift_v3.practice --config "$CONFIG" >>"$LOG_FILE" 2>&1; then
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="SO-101 练习启动失败" \
      --text="请查看日志：$LOG_FILE"
  fi
  exit 1
fi
