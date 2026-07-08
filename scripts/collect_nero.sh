#!/bin/bash
# NERO 一键遥操作+数据采集
# 用法: bash collect_nero.sh [数据集名称] [集数]
# 示例: bash collect_nero.sh my_pick_task 10

set -e

DATASET_NAME="${1:-nero_teleop}"
NUM_EPISODES="${2:-50}"
OUTPUT_DIR="/home/yuhang/datasets"
FPS=30
CAN_CHANNEL="can0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="$SCRIPT_DIR/nero_collect.py"

echo "=========================================="
echo "  NERO 一键遥操作 + 数据采集"
echo "=========================================="
echo "  数据集:   $DATASET_NAME"
echo "  集数:     $NUM_EPISODES"
echo "  频率:     ${FPS}Hz"
echo "  输出:     $OUTPUT_DIR/$DATASET_NAME"
echo "=========================================="
echo ""

# Step 1: 激活 CAN 接口
echo "[1/3] 激活 CAN 接口..."
CAN_SCRIPT="/home/yuhang/projects/lerobot/nero/QuestArmTeleop/src/agx_arm_ros/scripts/can_activate.sh"
if [ -f "$CAN_SCRIPT" ]; then
    bash "$CAN_SCRIPT"
    echo "  CAN $CAN_CHANNEL 已激活"
else
    echo "  [WARN] CAN 脚本不存在: $CAN_SCRIPT"
    echo "  尝试手动激活..."
    sudo ip link set "$CAN_CHANNEL" up type can bitrate 1000000 2>/dev/null || true
fi
echo ""

# Step 2: 激活 conda 环境
echo "[2/3] 激活 vt conda 环境..."
eval "$(conda shell.bash hook)"
conda activate vt
echo "  Python: $(python --version)"
echo ""

# Step 3: 启动采集
echo "[3/3] 启动遥操作+采集..."
echo ""
python "$COLLECT_SCRIPT" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$NUM_EPISODES" \
    --fps "$FPS" \
    --can_channel "$CAN_CHANNEL"
