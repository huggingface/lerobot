#!/bin/bash

# 定义目标列表: "IP|Hostname|User"
declare -a TARGETS=(
    "100.79.20.95|droplab-ms-7d37-1|droplab"
    "100.89.46.18|desktop-k13c5uu|droplab"
    "100.66.251.32|droplab-ms-7d37|droplab"
    "100.88.74.117|gehan-ms-7d37|gehan"
)

function print_usage() {
    echo "用法: $0 <文件或文件夹路径> <目标索引>"
    echo "可用目标:"
    for i in "${!TARGETS[@]}"; do
        IFS='|' read -r ip name user <<< "${TARGETS[$i]}"
        echo "  [$i] $name ($ip)"
    done
}

# 检查参数
if [ "$#" -ne 2 ]; then
    print_usage
    exit 1
fi

SOURCE_PATH="$1"
INDEX="$2"

# 检查源文件或文件夹
if [ ! -e "$SOURCE_PATH" ]; then
    echo "错误: 找不到文件或文件夹 '$SOURCE_PATH'"
    exit 1
fi

# 检查索引是否有效
if ! [[ "$INDEX" =~ ^[0-9]+$ ]] || [ "$INDEX" -ge "${#TARGETS[@]}" ]; then
    echo "错误: 无效的索引 '$INDEX'"
    print_usage
    exit 1
fi

# 解析目标信息
IFS='|' read -r TARGET_IP TARGET_NAME TARGET_USER <<< "${TARGETS[$INDEX]}"

# 构造目标地址
# 默认是 ~/Downloads/，如果需要针对 Windows (如 index 1, 2) 调整路径，可以在这里添加逻辑
# 这里假设所有目标都接受 ~/Downloads/
DEST_PATH="~/Downloads/"
FULL_TARGET="$TARGET_USER@$TARGET_IP"

echo "========================================"
echo "目标主机: $TARGET_NAME"
echo "目标 IP : $TARGET_IP"
echo "目标用户: $TARGET_USER"
echo "传输源  : $SOURCE_PATH"
echo "========================================"

# 执行 rsync
rsync -avzP "$SOURCE_PATH" "$FULL_TARGET:$DEST_PATH"

if [ $? -eq 0 ]; then
    echo "✅ 传输成功完成"
else
    echo "❌ 传输失败"
    exit 1
fi
