#!/bin/bash

# LeKiwi Flask 遥操作服务启动脚本

echo "=========================================="
echo "    LeKiwi Flask 遥操作服务启动脚本"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查是否在正确的目录
if [ ! -f "flask_teleop_server.py" ]; then
    echo "错误: 请在examples/lekiwi目录下运行此脚本"
    echo "当前目录: $(pwd)"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import flask" 2>/dev/null || {
    echo "安装Flask..."
    pip install flask flask-cors
}

python3 -c "import lerobot" 2>/dev/null || {
    echo "错误: 未找到LeRobot，请先安装LeRobot"
    echo "运行: pip install -e ."
    exit 1
}

echo "依赖检查完成"
echo ""

# 显示使用说明
echo "使用说明:"
echo "1. 确保LeKiwi主机正在运行:"
echo "   python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi"
echo ""
echo "2. 启动Flask服务后，在浏览器中访问:"
echo "   http://localhost:5000"
echo ""
echo "3. 在网页界面中输入连接参数并点击连接"
echo ""

# 询问是否继续
read -p "是否继续启动Flask服务? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消启动"
    exit 0
fi

echo ""
echo "启动Flask服务..."
echo "按 Ctrl+C 停止服务"
echo ""

# 启动Flask服务
python3 flask_teleop_server.py
