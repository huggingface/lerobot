#!/usr/bin/env python3
"""
LeKiwi 网页遥操作服务

提供网页界面来控制LeKiwi机器人，左侧显示控制按钮，右侧显示视频流。
使用Flask作为Web服务器，TailwindCSS进行样式设计。
"""

import base64
import json
import time
import threading
from io import BytesIO
from typing import Dict, Any

import cv2
import numpy as np
from flask import Flask, render_template_string, Response, request, jsonify
from flask_cors import CORS

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait

# Flask应用配置
app = Flask(__name__)
CORS(app)

# 全局变量
robot = None
leader_arm = None
keyboard = None
current_action = {}
latest_frame = None
frame_lock = threading.Lock()

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeKiwi 遥操作控制台</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .control-btn {
            transition: all 0.2s ease-in-out;
        }
        .control-btn:active {
            transform: scale(0.95);
            background-color: #1f2937;
        }
        .control-btn.active {
            background-color: #059669;
            color: white;
        }
    </style>
</head>
<body class="bg-gray-100 h-screen overflow-hidden">
    <div class="flex h-full">
        <!-- 左侧控制面板 -->
        <div class="w-1/3 bg-white shadow-lg p-6 flex flex-col">
            <div class="mb-6">
                <h1 class="text-2xl font-bold text-gray-800 mb-2">LeKiwi 控制台</h1>
                <div class="flex space-x-4 text-sm">
                    <span class="px-2 py-1 rounded" id="robot-status">机器人: 连接中...</span>
                    <span class="px-2 py-1 rounded" id="arm-status">机械臂: 连接中...</span>
                </div>
            </div>

            <!-- 速度控制 -->
            <div class="mb-6">
                <h2 class="text-lg font-semibold text-gray-700 mb-3">速度控制</h2>
                <div class="flex space-x-2">
                    <button id="speed-down" class="control-btn flex-1 bg-red-500 text-white py-2 px-4 rounded-lg font-medium">
                        减速 (F)
                    </button>
                    <span id="speed-display" class="flex-1 bg-gray-200 py-2 px-4 rounded-lg text-center font-medium">
                        中速
                    </span>
                    <button id="speed-up" class="control-btn flex-1 bg-green-500 text-white py-2 px-4 rounded-lg font-medium">
                        加速 (R)
                    </button>
                </div>
            </div>

            <!-- 方向控制 -->
            <div class="mb-6">
                <h2 class="text-lg font-semibold text-gray-700 mb-3">方向控制</h2>
                <div class="grid grid-cols-3 gap-3">
                    <div></div>
                    <button id="forward" class="control-btn bg-blue-500 text-white py-3 px-4 rounded-lg font-medium">
                        前进 (W)
                    </button>
                    <div></div>
                    
                    <button id="left" class="control-btn bg-blue-500 text-white py-3 px-4 rounded-lg font-medium">
                        左转 (A)
                    </button>
                    <button id="stop" class="control-btn bg-gray-500 text-white py-3 px-4 rounded-lg font-medium">
                        停止
                    </button>
                    <button id="right" class="control-btn bg-blue-500 text-white py-3 px-4 rounded-lg font-medium">
                        右转 (D)
                    </button>
                    
                    <div></div>
                    <button id="backward" class="control-btn bg-blue-500 text-white py-3 px-4 rounded-lg font-medium">
                        后退 (S)
                    </button>
                    <div></div>
                </div>
            </div>

            <!-- 旋转控制 -->
            <div class="mb-6">
                <h2 class="text-lg font-semibold text-gray-700 mb-3">旋转控制</h2>
                <div class="flex space-x-3">
                    <button id="rotate-left" class="control-btn flex-1 bg-purple-500 text-white py-3 px-4 rounded-lg font-medium">
                        左旋转 (Z)
                    </button>
                    <button id="rotate-right" class="control-btn flex-1 bg-purple-500 text-white py-3 px-4 rounded-lg font-medium">
                        右旋转 (X)
                    </button>
                </div>
            </div>

            <!-- 状态显示 -->
            <div class="mt-auto">
                <h2 class="text-lg font-semibold text-gray-700 mb-3">当前状态</h2>
                <div class="bg-gray-100 p-3 rounded-lg">
                    <div class="text-sm space-y-1">
                        <div>X速度: <span id="x-vel">0.0</span> m/s</div>
                        <div>Y速度: <span id="y-vel">0.0</span> m/s</div>
                        <div>角速度: <span id="theta-vel">0.0</span> deg/s</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 右侧视频显示 -->
        <div class="flex-1 bg-black flex items-center justify-center">
            <div class="relative">
                <img id="video-stream" src="/video_feed" alt="视频流" class="max-w-full max-h-full object-contain">
                <div class="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded text-sm">
                    实时视频流
                </div>
            </div>
        </div>
    </div>

    <script>
        // 键盘控制
        const keys = {
            'w': 'forward',
            's': 'backward', 
            'a': 'left',
            'd': 'right',
            'z': 'rotate-left',
            'x': 'rotate-right',
            'r': 'speed-up',
            'f': 'speed-down'
        };

        const pressedKeys = new Set();
        let activeButtons = new Set();

        // 键盘事件监听
        document.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();
            if (keys[key] && !pressedKeys.has(key)) {
                pressedKeys.add(key);
                const buttonId = keys[key];
                document.getElementById(buttonId)?.classList.add('active');
                activeButtons.add(buttonId);
                sendAction();
            }
        });

        document.addEventListener('keyup', (e) => {
            const key = e.key.toLowerCase();
            if (keys[key]) {
                pressedKeys.delete(key);
                const buttonId = keys[key];
                document.getElementById(buttonId)?.classList.remove('active');
                activeButtons.delete(buttonId);
                sendAction();
            }
        });

        // 鼠标/触摸控制
        function setupButtonControl(buttonId, key) {
            const button = document.getElementById(buttonId);
            if (!button) return;

            button.addEventListener('mousedown', () => {
                pressedKeys.add(key);
                button.classList.add('active');
                activeButtons.add(buttonId);
                sendAction();
            });

            button.addEventListener('mouseup', () => {
                pressedKeys.delete(key);
                button.classList.remove('active');
                activeButtons.delete(buttonId);
                sendAction();
            });

            button.addEventListener('mouseleave', () => {
                pressedKeys.delete(key);
                button.classList.remove('active');
                activeButtons.delete(buttonId);
                sendAction();
            });

            // 触摸事件
            button.addEventListener('touchstart', (e) => {
                e.preventDefault();
                pressedKeys.add(key);
                button.classList.add('active');
                activeButtons.add(buttonId);
                sendAction();
            });

            button.addEventListener('touchend', (e) => {
                e.preventDefault();
                pressedKeys.delete(key);
                button.classList.remove('active');
                activeButtons.delete(buttonId);
                sendAction();
            });
        }

        // 设置所有按钮控制
        Object.entries(keys).forEach(([key, buttonId]) => {
            setupButtonControl(buttonId, key);
        });

        // 停止按钮
        document.getElementById('stop').addEventListener('click', () => {
            pressedKeys.clear();
            activeButtons.forEach(id => {
                document.getElementById(id)?.classList.remove('active');
            });
            activeButtons.clear();
            sendAction();
        });

        // 发送动作到服务器
        function sendAction() {
            const action = {
                keys: Array.from(pressedKeys)
            };
            
            fetch('/send_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(action)
            });
        }

        // 更新状态显示
        function updateStatus() {
            fetch('/get_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('robot-status').textContent = `机器人: ${data.robot_connected ? '已连接' : '未连接'}`;
                    document.getElementById('arm-status').textContent = `机械臂: ${data.arm_connected ? '已连接' : '未连接'}`;
                    document.getElementById('speed-display').textContent = data.speed_level;
                    document.getElementById('x-vel').textContent = data.x_vel.toFixed(2);
                    document.getElementById('y-vel').textContent = data.y_vel.toFixed(2);
                    document.getElementById('theta-vel').textContent = data.theta_vel.toFixed(2);
                });
        }

        // 定期更新状态
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""

def init_robot():
    """初始化机器人连接"""
    global robot, leader_arm, keyboard
    
    # 创建配置
    robot_config = LeKiwiClientConfig(remote_ip="192.168.101.33", id="my_lekiwi")
    teleop_arm_config = SO100LeaderConfig(port="/dev/ttyACM0", id="my_awesome_leader_arm")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    print("=== 连接状态检查 ===")
    print(f"机器人配置: {robot_config}")
    print(f"键盘配置: {keyboard_config}")

    # 连接设备
    print("正在连接机器人...")
    robot.connect()
    print("正在连接机械臂...")
    leader_arm.connect()
    print("正在连接键盘...")
    keyboard.connect()

    print("=== 连接后状态检查 ===")
    print(f"机器人已连接: {robot.is_connected}")
    print(f"机械臂已连接: {leader_arm.is_connected}")
    print(f"键盘已连接: {keyboard.is_connected}")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm or keyboard is not connected!")

def robot_control_loop():
    """机器人控制循环"""
    global current_action, latest_frame
    
    FPS = 30
    loop_count = 0
    
    while True:
        t0 = time.perf_counter()
        loop_count += 1

        try:
            # 获取观察数据
            observation = robot.get_observation()
            
            # 获取机械臂动作
            arm_action = leader_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            # 获取键盘动作
            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)
            
            # 合并动作
            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            
            # 发送动作
            robot.send_action(action)
            
            # 更新全局变量
            current_action = action
            
            # 更新视频帧
            if 'front' in observation:
                with frame_lock:
                    latest_frame = observation['front']
            
            # 控制循环频率
            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
        except Exception as e:
            print(f"控制循环错误: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    # 编码图像为JPEG
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    if ret:
                        frame_data = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(1/30)  # 30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_action', methods=['POST'])
def send_action():
    """接收网页发送的动作"""
    global current_action
    
    try:
        data = request.get_json()
        keys = data.get('keys', [])
        
        # 将按键转换为动作
        pressed_keys = np.array([key in keys for key in ['w', 's', 'a', 'd', 'z', 'x', 'r', 'f']])
        base_action = robot._from_keyboard_to_base_action(pressed_keys)
        
        # 获取机械臂动作
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        
        # 合并动作
        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        
        # 发送动作
        robot.send_action(action)
        current_action = action
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_status')
def get_status():
    """获取机器人状态"""
    try:
        speed_levels = ["低速", "中速", "高速"]
        speed_level = speed_levels[robot.speed_index] if robot else "未知"
        
        x_vel = current_action.get('x.vel', 0.0)
        y_vel = current_action.get('y.vel', 0.0)
        theta_vel = current_action.get('theta.vel', 0.0)
        
        return jsonify({
            "robot_connected": robot.is_connected if robot else False,
            "arm_connected": leader_arm.is_connected if leader_arm else False,
            "speed_level": speed_level,
            "x_vel": x_vel,
            "y_vel": y_vel,
            "theta_vel": theta_vel
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    print("=== LeKiwi 网页遥操作服务 ===")
    print("正在初始化机器人连接...")
    
    try:
        # 初始化机器人
        init_robot()
        
        # 启动控制线程
        control_thread = threading.Thread(target=robot_control_loop, daemon=True)
        control_thread.start()
        
        print("机器人控制线程已启动")
        print("启动Web服务器...")
        print("请在浏览器中访问: http://localhost:5000")
        
        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理资源
        if robot and robot.is_connected:
            robot.disconnect()
        if leader_arm and leader_arm.is_connected:
            leader_arm.disconnect()
        if keyboard and keyboard.is_connected:
            keyboard.disconnect()
        print("服务已关闭")

if __name__ == "__main__":
    main() 