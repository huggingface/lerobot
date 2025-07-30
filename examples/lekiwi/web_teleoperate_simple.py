#!/usr/bin/env python3
"""
LeKiwi 简化网页遥操作服务

简化版本，使用标准库和基本依赖，提供基本的网页控制界面。
"""

import json
import time
import threading
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait

# 全局变量
robot = None
leader_arm = None
keyboard = None
current_action = {}
latest_frame = None
frame_lock = threading.Lock()

# 简单的HTML页面
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>LeKiwi 控制台</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .container { display: flex; height: 90vh; }
        .control-panel { width: 300px; background: white; padding: 20px; border-radius: 10px; margin-right: 20px; }
        .video-panel { flex: 1; background: black; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .control-btn { 
            width: 80px; height: 80px; margin: 5px; border: none; border-radius: 10px; 
            font-size: 14px; font-weight: bold; cursor: pointer; transition: background 0.2s;
        }
        .direction-btn { background: #4CAF50; color: white; }
        .direction-btn:hover { background: #45a049; }
        .direction-btn:active { background: #3d8b40; }
        .speed-btn { background: #2196F3; color: white; }
        .speed-btn:hover { background: #1976D2; }
        .stop-btn { background: #f44336; color: white; }
        .stop-btn:hover { background: #d32f2f; }
        .status { background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 10px 0; }
        .speed-control { display: flex; gap: 5px; margin: 10px 0; }
        .speed-display { flex: 1; background: #ddd; padding: 10px; text-align: center; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>LeKiwi 控制台</h1>
    <div class="container">
        <div class="control-panel">
            <h2>控制面板</h2>
            
            <div class="status">
                <div>机器人状态: <span id="robot-status">连接中...</span></div>
                <div>机械臂状态: <span id="arm-status">连接中...</span></div>
            </div>

            <h3>速度控制</h3>
            <div class="speed-control">
                <button class="speed-btn control-btn" onclick="sendKey('f')">减速</button>
                <div class="speed-display" id="speed-display">中速</div>
                <button class="speed-btn control-btn" onclick="sendKey('r')">加速</button>
            </div>

            <h3>方向控制</h3>
            <div class="grid">
                <div></div>
                <button class="direction-btn control-btn" onmousedown="sendKey('w')" onmouseup="sendKey('stop')">前进</button>
                <div></div>
                <button class="direction-btn control-btn" onmousedown="sendKey('a')" onmouseup="sendKey('stop')">左转</button>
                <button class="stop-btn control-btn" onclick="sendKey('stop')">停止</button>
                <button class="direction-btn control-btn" onmousedown="sendKey('d')" onmouseup="sendKey('stop')">右转</button>
                <div></div>
                <button class="direction-btn control-btn" onmousedown="sendKey('s')" onmouseup="sendKey('stop')">后退</button>
                <div></div>
            </div>

            <h3>旋转控制</h3>
            <div class="grid">
                <button class="direction-btn control-btn" onmousedown="sendKey('z')" onmouseup="sendKey('stop')">左旋转</button>
                <div></div>
                <button class="direction-btn control-btn" onmousedown="sendKey('x')" onmouseup="sendKey('stop')">右旋转</button>
            </div>

            <div class="status">
                <div>X速度: <span id="x-vel">0.0</span> m/s</div>
                <div>Y速度: <span id="y-vel">0.0</span> m/s</div>
                <div>角速度: <span id="theta-vel">0.0</span> deg/s</div>
            </div>
        </div>

        <div class="video-panel">
            <img id="video-stream" src="/video" alt="视频流" style="max-width: 100%; max-height: 100%;">
        </div>
    </div>

    <script>
        let pressedKeys = new Set();

        // 键盘控制
        document.addEventListener('keydown', function(e) {
            const key = e.key.toLowerCase();
            if (['w', 's', 'a', 'd', 'z', 'x', 'r', 'f'].includes(key)) {
                if (!pressedKeys.has(key)) {
                    pressedKeys.add(key);
                    sendKey(key);
                }
            }
        });

        document.addEventListener('keyup', function(e) {
            const key = e.key.toLowerCase();
            if (['w', 's', 'a', 'd', 'z', 'x', 'r', 'f'].includes(key)) {
                pressedKeys.delete(key);
                sendKey('stop');
            }
        });

        function sendKey(key) {
            fetch('/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({key: key})
            });
        }

        // 更新状态
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('robot-status').textContent = data.robot_connected ? '已连接' : '未连接';
                    document.getElementById('arm-status').textContent = data.arm_connected ? '已连接' : '未连接';
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

class LeKiwiHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            
        elif parsed_path.path == '/video':
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            
            with frame_lock:
                if latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    if ret:
                        self.wfile.write(buffer.tobytes())
                    else:
                        # 发送空白图像
                        self.wfile.write(b'')
                else:
                    self.wfile.write(b'')
                    
        elif parsed_path.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                speed_levels = ["低速", "中速", "高速"]
                speed_level = speed_levels[robot.speed_index] if robot else "未知"
                
                x_vel = current_action.get('x.vel', 0.0)
                y_vel = current_action.get('y.vel', 0.0)
                theta_vel = current_action.get('theta.vel', 0.0)
                
                status_data = {
                    "robot_connected": robot.is_connected if robot else False,
                    "arm_connected": leader_arm.is_connected if leader_arm else False,
                    "speed_level": speed_level,
                    "x_vel": x_vel,
                    "y_vel": y_vel,
                    "theta_vel": theta_vel
                }
                
                self.wfile.write(json.dumps(status_data).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        if self.path == '/control':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                key = data.get('key', '')
                
                # 处理按键
                if key == 'stop':
                    # 停止所有动作
                    pressed_keys = np.array([False] * 8)
                else:
                    # 设置对应按键
                    key_map = ['w', 's', 'a', 'd', 'z', 'x', 'r', 'f']
                    pressed_keys = np.array([k == key for k in key_map])
                
                # 转换为机器人动作
                if robot:
                    base_action = robot._from_keyboard_to_base_action(pressed_keys)
                    
                    # 获取机械臂动作
                    if leader_arm:
                        arm_action = leader_arm.get_action()
                        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                    else:
                        arm_action = {}
                    
                    # 合并动作
                    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                    
                    # 发送动作
                    robot.send_action(action)
                    
                    # 更新全局变量
                    global current_action
                    current_action = action
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def log_message(self, format, *args):
        # 减少日志输出
        pass

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
    
    while True:
        t0 = time.perf_counter()

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

def main():
    """主函数"""
    print("=== LeKiwi 简化网页遥操作服务 ===")
    print("正在初始化机器人连接...")
    
    try:
        # 初始化机器人
        init_robot()
        
        # 启动控制线程
        control_thread = threading.Thread(target=robot_control_loop, daemon=True)
        control_thread.start()
        
        print("机器人控制线程已启动")
        print("启动Web服务器...")
        print("请在浏览器中访问: http://localhost:8080")
        
        # 启动HTTP服务器
        server = HTTPServer(('0.0.0.0', 8080), LeKiwiHTTPHandler)
        server.serve_forever()
        
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