#!/usr/bin/env python3
"""
LeKiwi Flask 遥操作服务器
提供网页界面来控制LeKiwi机器人和SO100机械臂
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait

app = Flask(__name__)
CORS(app)

# 全局变量
robot: Optional[LeKiwiClient] = None
leader_arm: Optional[SO100Leader] = None
latest_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()
robot_status = {
    "robot_connected": False,
    "arm_connected": False,
    "speed_level": "中速",
    "x_vel": 0.0,
    "y_vel": 0.0,
    "theta_vel": 0.0
}

# 速度等级配置
speed_levels = [
    {"xy": 0.1, "theta": 30, "name": "慢速"},  # 慢速
    {"xy": 0.2, "theta": 60, "name": "中速"},  # 中速
    {"xy": 0.3, "theta": 90, "name": "快速"},  # 快速
]
current_speed_index = 1  # 默认中速

# 键盘映射
keyboard_mapping = {
    "w": "forward",
    "s": "backward", 
    "a": "left",
    "d": "right",
    "z": "rotate_left",
    "x": "rotate_right",
    "r": "speed_up",
    "f": "speed_down"
}

# 当前按键状态
pressed_keys = set()

def init_robot(remote_ip: str, arm_port: str):
    """初始化机器人和机械臂"""
    global robot, leader_arm
    
    try:
        # 创建配置
        robot_config = LeKiwiClientConfig(remote_ip=remote_ip, id="my_lekiwi")
        arm_config = SO100LeaderConfig(port=arm_port, id="my_awesome_leader_arm")
        
        # 创建实例
        robot = LeKiwiClient(robot_config)
        leader_arm = SO100Leader(arm_config)
        
        # 连接设备
        print("正在连接机器人...")
        robot.connect()
        robot_status["robot_connected"] = True
        
        print("正在连接机械臂...")
        leader_arm.connect()
        robot_status["arm_connected"] = True
        
        print("所有设备连接成功！")
        return True
        
    except Exception as e:
        print(f"连接失败: {e}")
        return False

def _from_keyboard_to_base_action(keys: set) -> Dict[str, float]:
    """将键盘按键转换为机器人动作"""
    if not robot:
        return {}
    
    action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    
    # 获取当前速度等级
    speed_config = speed_levels[current_speed_index]
    xy_speed = speed_config["xy"]
    theta_speed = speed_config["theta"]
    
    # 映射按键到动作
    if "stop" in keys:
        # 停止动作：所有速度设为0
        action["x.vel"] = 0.0
        action["y.vel"] = 0.0
        action["theta.vel"] = 0.0
    else:
        # 正常移动动作
        if "forward" in keys:
            action["x.vel"] = xy_speed
        if "backward" in keys:
            action["x.vel"] = -xy_speed
        if "left" in keys:
            action["y.vel"] = xy_speed
        if "right" in keys:
            action["y.vel"] = -xy_speed
        if "rotate_left" in keys:
            action["theta.vel"] = theta_speed
        if "rotate_right" in keys:
            action["theta.vel"] = -theta_speed
    
    return action




def video_stream_thread():
    """视频流线程"""
    global latest_frame, robot_status
    
    while True:
        try:
            if robot and robot.is_connected:
                # 获取观察数据
                observation = robot.get_observation()
                
                # 更新状态
                robot_status["x_vel"] = observation.get("x.vel", 0.0)
                robot_status["y_vel"] = observation.get("y.vel", 0.0)
                robot_status["theta_vel"] = observation.get("theta.vel", 0.0)
                
                # 获取手腕摄像头视频帧
                if "wrist" in observation:
                    with frame_lock:
                        latest_frame = observation["wrist"].copy()
                        # 顺时针旋转90度
                        latest_frame = cv2.rotate(latest_frame, cv2.ROTATE_90_CLOCKWISE)
                        # 转换颜色空间：BGR -> RGB
                        latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                    # print(f"获取到手腕视频帧，形状: {latest_frame.shape}")
                else:
                    print("警告: 观察数据中没有wrist视频帧")
                    print(f"可用的观察数据键: {list(observation.keys())}")
            else:
                # 机器人未连接时，清空视频帧
                with frame_lock:
                    latest_frame = None
                
            time.sleep(1/30)  # 30 FPS
            
        except Exception as e:
            print(f"视频流错误: {e}")
            # 出错时也清空视频帧
            with frame_lock:
                latest_frame = None
            time.sleep(1)

def control_thread():
    """控制线程"""
    global pressed_keys, current_speed_index
    
    while True:
        try:
            if robot and robot.is_connected:
                # 处理速度控制
                if "speed_up" in pressed_keys:
                    current_speed_index = min(current_speed_index + 1, len(speed_levels) - 1)
                    pressed_keys.discard("speed_up")
                if "speed_down" in pressed_keys:
                    current_speed_index = max(current_speed_index - 1, 0)
                    pressed_keys.discard("speed_down")
                
                # 更新速度等级显示
                robot_status["speed_level"] = speed_levels[current_speed_index]["name"]
                
                # 获取机械臂动作
                arm_action = {}
                if leader_arm and leader_arm.is_connected:
                    try:
                        arm_action = leader_arm.get_action()
                        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                    except Exception as e:
                        print(f"获取机械臂动作失败: {e}")
                
                # 获取键盘动作
                base_action = _from_keyboard_to_base_action(pressed_keys)
                
                # 合并动作
                action = {**arm_action, **base_action}
                
                # 发送动作到机器人
                if action:
                    robot.send_action(action)
            
            time.sleep(1/30)  # 30 FPS
            
        except Exception as e:
            print(f"控制线程错误: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """主页"""
    return render_template('lekiwi_control.html')

@app.route('/connect', methods=['POST'])
def connect():
    """连接机器人"""
    data = request.get_json()
    remote_ip = data.get('remote_ip', '192.168.101.33')
    arm_port = data.get('arm_port', '/dev/ttyACM0')
    
    success = init_robot(remote_ip, arm_port)
    
    if success:
        # 启动控制线程（视频流线程已在服务器启动时运行）
        threading.Thread(target=control_thread, daemon=True).start()
    
    return jsonify({"success": success})

@app.route('/send_action', methods=['POST'])
def send_action():
    """接收前端发送的动作"""
    global pressed_keys
    
    data = request.get_json()
    keys = data.get('keys', [])
    print(f"接收到的按键: {keys}")
    
    # 将原始按键映射为动作名称
    mapped_keys = set()
    for key in keys:
        if key in keyboard_mapping:
            mapped_keys.add(keyboard_mapping[key])
        else:
            mapped_keys.add(key)  # 保留未映射的按键
    
    # 更新按键状态
    pressed_keys = mapped_keys
    
    return jsonify({"success": True})

@app.route('/get_status')
def get_status():
    """获取机器人状态"""
    return jsonify(robot_status)

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
                else:
                    # 当没有视频帧时，生成一个占位图像
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    # 添加文字
                    cv2.putText(placeholder, "No Video Feed", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(placeholder, "Please connect robot", (180, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_data = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(1/30)  # 30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/disconnect', methods=['POST'])
def disconnect():
    """断开连接"""
    global robot, leader_arm, robot_status
    
    try:
        if robot:
            robot.disconnect()
            robot = None
        if leader_arm:
            leader_arm.disconnect()
            leader_arm = None
        
        robot_status["robot_connected"] = False
        robot_status["arm_connected"] = False
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # 禁用Flask的日志输出
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print("启动LeKiwi Flask遥操作服务器...")
    print("请在浏览器中访问: http://localhost:5555")
    
    # 启动视频流线程（在服务器启动时就运行）
    threading.Thread(target=video_stream_thread, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5555, debug=True, threaded=True) 