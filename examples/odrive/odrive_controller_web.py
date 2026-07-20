#!/usr/bin/env python3
"""
ODrive USB Web 控制器
基于 odrive_controller.py 的控制逻辑，提供Web界面控制
"""
import odrive
from odrive.enums import *
import time
import threading
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS

from odrive_controller import ODriveController

# Flask app
app = Flask(__name__)
CORS(app)

# 全局变量
robot = None
running = True
command_lock = threading.Lock()

# 速度参数（与 odrive_controller.py 保持一致）
base_speed = 0.3  # 基础速度 (m/s)
turn_speed = 0.2  # 转向速度 (m/s)
speed_step = 0.05  # 速度调节步长

# 全局变量 - 当前控制状态
control_state = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False,
    "stop": False
}

# HTML模板 - 带虚拟摇杆和按键控制
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ODrive Web Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .status-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .status-item h3 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }

        .status-item .value {
            font-size: 1.4em;
            font-weight: bold;
            color: #333;
        }

        .connected { color: #28a745; }
        .disconnected { color: #dc3545; }

        .control-section {
            margin-top: 30px;
        }

        .control-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
        }

        .joystick-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 30px 0;
            flex-direction: column;
        }

        .joystick-wrapper {
            position: relative;
            width: 250px;
            height: 250px;
            margin-bottom: 20px;
        }

        .joystick-base {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%);
            box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);
            border: 3px solid #667eea;
        }

        .joystick-crosshair {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
        }

        .joystick-crosshair::before,
        .joystick-crosshair::after {
            content: '';
            position: absolute;
            background: rgba(102, 126, 234, 0.2);
        }

        .joystick-crosshair::before {
            width: 2px;
            height: 100%;
            left: 50%;
            transform: translateX(-50%);
        }

        .joystick-crosshair::after {
            width: 100%;
            height: 2px;
            top: 50%;
            transform: translateY(-50%);
        }

        .joystick-stick {
            position: absolute;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            cursor: grab;
            transition: transform 0.1s;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            touch-action: none;
            user-select: none;
        }

        .joystick-stick:active {
            cursor: grabbing;
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        }

        .joystick-stick::after {
            content: '⊕';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 32px;
            color: white;
            font-weight: bold;
        }

        .joystick-info {
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-size: 14px;
        }

        .speed-control {
            margin-top: 30px;
        }

        .speed-control-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            align-items: center;
        }

        .speed-btn {
            padding: 12px 24px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.15s;
        }

        .speed-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .speed-btn:active {
            transform: translateY(0);
        }

        .speed-display {
            padding: 12px 24px;
            background: #f8f9fa;
            border: 2px solid #667eea;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
            min-width: 120px;
            text-align: center;
        }

        .keyboard-control {
            margin-top: 30px;
        }

        .keyboard-grid {
            display: grid;
            grid-template-columns: repeat(3, 100px);
            grid-template-rows: repeat(3, 100px);
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }

        .control-btn {
            border: 3px solid #667eea;
            background: white;
            border-radius: 10px;
            font-size: 2em;
            cursor: pointer;
            transition: all 0.1s;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: #667eea;
        }

        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .control-btn:active,
        .control-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: translateY(0);
        }

        .control-btn.stop {
            grid-column: 2;
            grid-row: 2;
        }

        .info-panel {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .info-panel h3 {
            color: #2196F3;
            margin-bottom: 10px;
        }

        .info-panel ul {
            list-style-position: inside;
            color: #555;
        }

        .info-panel li {
            margin: 5px 0;
        }

        @media (max-width: 600px) {
            h1 {
                font-size: 1.8em;
            }

            .keyboard-grid {
                grid-template-columns: repeat(3, 80px);
                grid-template-rows: repeat(3, 80px);
                gap: 10px;
            }

            .control-btn {
                font-size: 24px;
            }

            .speed-control-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ODrive Web Controller</h1>

        <div class="card">
            <h2 style="color: #667eea; margin-bottom: 20px;">📊 Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <h3>Connection</h3>
                    <div class="value" id="status">Connecting...</div>
                </div>
                <div class="status-item">
                    <h3>Base Speed (m/s)</h3>
                    <div class="value" id="base-speed-display">0.30</div>
                </div>
                <div class="status-item">
                    <h3>Left Wheel (RPM)</h3>
                    <div class="value" id="left-rpm">0.0</div>
                </div>
                <div class="status-item">
                    <h3>Right Wheel (RPM)</h3>
                    <div class="value" id="right-rpm">0.0</div>
                </div>
                <div class="status-item">
                    <h3>Voltage (V)</h3>
                    <div class="value" id="voltage">0.0</div>
                </div>
                <div class="status-item">
                    <h3>Left Speed (m/s)</h3>
                    <div class="value" id="left-speed">0.00</div>
                </div>
                <div class="status-item">
                    <h3>Right Speed (m/s)</h3>
                    <div class="value" id="right-speed">0.00</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="control-section">
                <h2>🎮 Joystick Control</h2>
                <div class="joystick-container">
                    <div class="joystick-wrapper">
                        <div class="joystick-base"></div>
                        <div class="joystick-crosshair"></div>
                        <div class="joystick-stick" id="joystick-stick"></div>
                    </div>
                    <div class="joystick-info">
                        <strong>上下:</strong> 前进/后退 | <strong>左右:</strong> 左转/右转
                    </div>
                </div>
            </div>

            <div class="keyboard-control">
                <h2>⌨️ Keyboard Control</h2>
                <div class="keyboard-grid">
                    <div></div>
                    <button class="control-btn" id="btn-forward" onmousedown="setControl('forward', true)" onmouseup="setControl('forward', false)" ontouchstart="setControl('forward', true)" ontouchend="setControl('forward', false)">↑</button>
                    <div></div>
                    <button class="control-btn" id="btn-left" onmousedown="setControl('left', true)" onmouseup="setControl('left', false)" ontouchstart="setControl('left', true)" ontouchend="setControl('left', false)">←</button>
                    <button class="control-btn stop" id="btn-stop" onmousedown="setControl('stop', true)" onmouseup="setControl('stop', false)" ontouchstart="setControl('stop', true)" ontouchend="setControl('stop', false)">■</button>
                    <button class="control-btn" id="btn-right" onmousedown="setControl('right', true)" onmouseup="setControl('right', false)" ontouchstart="setControl('right', true)" ontouchend="setControl('right', false)">→</button>
                    <div></div>
                    <button class="control-btn" id="btn-backward" onmousedown="setControl('backward', true)" onmouseup="setControl('backward', false)" ontouchstart="setControl('backward', true)" ontouchend="setControl('backward', false)">↓</button>
                    <div></div>
                </div>
            </div>

            <div class="speed-control">
                <h2>⚡ Speed Control</h2>
                <div class="speed-control-buttons">
                    <button class="speed-btn" onclick="adjustSpeed(-1)">−</button>
                    <div class="speed-display" id="speed-display">0.30 m/s</div>
                    <button class="speed-btn" onclick="adjustSpeed(1)">+</button>
                    <button class="speed-btn" onclick="resetSpeed()">Reset</button>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="info-panel">
                <h3>💡 Control Instructions</h3>
                <ul>
                    <li><strong>Joystick:</strong> Drag to control direction - up/down for forward/back, left/right for rotation</li>
                    <li><strong>Keyboard Buttons:</strong> Click or touch to control - W/↑ forward, S/↓ backward, A/← left turn, D/→ right turn</li>
                    <li><strong>Speed Control:</strong> Use +/- buttons to adjust base speed (0.05-4.0 m/s), or Reset to default (0.30 m/s)</li>
                    <li><strong>Turn Speed:</strong> Fixed at 0.20 m/s for pure rotation</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        let baseSpeed = 0.30;
        let joystickActive = false;
        let joystickX = 0; // -1 to 1 (左到右)
        let joystickY = 0; // -1 to 1 (上到下，但我们会反转用于前后)
        let controlState = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            stop: false
        };

        // 摇杆控制
        const joystickStick = document.getElementById('joystick-stick');
        const joystickWrapper = joystickStick.parentElement;
        const maxDistance = 85; // 最大偏移距离（像素）

        function updateJoystick(x, y) {
            // 限制在圆形范围内
            const distance = Math.sqrt(x * x + y * y);
            if (distance > maxDistance) {
                const angle = Math.atan2(y, x);
                x = Math.cos(angle) * maxDistance;
                y = Math.sin(angle) * maxDistance;
            }

            // 更新摇杆位置
            joystickStick.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
            joystickStick.style.transition = joystickActive ? 'none' : 'transform 0.2s';

            // 计算归一化值 (-1 to 1)
            joystickX = x / maxDistance;
            joystickY = -y / maxDistance; // 反转Y轴（上为正）

            // 发送到服务器
            sendJoystickState();
        }

        function resetJoystick() {
            updateJoystick(0, 0);
            joystickActive = false;
        }

        function sendJoystickState() {
            fetch('/joystick', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: joystickX, y: joystickY})
            });
        }

        // 鼠标事件
        joystickStick.addEventListener('mousedown', (e) => {
            joystickActive = true;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!joystickActive) return;

            const rect = joystickWrapper.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            updateJoystick(e.clientX - centerX, e.clientY - centerY);
        });

        document.addEventListener('mouseup', () => {
            if (joystickActive) {
                resetJoystick();
            }
        });

        // 触摸事件
        joystickStick.addEventListener('touchstart', (e) => {
            joystickActive = true;
            e.preventDefault();
        });

        document.addEventListener('touchmove', (e) => {
            if (!joystickActive) return;

            const touch = e.touches[0];
            const rect = joystickWrapper.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            updateJoystick(touch.clientX - centerX, touch.clientY - centerY);
            e.preventDefault();
        });

        document.addEventListener('touchend', () => {
            if (joystickActive) {
                resetJoystick();
            }
        });

        // 键盘按钮控制
        function setControl(direction, pressed) {
            controlState[direction] = pressed;
            updateButtonState(direction, pressed);
            sendControlState();
        }

        function updateButtonState(direction, pressed) {
            const btn = document.getElementById('btn-' + direction);
            if (btn) {
                if (pressed) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            }
        }

        function sendControlState() {
            fetch('/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(controlState)
            });
        }

        // 速度控制
        function adjustSpeed(delta) {
            baseSpeed = Math.max(0.05, Math.min(4.0, baseSpeed + delta * 0.05));
            updateSpeedDisplay();
            fetch('/speed', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({speed: baseSpeed})
            });
        }

        function resetSpeed() {
            baseSpeed = 0.30;
            updateSpeedDisplay();
            fetch('/speed', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({speed: baseSpeed})
            });
        }

        function updateSpeedDisplay() {
            document.getElementById('speed-display').textContent = baseSpeed.toFixed(2) + ' m/s';
        }

        // 键盘事件（支持物理键盘）
        document.addEventListener('keydown', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w':
                case 'arrowup':
                    setControl('forward', true);
                    break;
                case 's':
                case 'arrowdown':
                    setControl('backward', true);
                    break;
                case 'a':
                case 'arrowleft':
                    setControl('left', true);
                    break;
                case 'd':
                case 'arrowright':
                    setControl('right', true);
                    break;
                case ' ':
                    e.preventDefault();
                    setControl('stop', true);
                    break;
            }
        });

        document.addEventListener('keyup', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w':
                case 'arrowup':
                    setControl('forward', false);
                    break;
                case 's':
                case 'arrowdown':
                    setControl('backward', false);
                    break;
                case 'a':
                case 'arrowleft':
                    setControl('left', false);
                    break;
                case 'd':
                case 'arrowright':
                    setControl('right', false);
                    break;
                case ' ':
                    setControl('stop', false);
                    break;
            }
        });

        // 更新状态
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Connection status
                    const statusEl = document.getElementById('status');
                    if (data.connected) {
                        statusEl.textContent = 'Connected';
                        statusEl.className = 'value connected';
                    } else {
                        statusEl.textContent = 'Disconnected';
                        statusEl.className = 'value disconnected';
                    }

                    // Speed
                    document.getElementById('base-speed-display').textContent = data.base_speed.toFixed(2);
                    document.getElementById('left-rpm').textContent = data.left_rpm.toFixed(1);
                    document.getElementById('right-rpm').textContent = data.right_rpm.toFixed(1);
                    document.getElementById('voltage').textContent = data.voltage.toFixed(1);
                    document.getElementById('left-speed').textContent = data.left_speed.toFixed(2);
                    document.getElementById('right-speed').textContent = data.right_speed.toFixed(2);
                })
                .catch(err => console.error('Status error:', err));
        }

        // 定期更新状态
        setInterval(updateStatus, 200);
        updateStatus();
    </script>
</body>
</html>
"""


def calculate_velocities(forward, backward, left, right, stop, base_speed, turn_speed):
    """
    计算左右轮速度 - 完全按照 odrive_controller.py 的逻辑
    """
    left_speed = 0.0
    right_speed = 0.0
    
    if stop:
        left_speed = 0.0
        right_speed = 0.0
    elif forward and backward:
        # 同时按下前进和后退，停止
        left_speed = 0.0
        right_speed = 0.0
    elif forward:
        if left:
            # 前进+左转 = 左前（左轮更慢）
            left_speed = -base_speed * 0.5
            right_speed = base_speed
        elif right:
            # 前进+右转 = 右前（右轮更慢）
            left_speed = -base_speed
            right_speed = base_speed * 0.5
        else:
            # 纯前进（左轮用负数，因为电机方向接反）
            left_speed = -base_speed
            right_speed = base_speed
    elif backward:
        if left:
            # 后退+左转 = 左后（左轮更慢）
            left_speed = base_speed * 0.5
            right_speed = -base_speed
        elif right:
            # 后退+右转 = 右后（右轮更慢）
            left_speed = base_speed
            right_speed = -base_speed * 0.5
        else:
            # 纯后退（左轮用正数，右轮用负数）
            left_speed = base_speed
            right_speed = -base_speed
    elif left:
        # 纯左转（原地）- 左轮向后，右轮向前
        left_speed = turn_speed
        right_speed = turn_speed
    elif right:
        # 纯右转（原地）- 左轮向前，右轮向后
        left_speed = -turn_speed
        right_speed = -turn_speed
    
    return left_speed, right_speed


def calculate_velocities_from_joystick(joystick_x, joystick_y, base_speed, turn_speed):
    """
    从摇杆输入计算左右轮速度
    joystick_x: -1 to 1 (左到右)
    joystick_y: -1 to 1 (下到上)
    """
    deadzone = 0.05
    
    # 检查是否有输入
    if abs(joystick_x) < deadzone and abs(joystick_y) < deadzone:
        return 0.0, 0.0
    
    # 计算前进/后退和左转/右转
    forward = joystick_y > deadzone
    backward = joystick_y < -deadzone
    left = joystick_x < -deadzone
    right = joystick_x > deadzone
    
    # 计算速度大小（基于摇杆距离中心的距离）
    forward_magnitude = abs(joystick_y) if forward else 0
    backward_magnitude = abs(joystick_y) if backward else 0
    left_magnitude = abs(joystick_x) if left else 0
    right_magnitude = abs(joystick_x) if right else 0
    
    # 计算实际速度（考虑摇杆幅度）
    effective_base_speed = base_speed * max(forward_magnitude, backward_magnitude)
    effective_turn_speed = turn_speed * max(left_magnitude, right_magnitude)
    
    # 使用相同的逻辑计算速度
    left_speed = 0.0
    right_speed = 0.0
    
    if forward and backward:
        left_speed = 0.0
        right_speed = 0.0
    elif forward:
        if left:
            left_speed = -effective_base_speed * 0.5
            right_speed = effective_base_speed
        elif right:
            left_speed = -effective_base_speed
            right_speed = effective_base_speed * 0.5
        else:
            left_speed = -effective_base_speed
            right_speed = effective_base_speed
    elif backward:
        if left:
            left_speed = effective_base_speed * 0.5
            right_speed = -effective_base_speed
        elif right:
            left_speed = effective_base_speed
            right_speed = -effective_base_speed * 0.5
        else:
            left_speed = effective_base_speed
            right_speed = -effective_base_speed
    elif left:
        left_speed = -effective_turn_speed
        right_speed = -effective_turn_speed
    elif right:
        left_speed = effective_turn_speed
        right_speed = effective_turn_speed
    
    return left_speed, right_speed


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/status')
def get_status():
    """获取状态"""
    global base_speed
    
    if robot:
        try:
            left_rpm, right_rpm = robot.get_velocity_rpm()
            voltage = robot.get_voltage()
            # 计算当前设置的速度（从RPM反推）
            left_speed = left_rpm / 60 * robot.wheel_circumference
            right_speed = right_rpm / 60 * robot.wheel_circumference
        except:
            left_rpm = right_rpm = 0.0
            voltage = 0.0
            left_speed = right_speed = 0.0
    else:
        left_rpm = right_rpm = 0.0
        voltage = 0.0
        left_speed = right_speed = 0.0
    
    return jsonify({
        'connected': robot is not None,
        'base_speed': base_speed,
        'left_rpm': left_rpm,
        'right_rpm': right_rpm,
        'voltage': voltage,
        'left_speed': left_speed,
        'right_speed': right_speed
    })


@app.route('/control', methods=['POST'])
def control():
    """按键控制"""
    global control_state
    
    data = request.json
    with command_lock:
        control_state.update({
            'forward': bool(data.get('forward', False)),
            'backward': bool(data.get('backward', False)),
            'left': bool(data.get('left', False)),
            'right': bool(data.get('right', False)),
            'stop': bool(data.get('stop', False))
        })
    
    return jsonify({'success': True})


@app.route('/joystick', methods=['POST'])
def joystick():
    """摇杆输入"""
    data = request.json
    joy_x = float(data.get('x', 0))
    joy_y = float(data.get('y', 0))
    
    # 将摇杆输入转换为控制状态
    deadzone = 0.05
    with command_lock:
        control_state.update({
            'forward': joy_y > deadzone,
            'backward': joy_y < -deadzone,
            'left': joy_x < -deadzone,
            'right': joy_x > deadzone,
            'stop': False
        })
    
    return jsonify({'success': True})


@app.route('/speed', methods=['POST'])
def set_speed():
    """设置速度"""
    global base_speed
    
    data = request.json
    new_speed = float(data.get('speed', 0.3))
    base_speed = max(0.05, min(4.0, new_speed))
    
    return jsonify({'success': True, 'speed': base_speed})


def robot_control_loop(fps=30):
    """机器人控制循环"""
    global running, base_speed, control_state
    
    dt = 1.0 / fps
    print(f"[CONTROL] Robot control loop started at {fps} Hz")
    
    while running:
        try:
            if robot:
                with command_lock:
                    # 获取控制状态
                    forward = control_state['forward']
                    backward = control_state['backward']
                    left = control_state['left']
                    right = control_state['right']
                    stop = control_state['stop']
                
                # 计算速度
                left_speed, right_speed = calculate_velocities(
                    forward, backward, left, right, stop,
                    base_speed, turn_speed
                )
                
                # 设置速度
                robot.set_velocity(left_speed, right_speed)
            
            time.sleep(dt)
        
        except Exception as e:
            print(f"[CONTROL] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    print("[CONTROL] Robot control loop stopped")


def main():
    global robot, running, base_speed
    
    # 配置
    HOST = '0.0.0.0'
    PORT = 5001  # 使用不同的端口避免冲突
    FPS = 30
    
    # 初始化机器人
    print("[MAIN] Initializing ODrive controller...")
    try:
        robot = ODriveController()
        print("[MAIN] ✓ Successfully connected to ODrive")
    except Exception as e:
        print(f"[MAIN] ✗ Failed to connect to ODrive: {e}")
        print("[MAIN] Starting server anyway (robot will be unavailable)...")
        robot = None
    
    # 启动控制线程
    control_thread = threading.Thread(target=robot_control_loop, args=(FPS,), daemon=True)
    control_thread.start()
    print("[MAIN] ✓ Robot control thread started")
    
    # 获取本地IP
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostname()
        # 尝试获取实际IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print("\n" + "="*70)
    print("🤖 ODrive Web Controller Server")
    print("="*70)
    print(f"📱 Local:   http://localhost:{PORT}")
    print(f"🌐 Network: http://{local_ip}:{PORT}")
    print("="*70)
    if robot:
        print(f"⚙️  Robot Config:")
        print(f"   Wheel Diameter: {robot.wheel_diameter:.3f}m")
        print(f"   Wheel Circumference: {robot.wheel_circumference:.3f}m")
        print(f"   Base Speed: {base_speed:.2f} m/s")
        print(f"   Turn Speed: {turn_speed:.2f} m/s")
    print("="*70)
    print("Press Ctrl+C to stop\n")
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
    finally:
        running = False
        if robot:
            robot.stop()
            time.sleep(0.5)
            robot.shutdown()
        print("[MAIN] ✓ Cleanup complete")


if __name__ == "__main__":
    main()

