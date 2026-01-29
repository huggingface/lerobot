import logging
import time
import signal
import sys
import threading
import json
import numpy as np
import cv2
from flask import Flask, jsonify, render_template_string, Response, request
from dataclasses import asdict

import lerobot
from lerobot.robots.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1
from lerobot.cameras.zmq import ZMQCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.processor import RobotAction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
robot = None
HEAD_CAMERA = None

# HTML Template for the Dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Unitree G1 Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; }
        h1 { text-align: center; color: #333; }
        .status-bar { text-align: center; margin-bottom: 20px; padding: 10px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
        .chart-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 15px; }
        .chart { width: 100%; height: 300px; }
        .video-card { grid-column: span 2; display: flex; justify-content: center; align-items: center; background: #000; }
        .video-img { max-width: 100%; max-height: 480px; }
        .control-panel { grid-column: span 2; background: #fff; border-radius: 8px; padding: 15px; }
        .slider-container { display: flex; align-items: center; margin-bottom: 5px; }
        .slider-label { width: 150px; font-size: 12px; }
        .slider-input { flex-grow: 1; }
        .slider-value { width: 50px; text-align: right; font-size: 12px; margin-left: 10px; }
        h3 { margin-top: 0; }
        .hand-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    </style>
</head>
<body>
    <h1>Unitree G1 Observations & Control</h1>
    <div class="status-bar">
        Status: <span id="connection-status" style="font-weight:bold; color:orange;">Connecting...</span> | 
        Update Rate: <span id="fps">0</span> Hz
    </div>
    
    <div class="grid-container">
        <!-- Camera Feed -->
        <div class="chart-card video-card">
            <img src="/video_feed" class="video-img" alt="Camera Feed (Waiting for connection...)">
        </div>

        <!-- Hand Controls -->
        <div class="control-panel">
            <h3>Hand Control (Teleoperation)</h3>
            <div class="hand-controls">
                <div id="left-hand-controls">
                    <h4>Left Hand</h4>
                    <!-- Sliders injected by JS -->
                </div>
                <div id="right-hand-controls">
                    <h4>Right Hand</h4>
                    <!-- Sliders injected by JS -->
                </div>
            </div>
            <button onclick="resetHands()" style="margin-top:10px; padding: 5px 15px;">Reset Hands</button>
        </div>

        <div class="chart-card">
            <h3>Left Hand Joints (q)</h3>
            <div id="left_hand_q" class="chart"></div>
        </div>
        <div class="chart-card">
            <h3>Right Hand Joints (q)</h3>
            <div id="right_hand_q" class="chart"></div>
        </div>

        <div class="chart-card">
            <h3>Body Motor Positions (q)</h3>
            <div id="motor_q" class="chart"></div>
        </div>
        <div class="chart-card">
            <h3>Body Motor Velocities (dq)</h3>
            <div id="motor_dq" class="chart"></div>
        </div>
        <div class="chart-card">
            <h3>Body Motor Torque (tau)</h3>
            <div id="motor_tau" class="chart"></div>
        </div>
        <div class="chart-card">
            <h3>IMU Accelerometer</h3>
            <div id="imu_accel" class="chart"></div>
        </div>
        <div class="chart-card">
            <h3>IMU Gyroscope</h3>
            <div id="imu_gyro" class="chart"></div>
        </div>
        <div class="chart-card">
            <h3>IMU Orientation (RPY)</h3>
            <div id="imu_rpy" class="chart"></div>
        </div>
    </div>

    <script>
        const layoutConfig = {
            margin: { t: 30, r: 20, l: 40, b: 80 },
            font: { size: 10 },
            xaxis: { automargin: true }
        };

        Plotly.newPlot('motor_q', [{x: [], y: [], type: 'bar'}], {title: 'Body Joint Positions', ...layoutConfig});
        Plotly.newPlot('motor_dq', [{x: [], y: [], type: 'bar'}], {title: 'Body Joint Velocities', ...layoutConfig});
        Plotly.newPlot('motor_tau', [{x: [], y: [], type: 'bar'}], {title: 'Body Est. Torque', ...layoutConfig});
        
        // Hand charts with fixed Y-axis to show full Dex3 joint range (-1.75 to 1.75 rad)
        const handLayoutConfig = {...layoutConfig, yaxis: {range: [-1.8, 1.8], title: 'rad'}};
        Plotly.newPlot('left_hand_q', [{x: [], y: [], type: 'bar', marker:{color:'orange'}}], {title: 'Left Hand Positions', ...handLayoutConfig});
        Plotly.newPlot('right_hand_q', [{x: [], y: [], type: 'bar', marker:{color:'orange'}}], {title: 'Right Hand Positions', ...handLayoutConfig});

        Plotly.newPlot('imu_accel', [
            {y: [], mode: 'lines+markers', name: 'X'}, 
            {y: [], mode: 'lines+markers', name: 'Y'}, 
            {y: [], mode: 'lines+markers', name: 'Z'}
        ], {title: 'Linear Acceleration (m/s²)', ...layoutConfig});
        Plotly.newPlot('imu_gyro', [
            {y: [], mode: 'lines+markers', name: 'X'}, 
            {y: [], mode: 'lines+markers', name: 'Y'}, 
            {y: [], mode: 'lines+markers', name: 'Z'}
        ], {title: 'Angular Velocity (rad/s)', ...layoutConfig});
        Plotly.newPlot('imu_rpy', [
            {y: [], mode: 'lines+markers', name: 'Roll'}, 
            {y: [], mode: 'lines+markers', name: 'Pitch'}, 
            {y: [], mode: 'lines+markers', name: 'Yaw'}
        ], {title: 'Roll Pitch Yaw (rad)', ...layoutConfig});

        let lastTime = Date.now();
        let frameCount = 0;
        let controlsInitialized = false;
        
        // Hand Joint Names (Standard Dex3 ordering)
        const leftHandJoints = [
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint"
        ];
        const rightHandJoints = [
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint"
        ];

        function initControls() {
            const createSlider = (id, label) => `
                <div class="slider-container">
                    <span class="slider-label">${label}</span>
                    <input type="range" min="-1.57" max="1.57" step="0.01" value="0" class="slider-input" id="slider_${id}" oninput="sendAction('${id}', this.value)">
                    <span class="slider-value" id="val_${id}">0.00</span>
                </div>`;

            const leftContainer = document.getElementById('left-hand-controls');
            leftContainer.innerHTML = '<h4>Left Hand</h4>' + leftHandJoints.map(j => createSlider(j, j.replace('left_hand_', ''))).join('');

            const rightContainer = document.getElementById('right-hand-controls');
            rightContainer.innerHTML = '<h4>Right Hand</h4>' + rightHandJoints.map(j => createSlider(j, j.replace('right_hand_', ''))).join('');
            
            controlsInitialized = true;
        }

        function sendAction(joint, value) {
            document.getElementById(`val_${joint}`).innerText = value;
            fetch('/set_action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[joint + '.q']: parseFloat(value)})
            }).catch(console.error);
        }
        
        function resetHands() {
            const action = {};
            [...leftHandJoints, ...rightHandJoints].forEach(j => {
                action[j + '.q'] = 0.0;
                const slider = document.getElementById(`slider_${j}`);
                if(slider) {
                    slider.value = 0;
                    document.getElementById(`val_${j}`).innerText = "0.00";
                }
            });
            
            fetch('/set_action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(action)
            }).catch(console.error);
        }

        function updateCharts(data) {
            frameCount++;
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').innerText = frameCount;
                frameCount = 0;
                lastTime = now;
            }
            document.getElementById('connection-status').innerText = "Connected";
            document.getElementById('connection-status').style.color = "green";
            
            if (!controlsInitialized) initControls();

            // Filter Joints
            const allJointKeys = Object.keys(data).filter(k => k.endsWith('.q'));
            
            // Separate Hand and Body
            const isHand = k => k.includes('left_hand') || k.includes('right_hand');
            
            const bodyKeys = allJointKeys.filter(k => !isHand(k)).sort();
            const leftKeys = allJointKeys.filter(k => k.includes('left_hand')).sort();
            const rightKeys = allJointKeys.filter(k => k.includes('right_hand')).sort();

            // Update Body Charts
            Plotly.animate('motor_q', {
                data: [{x: bodyKeys.map(k=>k.replace('.q','')), y: bodyKeys.map(k=>data[k])}]
            }, {transition: {duration: 0}, frame: {duration: 0, redraw: false}});
            
            Plotly.animate('motor_dq', {
                data: [{x: bodyKeys.map(k=>k.replace('.q','')), y: bodyKeys.map(k=>data[k.replace('.q','.dq')]||0)}]
            }, {transition: {duration: 0}, frame: {duration: 0, redraw: false}});
            
            Plotly.animate('motor_tau', {
                data: [{x: bodyKeys.map(k=>k.replace('.q','')), y: bodyKeys.map(k=>data[k.replace('.q','.tau')]||0)}]
            }, {transition: {duration: 0}, frame: {duration: 0, redraw: false}});

            // Update Hand Charts
            Plotly.animate('left_hand_q', {
                data: [{x: leftKeys.map(k=>k.replace('.q','')), y: leftKeys.map(k=>data[k])}]
            }, {transition: {duration: 0}, frame: {duration: 0, redraw: false}});

            Plotly.animate('right_hand_q', {
                data: [{x: rightKeys.map(k=>k.replace('.q','')), y: rightKeys.map(k=>data[k])}]
            }, {transition: {duration: 0}, frame: {duration: 0, redraw: false}});

            // Update IMU
            const accel = [data['imu.accel.x'], data['imu.accel.y'], data['imu.accel.z']];
            const gyro = [data['imu.gyro.x'], data['imu.gyro.y'], data['imu.gyro.z']];
            const rpy = [data['imu.rpy.roll'], data['imu.rpy.pitch'], data['imu.rpy.yaw']];

            Plotly.extendTraces('imu_accel', {y: [[accel[0]], [accel[1]], [accel[2]]] }, [0, 1, 2], 50);
            Plotly.extendTraces('imu_gyro', {y: [[gyro[0]], [gyro[1]], [gyro[2]]] }, [0, 1, 2], 50);
            Plotly.extendTraces('imu_rpy', {y: [[rpy[0]], [rpy[1]], [rpy[2]]] }, [0, 1, 2], 50);
        }

        function fetchData() {
            fetch('/data')
                .then(response => {
                    if (!response.ok) throw new Error("Network response was not ok");
                    return response.json();
                })
                .then(data => {
                    updateCharts(data);
                })
                .catch(err => {
                    console.error("Fetch error:", err);
                    document.getElementById('connection-status').innerText = "Disconnected";
                    document.getElementById('connection-status').style.color = "red";
                });
        }

        // Poll every 100ms
        setInterval(fetchData, 100);
    </script>
</body>
</html>
"""

@app.route('/')
def idx():
    return render_template_string(HTML_TEMPLATE)

def gen_frames():
    while True:
        if robot and hasattr(robot, 'cameras') and 'head_camera' in robot.cameras:
            try:
                # async_read to allow other threads to run
                frame = robot.cameras['head_camera'].async_read(timeout_ms=100)
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                # logger.warning(f"Cam error: {e}")
                pass
            time.sleep(0.01)
        else:
            time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def create_placeholder_frame(text="No Signal"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def gen_frames():
    placeholder = create_placeholder_frame("Waiting for Camera...")
    no_cam_placeholder = create_placeholder_frame("Camera Disconnected")
    
    while True:
        # Use decoupled global camera if available
        if HEAD_CAMERA:
            try:
                # async_read waits for a new frame (blocking). 
                frame = HEAD_CAMERA.async_read(timeout_ms=500)
                if frame is not None:
                    # Fix Color: ImageServer sends RGB-in-BGR-container (channel swapped). 
                    # We swap it back to make it look correct.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Encode with lower quality for speed/low-latency
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            except Exception as e:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.1)
        else:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + no_cam_placeholder + b'\r\n')
             time.sleep(1.0)

@app.route('/set_action', methods=['POST'])
def set_action():
    if robot and robot.is_connected:
        try:
            data = request.json
            # logger.info(f"Received action: {data}")
            
            # Construct Action dictionary
            # The dashboard sends e.g. {'left_hand_thumb_0_joint.q': 0.5}
            # We pass this directly to send_action
            
            # Note: send_action expects a RobotAction (dict)
            robot.send_action(data)
            
            return jsonify({"status": "success"})
        except Exception as e:
            logger.error(f"Error setting action: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Robot not connected"}), 503

@app.route('/data')
def get_data():
    if robot and robot.is_connected:
        try:
            obs = robot.get_observation()
            # Clean data for JSON
            clean_obs = {}
            for k, v in obs.items():
                if isinstance(v, (np.integer, int)):
                    clean_obs[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_obs[k] = float(v)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    if isinstance(v, np.ndarray):
                        clean_obs[k] = v.tolist()
                    else:
                        clean_obs[k] = list(v)
                else:
                     clean_obs[k] = str(v)
            
            # Filter out images (should be none now if optimization worked)
            clean_obs = {k: v for k, v in clean_obs.items() if 'image' not in k}
            
            return jsonify(clean_obs)
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Robot not connected"}), 503

def connect_robot():
    global robot, HEAD_CAMERA
    try:
        logger.info("Attempting to connect to Unitree G1 Dex3...")
        config = UnitreeG1Dex3Config()
        config.is_simulation = False 
        
        # Configure Camera
        config.cameras = {
            "head_camera": ZMQCameraConfig(
                server_address=config.robot_ip,
                port=5555,
                camera_name="head_camera",
                color_mode=ColorMode.BGR,
                timeout_ms=3000
            ) 
        }

        robot = UnitreeG1Dex3(config)
        
        try:
            robot.connect()
        except Exception as e:
            logger.warning(f"UnitreeG1Dex3 connect failed (likely hand controller): {e}")
            if robot:
                try:
                    robot.disconnect()
                except Exception:
                    pass

            logger.info("Falling back to standard UnitreeG1 (body only)...")
            # Keep configuration for camera if possible
            robot = UnitreeG1(config)
            robot.connect()
            
        logger.info("Robot Connected!")
        
        # Optimize: Decouple camera logic
        if hasattr(robot, 'cameras') and 'head_camera' in robot.cameras:
            logger.info("Optimizing: Decoupling camera from main loop...")
            HEAD_CAMERA = robot.cameras['head_camera']
            # Remove from robot.cameras to prevent double-read in get_observation
            try:
                del robot.cameras['head_camera']
            except:
                pass
            if hasattr(robot, '_cameras') and 'head_camera' in robot._cameras:
                del robot._cameras['head_camera']
        elif hasattr(robot, '_cameras') and 'head_camera' in robot._cameras:
             HEAD_CAMERA = robot._cameras['head_camera']
             del robot._cameras['head_camera']
        
    except Exception as e:
        logger.error(f"Failed to initialize robot: {e}")
        sys.exit(1)

def signal_handler(sig, frame):
    logger.info('Shutting down...')
    if HEAD_CAMERA:
        try:
             HEAD_CAMERA.disconnect()
        except:
             pass
    if robot:
        robot.disconnect()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start connection
    connect_robot()
    
    # Run server
    logger.info("Starting Web Server on http://0.0.0.0:5000")
    # Restore threading to prevent blocking
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
