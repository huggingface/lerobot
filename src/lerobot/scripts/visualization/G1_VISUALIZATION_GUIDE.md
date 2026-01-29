# Unitree G1 Visualization Guide

This guide describes how to use the visualization tools for the Unitree G1 robot, including the 2D Dashboard and 3D Visualization.

## Prerequisites

Ensure you are in the project root and have the environment set up:

```bash
source ~/unitree_ros2/setup.sh
```

## 1. Robot Setup

Before running the visualizations, the robot must be streaming data.

### A. Start the ZMQ Bridge
This streams motor and IMU data.
**On the Robot:**
```bash
python lerobot/src/lerobot/robots/unitree_g1/run_g1_server.py
```

### B. Start the Camera Server
This streams the head camera feed.
**On the Robot:**
```bash
python lerobot/src/lerobot/cameras/zmq/image_server.py
```
*Note: If this is not running, the dashboard will show "Camera Disconnected" but will otherwise function normally.*

---

## 2. 2D Dashboard (`visualize_g1.py`)

A Flask-based web dashboard showing real-time plots of motor states, IMU data, and the camera feed.

### Running the Dashboard
**On the Local Machine:**
```bash
uv run lerobot/src/lerobot/scripts/visualize_g1.py
```

### Accessing the Dashboard
Open your browser to: [http://localhost:5000](http://localhost:5000)

### Features
- **Camera Feed:** Real-time stream from the robot's head camera (RGB).
- **Motor Plots:** Joint positions (q), velocities (dq), and torque (tau).
- **IMU Plots:** Accelerometer, Gyroscope, and Orientation (RPY).
- **Status Indicators:** Connection status and Update Rate (FPS).

### Troubleshooting
- **"Camera Disconnected":** Ensure `image_server.py` is running on the robot (Port 5555).
- **"Robot not connected":** Ensure `run_g1_server.py` is running on the robot and the computer is on the same network (check `192.168.123.x`).

---

## 3. 3D Visualization (`visualize_g1_3d.py`)

A 3D viewer using `vuer` to render the robot's URDF and real-time pose.

### Running the Viewer
**On the Local Machine:**
```bash
uv run lerobot/src/lerobot/scripts/visualize_g1_3d.py
```

### Accessing the Viewer
Open your browser to: [http://localhost:8012](http://localhost:8012)

### Features
- **Real-time Pose:** The 3D model updates to match the physical robot.
- **Interactive Camera:** Orbit, zoom, and pan around the robot.

---

## Architecture Notes
- **Dashboard Optimization:** The dashboard decouples camera reading from the main robot loop to ensure high-frequency motor data updates (10Hz+) even if the video stream is slower.
- **Fallback Mode:** If the camera is unavailable, the dashboard automatically disables the video feed to prevent freezing.
