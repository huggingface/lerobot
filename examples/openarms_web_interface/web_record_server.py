"""
OpenArms Web Recording Server

FastAPI backend for recording OpenArms datasets with gravity compensation.
Provides camera streaming, robot control, and automatic HuggingFace upload.
"""

import asyncio
import platform
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from evdev import InputDevice, categorize, ecodes
    PEDAL_AVAILABLE = True
except ImportError:
    PEDAL_AVAILABLE = False
    print("[Pedal] evdev not installed. Pedal support disabled. Install with: pip install evdev")

# LeRobot imports
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.teleoperators.openarms_mini.config_openarms_mini import OpenArmsMiniConfig
from lerobot.teleoperators.openarms_mini.openarms_mini import OpenArmsMini

app = FastAPI(title="OpenArms Recording Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
recording_state = {
    "is_recording": False,
    "is_initializing": False,
    "is_encoding": False,
    "is_uploading": False,
    "robots_ready": False,
    "start_time": None,
    "task": "",
    "episode_count": 0,
    "error": None,
    "status_message": "Ready",
    "upload_status": None,
    "current_fps": 0.0,
    "loop_fps": 0.0,  # Actual control loop FPS (critical - must be 30)
    "ramp_up_remaining": 0.0,  # Remaining seconds for PID ramp-up
    "recording_started_time": None,  # Time when actual recording starts (after ramp-up)
    "moving_to_zero": False,  # Whether robot is moving to zero position
    "config": {
        "leader_type": "openarms",  # "openarms" or "openarms_mini"
        "leader_left": "can0",
        "leader_right": "can1",
        "follower_left": "can2",
        "follower_right": "can3",
        "left_wrist": "/dev/video0",
        "right_wrist": "/dev/video1",
        "base": "/dev/video4",
    }
}

# Global robot instances
robot_instances = {
    "follower": None,
    "leader": None,
    "dataset": None,
    "dataset_features": None,
}

# Camera frames from recording loop (for streaming during recording)
# Store raw RGB frames directly from robot observation (no pre-encoding)
camera_frames_raw = {
    "left_wrist": None,
    "right_wrist": None,
    "base": None,
}
camera_frames_timestamp = {
    "left_wrist": 0.0,
    "right_wrist": 0.0,
    "base": 0.0,
}
camera_frames_lock = threading.Lock()
# Maximum age for frames (in seconds) - discard frames older than this
MAX_FRAME_AGE = 0.15  # 150ms = allows for ~6-7 FPS freshness 
# Recording control
recording_thread = None
stop_recording_flag = threading.Event()
FPS = 30
FRICTION_SCALE = 1.0
RAMP_UP_DURATION = 3.0  # Seconds to ramp up PID/torque from 0 to full

# OpenArms Mini joint mappings (from teleop_openarms_mini.py)
# Note: Direction flipping is now handled in OpenArmsMini.get_action() method
SWAPPED_JOINTS_MINI = {
    "right_joint_6": "right_joint_7",
    "right_joint_7": "right_joint_6",
    "left_joint_6": "left_joint_7",
    "left_joint_7": "left_joint_6",
}

# Pedal configuration
PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
PEDAL_ENABLED = True  # Set to False to disable pedal
pedal_thread = None
stop_pedal_flag = threading.Event()
pedal_action_lock = threading.Lock()  # Prevent concurrent pedal actions


class RecordingConfig(BaseModel):
    task: str
    leader_type: str  # "openarms" or "openarms_mini"
    leader_left: str
    leader_right: str
    follower_left: str
    follower_right: str
    left_wrist: str
    right_wrist: str
    base: str


class RobotSetupConfig(BaseModel):
    """Configuration for robot setup (no task required)."""
    leader_type: str  # "openarms" or "openarms_mini"
    leader_left: str
    leader_right: str
    follower_left: str
    follower_right: str
    left_wrist: str
    right_wrist: str
    base: str


class CounterUpdate(BaseModel):
    value: int


def discover_cameras_sync():
    """Discover available OpenCV cameras."""
    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        cameras = OpenCVCamera.find_cameras()
        print(f"[Cameras] Found {len(cameras)} camera(s)")
        return cameras
    except Exception as e:
        print(f"[Cameras] Error: {e}")
        return []


def discover_usb_ports_sync():
    """Discover available USB/serial ports for OpenArms Mini."""
    try:
        import glob
        import serial
        # Find all USB serial devices
        ports = []
        for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
            for port in glob.glob(pattern):
                # Check if port is accessible
                try:
                    # Try to open and close the port to verify access
                    ser = serial.Serial(port, timeout=0.1)
                    ser.close()
                    ports.append(port)
                except (OSError, serial.SerialException) as e:
                    print(f"[USB] Port {port} exists but not accessible: {e}")
        ports.sort()
        print(f"[USB] Found {len(ports)} accessible USB port(s): {ports}")
        return ports
    except Exception as e:
        print(f"[USB] Error: {e}")
        return []


@app.get("/api/cameras/discover")
async def discover_cameras():
    """Discover available cameras."""
    cameras = discover_cameras_sync()
    return {"cameras": cameras}


@app.get("/api/usb/discover")
async def discover_usb_ports():
    """Discover available USB/serial ports."""
    ports = discover_usb_ports_sync()
    return {"ports": ports}


@app.get("/api/usb/test/{port:path}")
async def test_usb_port(port: str):
    """Test if a USB port is accessible."""
    try:
        import serial
        print(f"[USB Test] Testing port: {port}")
        ser = serial.Serial(port, baudrate=1000000, timeout=0.1)
        ser.close()
        return {"status": "success", "message": f"Port {port} is accessible"}
    except PermissionError as e:
        return {"status": "error", "message": f"Permission denied: {e}"}
    except serial.SerialException as e:
        return {"status": "error", "message": f"Serial error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}"}


@app.get("/api/can/interfaces")
async def get_can_interfaces():
    """Get available CAN interfaces."""
    return {"interfaces": ["can0", "can1", "can2", "can3"]}


def initialize_robots_only(config: RobotSetupConfig):
    """Initialize robots only (no dataset) for pre-setup."""
    global robot_instances, recording_state
    
    print(f"[Setup] Initializing robots with leader type: {config.leader_type}")
    print(f"[Setup] Leader ports: left={config.leader_left}, right={config.leader_right}")
    print(f"[Setup] Follower ports: left={config.follower_left}, right={config.follower_right}")

    # Update status: Configuring cameras
    recording_state["status_message"] = "Configuring cameras..."
    camera_config = {
        "left_wrist": OpenCVCameraConfig(index_or_path=config.left_wrist, width=1280, height=720, fps=FPS),
        "right_wrist": OpenCVCameraConfig(index_or_path=config.right_wrist, width=1280, height=720, fps=FPS),
        "base": OpenCVCameraConfig(index_or_path=config.base, width=640, height=480, fps=FPS),
    }

    # Configure follower robot with cameras
    recording_state["status_message"] = "Configuring follower robot..."
    print(f"[Setup] Configuring follower with CAN: {config.follower_left}, {config.follower_right}")
    follower_config = OpenArmsFollowerConfig(
        port_left=config.follower_left,
        port_right=config.follower_right,
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=30.0,
        cameras=camera_config,
    )

    # Configure leader teleoperator based on type
    recording_state["status_message"] = "Configuring leader teleoperator..."
    if config.leader_type == "openarms_mini":
        # OpenArms Mini: USB/serial ports, Feetech motors
        leader_config = OpenArmsMiniConfig(
            port_left=config.leader_left,
            port_right=config.leader_right,
            id="openarms_mini",
            use_degrees=True,
        )
        leader = OpenArmsMini(leader_config)
    else:
        # OpenArms: CAN bus, Damiao motors
        leader_config = OpenArmsLeaderConfig(
            port_left=config.leader_left,
            port_right=config.leader_right,
            can_interface="socketcan",
            id="openarms_leader",
            manual_control=False,
        )
        leader = OpenArmsLeader(leader_config)

    # Initialize and connect
    recording_state["status_message"] = "Connecting to follower robot..."
    follower = OpenArmsFollower(follower_config)
    try:
        follower.connect(calibrate=False)  # Skip calibration in web mode
    except Exception as e:
        print(f"[Setup] Follower connection error: {e}")
        raise RuntimeError(f"Failed to connect to follower robot: {e}")

    recording_state["status_message"] = "Connecting to leader teleoperator..."
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"[Setup] Leader connection error: {e}")
        # Disconnect follower on leader connection failure
        try:
            follower.disconnect()
        except:
            pass
        raise RuntimeError(f"Failed to connect to leader teleoperator: {e}")

    # Enable torque/gravity compensation based on leader type
    if config.leader_type == "openarms":
        # Verify URDF is loaded for OpenArms (needs gravity compensation)
        recording_state["status_message"] = "Loading URDF model for gravity compensation..."
        if leader.pin_robot is None:
            raise RuntimeError("URDF model not loaded on leader. Gravity compensation not available.")
        
        # Enable gravity compensation
        recording_state["status_message"] = "Enabling gravity compensation..."
        leader.bus_right.enable_torque()
        leader.bus_left.enable_torque()
        time.sleep(0.1)
    else:
        # OpenArms Mini: No gravity compensation needed (Feetech motors)
        recording_state["status_message"] = "Leader ready (no gravity compensation needed)..."

    robot_instances["follower"] = follower
    robot_instances["leader"] = leader
    robot_instances["leader_type"] = config.leader_type


def initialize_robot_systems(config: RecordingConfig):
    """Initialize robot, leader, and dataset."""
    global robot_instances, recording_state

    # Check if robots are already initialized
    if robot_instances.get("follower") and robot_instances.get("leader"):
        recording_state["status_message"] = "Using existing robots..."
        follower = robot_instances["follower"]
        leader = robot_instances["leader"]
        print(f"[Initialize] Reusing existing robots")
    else:
        # Full initialization required
        # Update status: Configuring cameras
        recording_state["status_message"] = "Configuring cameras..."
        camera_config = {
            "left_wrist": OpenCVCameraConfig(index_or_path=config.left_wrist, width=640, height=480, fps=FPS),
            "right_wrist": OpenCVCameraConfig(index_or_path=config.right_wrist, width=640, height=480, fps=FPS),
            "base": OpenCVCameraConfig(index_or_path=config.base, width=640, height=480, fps=FPS),
        }

        # Configure follower robot with cameras
        recording_state["status_message"] = "Configuring follower robot..."
        follower_config = OpenArmsFollowerConfig(
            port_left=config.follower_left,
            port_right=config.follower_right,
            can_interface="socketcan",
            id="openarms_follower",
            disable_torque_on_disconnect=True,
            max_relative_target=10.0,
            cameras=camera_config,
        )

        # Configure leader teleoperator based on type
        recording_state["status_message"] = "Configuring leader teleoperator..."
        if config.leader_type == "openarms_mini":
            # OpenArms Mini: USB/serial ports, Feetech motors
            leader_config = OpenArmsMiniConfig(
                port_left=config.leader_left,
                port_right=config.leader_right,
                id="openarms_mini",
                use_degrees=True,
            )
            leader = OpenArmsMini(leader_config)
        else:
            # OpenArms: CAN bus, Damiao motors
            leader_config = OpenArmsLeaderConfig(
                port_left=config.leader_left,
                port_right=config.leader_right,
                can_interface="socketcan",
                id="openarms_leader",
                manual_control=False,
            )
            leader = OpenArmsLeader(leader_config)

        # Initialize and connect
        recording_state["status_message"] = "Connecting to follower robot..."
        follower = OpenArmsFollower(follower_config)
        follower.connect(calibrate=False)  # Skip calibration in web mode

        recording_state["status_message"] = "Connecting to leader teleoperator..."
        leader.connect(calibrate=False)

        # Enable torque/gravity compensation based on leader type
        if config.leader_type == "openarms":
            # Verify URDF is loaded for OpenArms (needs gravity compensation)
            recording_state["status_message"] = "Loading URDF model for gravity compensation..."
            if leader.pin_robot is None:
                raise RuntimeError("URDF model not loaded on leader. Gravity compensation not available.")
            
            # Enable gravity compensation
            recording_state["status_message"] = "Enabling gravity compensation..."
            leader.bus_right.enable_torque()
            leader.bus_left.enable_torque()
            time.sleep(0.1)
        else:
            # OpenArms Mini: No gravity compensation needed (Feetech motors)
            recording_state["status_message"] = "Leader ready (no gravity compensation needed)..."
        
        robot_instances["leader_type"] = config.leader_type

    # Configure dataset features
    recording_state["status_message"] = "Configuring dataset features..."
    action_features_hw = {}
    for key, value in follower.action_features.items():
        if key.endswith(".pos"):
            action_features_hw[key] = value

    action_features = hw_to_dataset_features(action_features_hw, "action", use_video=False)
    obs_features = hw_to_dataset_features(follower.observation_features, "observation", use_video=False)
    dataset_features = {**action_features, **obs_features}

    # Create dataset
    recording_state["status_message"] = "Creating dataset..."
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M")
    
    # Validate and sanitize task name
    task_raw = config.task.strip()
    if not task_raw:
        raise ValueError("Task description cannot be empty")
    
    # Replace spaces with hyphens and convert to lowercase
    task = task_raw.replace(" ", "-").lower()
    
    # Remove any leading/trailing hyphens or dots
    task = task.strip("-.")
    
    # Ensure task name is valid (alphanumeric, hyphens, underscores, dots only)
    task = re.sub(r'[^a-z0-9\-_.]', '', task)
    
    if not task:
        raise ValueError("Task description must contain alphanumeric characters")
    
    dataset_name = f"{task}-{date_str}-{time_str}"
    repo_id = f"lerobot-data-collection/{dataset_name}"
    
    print(f"[Initialize] Sanitized task: '{config.task}' -> '{task}'")
    print(f"[Initialize] Dataset name: {dataset_name}")
    print(f"[Initialize] Repo ID: {repo_id}")

    # Remove existing dataset if it exists
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if dataset_path.exists():
        recording_state["status_message"] = "Removing existing dataset..."
        shutil.rmtree(dataset_path)

    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=dataset_features,
            robot_type=follower.name,
            use_videos=True,
            image_writer_processes=0,  # Use threads only (no multiprocessing)
            image_writer_threads=12,    # 4 threads per camera for 3 cameras
        )
    except Exception as e:
        print(f"[Dataset] Error: {e}")
        raise RuntimeError(f"Failed to create dataset: {e}")
    
    robot_instances["follower"] = follower
    robot_instances["leader"] = leader
    robot_instances["dataset"] = dataset
    robot_instances["dataset_features"] = dataset_features
    robot_instances["repo_id"] = repo_id

    if robot_instances.get("dataset") is None:
        raise RuntimeError("Dataset was not stored!")
    
    print(f"[Initialize] All systems ready")

    recording_state["status_message"] = "Recording..."
    # Keep robots_ready = True so we can reuse them for next recording
    # recording_state["robots_ready"] = False  # Don't mark as consumed anymore

    return follower, leader, dataset, dataset_features


def record_loop_with_compensation():
    """Main recording loop (with or without compensation based on leader type)."""
    global recording_state, stop_recording_flag
    
    follower = robot_instances.get("follower")
    leader = robot_instances.get("leader")
    dataset = robot_instances.get("dataset")
    dataset_features = robot_instances.get("dataset_features")
    leader_type = robot_instances.get("leader_type", "openarms")
    task = recording_state.get("task", "")
    
    # Profiling: Track time spent in each operation
    timing_stats = {
        "get_action": [],
        "gravity_calc": [],
        "friction_calc": [],
        "leader_control": [],
        "follower_action": [],
        "get_observation": [],
        "frame_store": [],
        "dataset_add": [],
        "sleep": [],
    }
    profile_counter = 0
    PROFILE_INTERVAL = 150  # Log every 150 frames (5 seconds @ 30 Hz)
    
    if follower is None or leader is None or dataset is None:
        recording_state["error"] = "Robot or dataset not initialized"
        print(f"[Recording] Error: Missing components")
        return
    
    print(f"[Recording] Starting recording loop (leader: {leader_type}) with ramp-up...")
    
    dt = 1 / FPS
    loop_start_time = time.perf_counter()
    episode_start_time = None  # Will be set after ramp-up completes
    frame_count = 0
    last_fps_update = loop_start_time
    fps_frame_count = 0
    
    # Separate tracking for actual loop FPS (critical for control)
    loop_fps_frame_count = 0
    last_loop_fps_update = loop_start_time
    
    # All joints (both arms)
    all_joints = []
    for motor in leader.bus_right.motors:
        all_joints.append(f"right_{motor}")
    for motor in leader.bus_left.motors:
        all_joints.append(f"left_{motor}")

    # Disable torque on OpenArms Mini leader for manual control
    if leader_type == "openarms_mini":
        print(f"[Recording] Disabling torque on OpenArms Mini leader for manual control")
        leader.bus_right.disable_torque()
        leader.bus_left.disable_torque()
    
    try:
        while not stop_recording_flag.is_set():
            loop_start = time.perf_counter()
            elapsed_total = loop_start - loop_start_time
            
            # PID ramp-up: calculate ramp factor and completion status
            ramp_up_remaining = max(0.0, RAMP_UP_DURATION - elapsed_total)
            ramp_factor = max(0.0, min(1.0, 1.0 - (ramp_up_remaining / RAMP_UP_DURATION)))
            is_ramp_up_complete = ramp_up_remaining <= 0.0
            
            # Start episode timer after ramp-up completes
            if is_ramp_up_complete and episode_start_time is None:
                episode_start_time = loop_start
                recording_state["recording_started_time"] = time.time()  # Set timestamp for UI
                print(f"[Recording] Ramp-up complete, starting data recording")
                
            
            # Update ramp-up status for UI
            recording_state["ramp_up_remaining"] = round(ramp_up_remaining, 2)
            
            # CRITICAL: Track actual control loop FPS (must be 30 Hz)
            # Calculate every second for accurate monitoring
            loop_fps_frame_count += 1
            loop_elapsed = loop_start - last_loop_fps_update
            if loop_elapsed >= 1.0:
                actual_loop_fps = loop_fps_frame_count / loop_elapsed
                recording_state["loop_fps"] = round(actual_loop_fps, 1)
                loop_fps_frame_count = 0
                last_loop_fps_update = loop_start
                
                # Log if we're falling behind
                if actual_loop_fps < 29.0:
                    print(f"[Recording] WARNING: Loop FPS low: {actual_loop_fps:.1f} Hz")
            
            # Calculate frame recording FPS every second (only after ramp-up)
            if is_ramp_up_complete:
                elapsed = loop_start - episode_start_time
                fps_frame_count += 1
                if elapsed - (last_fps_update - episode_start_time) >= 1.0:
                    actual_fps = fps_frame_count / (elapsed - (last_fps_update - episode_start_time))
                    recording_state["current_fps"] = round(actual_fps, 1)
                    fps_frame_count = 0
                    last_fps_update = loop_start
            
            # Get leader state
            t0 = time.perf_counter()
            leader_action = leader.get_action()
            timing_stats["get_action"].append(time.perf_counter() - t0)
            
            # Apply leader-specific control logic
            if leader_type == "openarms":
                # OpenArms: Apply gravity and friction compensation
                # Extract positions and velocities
                leader_positions_deg = {}
                leader_velocities_deg_per_sec = {}
                
                for motor in leader.bus_right.motors:
                    pos_key = f"right_{motor}.pos"
                    vel_key = f"right_{motor}.vel"
                    if pos_key in leader_action:
                        leader_positions_deg[f"right_{motor}"] = leader_action[pos_key]
                    if vel_key in leader_action:
                        leader_velocities_deg_per_sec[f"right_{motor}"] = leader_action[vel_key]
                
                for motor in leader.bus_left.motors:
                    pos_key = f"left_{motor}.pos"
                    vel_key = f"left_{motor}.vel"
                    if pos_key in leader_action:
                        leader_positions_deg[f"left_{motor}"] = leader_action[pos_key]
                    if vel_key in leader_action:
                        leader_velocities_deg_per_sec[f"left_{motor}"] = leader_action[vel_key]
                
                # Calculate gravity and friction torques
                leader_positions_rad = {k: np.deg2rad(v) for k, v in leader_positions_deg.items()}
                
                t0 = time.perf_counter()
                leader_gravity_torques_nm = leader._gravity_from_q(leader_positions_rad)
                timing_stats["gravity_calc"].append(time.perf_counter() - t0)
                
                leader_velocities_rad_per_sec = {k: np.deg2rad(v) for k, v in leader_velocities_deg_per_sec.items()}
                
                t0 = time.perf_counter()
                leader_friction_torques_nm = leader._friction_from_velocity(
                    leader_velocities_rad_per_sec,
                    friction_scale=FRICTION_SCALE
                )
                timing_stats["friction_calc"].append(time.perf_counter() - t0)
                
                # Combine torques
                leader_total_torques_nm = {}
                for motor_name in leader_gravity_torques_nm:
                    gravity = leader_gravity_torques_nm.get(motor_name, 0.0)
                    friction = leader_friction_torques_nm.get(motor_name, 0.0)
                    leader_total_torques_nm[motor_name] = gravity + friction
                
                # Apply compensation to RIGHT arm with ramp-up scaling
                t0 = time.perf_counter()
                for motor in leader.bus_right.motors:
                    full_name = f"right_{motor}"
                    position = leader_positions_deg.get(full_name, 0.0)
                    torque_raw = leader_total_torques_nm.get(full_name, 0.0)
                    # Scale torque by ramp factor during ramp-up phase
                    torque = torque_raw * ramp_factor
                    kd = leader.get_damping_kd(motor)
                    
                    leader.bus_right._mit_control(
                        motor=motor, kp=0.0, kd=kd,
                        position_degrees=position,
                        velocity_deg_per_sec=0.0,
                        torque=torque,
                    )
                
                # Apply compensation to LEFT arm with ramp-up scaling
                for motor in leader.bus_left.motors:
                    full_name = f"left_{motor}"
                    position = leader_positions_deg.get(full_name, 0.0)
                    torque_raw = leader_total_torques_nm.get(full_name, 0.0)
                    # Scale torque by ramp factor during ramp-up phase
                    torque = torque_raw * ramp_factor
                    kd = leader.get_damping_kd(motor)
                    
                    leader.bus_left._mit_control(
                        motor=motor, kp=0.0, kd=kd,
                        position_degrees=position,
                        velocity_deg_per_sec=0.0,
                        torque=torque,
                    )
                timing_stats["leader_control"].append(time.perf_counter() - t0)
            
            # Build follower action from leader action
            follower_action = {}
            if leader_type == "openarms_mini":
                # OpenArms Mini: Apply joint direction and swap mappings
                for joint in all_joints:
                    leader_key = f"{joint}.pos"
                    
                    # Determine which follower joint this leader joint controls
                    follower_joint = SWAPPED_JOINTS_MINI.get(joint, joint)
                    follower_key = f"{follower_joint}.pos"
                    
                    # Get leader position (default 0 if missing)
                    pos = leader_action.get(leader_key, 0.0)
                    
                    # Convert gripper values: Mini uses 0-100, OpenArms uses 0 to -65 degrees
                    if "gripper" in joint:
                        # Map 0-100 (Mini) to 0 to -65 (OpenArms)
                        # 0 (closed) -> 0°, 100 (open) -> -65°
                        pos = (pos / 100.0) * -65.0
                    
                    follower_action[follower_key] = pos
            else:
                # OpenArms: Direct position mapping (no joint swapping)
                for joint in all_joints:
                    pos_key = f"{joint}.pos"
                    if pos_key in leader_action:
                        follower_action[pos_key] = leader_action[pos_key]
            
            if follower_action:
                # Build custom ramped PID gains for all motors
                motor_index = {
                    "joint_1": 0, "joint_2": 1, "joint_3": 2, "joint_4": 3,
                    "joint_5": 4, "joint_6": 5, "joint_7": 6, "gripper": 7,
                }
                
                custom_kp = {}
                custom_kd = {}
                
                # Calculate ramped gains for all motors in both arms
                for motor in follower.bus_right.motors:
                    full_name = f"right_{motor}"
                    idx = motor_index.get(motor, 0)
                    kp_base = follower.config.position_kp[idx] if isinstance(follower.config.position_kp, list) else follower.config.position_kp
                    kd_base = follower.config.position_kd[idx] if isinstance(follower.config.position_kd, list) else follower.config.position_kd
                    custom_kp[full_name] = kp_base * ramp_factor
                    custom_kd[full_name] = kd_base * ramp_factor
                
                for motor in follower.bus_left.motors:
                    full_name = f"left_{motor}"
                    idx = motor_index.get(motor, 0)
                    kp_base = follower.config.position_kp[idx] if isinstance(follower.config.position_kp, list) else follower.config.position_kp
                    kd_base = follower.config.position_kd[idx] if isinstance(follower.config.position_kd, list) else follower.config.position_kd
                    custom_kp[full_name] = kp_base * ramp_factor
                    custom_kd[full_name] = kd_base * ramp_factor
                
                # Send action with custom ramped gains (includes safety checks)
                t0 = time.perf_counter()
                follower.send_action(follower_action, custom_kp=custom_kp, custom_kd=custom_kd)
                timing_stats["follower_action"].append(time.perf_counter() - t0)
            
            # Get observation (with detailed breakdown)
            t0 = time.perf_counter()
            observation = follower.get_observation()
            obs_time = time.perf_counter() - t0
            timing_stats["get_observation"].append(obs_time)
            
            # Extract detailed timing breakdown from observation
            if "_timing_breakdown" in observation:
                for key, value_ms in observation["_timing_breakdown"].items():
                    if key not in timing_stats:
                        timing_stats[key] = []
                    timing_stats[key].append(value_ms / 1000)  # Convert ms to seconds
                # Remove timing metadata before storing in dataset
                del observation["_timing_breakdown"]
            
            # Store raw camera frames directly for streaming (no encoding in control loop!)
            # Streaming endpoint will encode at 10 FPS as needed
            # PRIORITY: 30 Hz control loop must not be slowed by encoding
            frame_timestamp = time.perf_counter()
            
            t0 = time.perf_counter()
            with camera_frames_lock:
                for cam_name in ["left_wrist", "right_wrist", "base"]:
                    if cam_name in observation and observation[cam_name] is not None:
                        # Store raw RGB frame directly - no encoding overhead
                        camera_frames_raw[cam_name] = observation[cam_name]
                        camera_frames_timestamp[cam_name] = frame_timestamp
            timing_stats["frame_store"].append(time.perf_counter() - t0)
            
            # Add to dataset ONLY after ramp-up completes
            if is_ramp_up_complete:
                try:
                    t0 = time.perf_counter()
                    obs_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
                    action_frame = build_dataset_frame(dataset_features, follower_action, prefix="action")
                    frame = {**obs_frame, **action_frame}
                    frame["task"] = task
                    dataset.add_frame(frame)
                    timing_stats["dataset_add"].append(time.perf_counter() - t0)
                    frame_count += 1
                    profile_counter += 1
                    
                    # Log progress and profiling every 5 seconds (150 frames @ 30 FPS)
                    if profile_counter >= PROFILE_INTERVAL:
                        profile_counter = 0
                        # Calculate averages
                        avg_times = {k: (sum(v) / len(v) * 1000) if v else 0 for k, v in timing_stats.items()}
                        
                        # Core operations for total time
                        core_ops = ['get_action', 'gravity_calc', 'friction_calc', 'leader_control', 
                                   'follower_action', 'get_observation', 'frame_store', 'dataset_add']
                        total_time = sum(avg_times.get(k, 0) for k in core_ops)
                        
                        print(f"[Recording] {frame_count} frames | Loop: {recording_state['loop_fps']} Hz | Recording: {recording_state['current_fps']} FPS")
                        print(f"[Profiling] Time per operation (ms):")
                        print(f"  get_action:      {avg_times.get('get_action', 0):6.2f} ms ({avg_times.get('get_action', 0)/total_time*100:5.1f}%)")
                        print(f"  gravity_calc:    {avg_times.get('gravity_calc', 0):6.2f} ms ({avg_times.get('gravity_calc', 0)/total_time*100:5.1f}%)")
                        print(f"  friction_calc:   {avg_times.get('friction_calc', 0):6.2f} ms ({avg_times.get('friction_calc', 0)/total_time*100:5.1f}%)")
                        print(f"  leader_control:  {avg_times.get('leader_control', 0):6.2f} ms ({avg_times.get('leader_control', 0)/total_time*100:5.1f}%)")
                        print(f"  follower_action: {avg_times.get('follower_action', 0):6.2f} ms ({avg_times.get('follower_action', 0)/total_time*100:5.1f}%)")
                        print(f"  get_observation: {avg_times.get('get_observation', 0):6.2f} ms ({avg_times.get('get_observation', 0)/total_time*100:5.1f}%)")
                        
                        # Show detailed breakdown of get_observation
                        if any(k.startswith('cam_') or k in ['right_motors', 'left_motors'] for k in avg_times):
                            print(f"    └─ Breakdown:")
                            if 'right_motors' in avg_times:
                                print(f"       right_motors:  {avg_times['right_motors']:6.2f} ms")
                            if 'left_motors' in avg_times:
                                print(f"       left_motors:   {avg_times['left_motors']:6.2f} ms")
                            for cam in ['left_wrist', 'right_wrist', 'base']:
                                cam_key = f'cam_{cam}'
                                if cam_key in avg_times:
                                    print(f"       {cam:14s}: {avg_times[cam_key]:6.2f} ms")
                        
                        print(f"  frame_store:     {avg_times.get('frame_store', 0):6.2f} ms ({avg_times.get('frame_store', 0)/total_time*100:5.1f}%)")
                        print(f"  dataset_add:     {avg_times.get('dataset_add', 0):6.2f} ms ({avg_times.get('dataset_add', 0)/total_time*100:5.1f}%)")
                        print(f"  TOTAL:           {total_time:6.2f} ms (target: 33.33 ms @ 30 Hz)")
                        
                        # Clear stats for next interval
                        for k in timing_stats:
                            timing_stats[k].clear()
                except Exception as frame_error:
                    print(f"[Recording] Frame error: {frame_error}")
                    # Continue recording even if one frame fails
            
            # Maintain loop rate - don't wait if behind, just continue
            loop_duration = time.perf_counter() - loop_start
            sleep_time = dt - loop_duration
            
            # Only sleep if we're ahead of schedule (more than 2ms remaining)
            if sleep_time > 0.002:
                t0 = time.perf_counter()
                time.sleep(min(sleep_time, dt * 0.5))  # Never sleep more than half a frame time
                timing_stats["sleep"].append(time.perf_counter() - t0)
            # If behind, just continue without sleeping - maintain real-time rate
    
    except Exception as e:
        recording_state["error"] = str(e)
        print(f"[Recording] Error: {e}")
    finally:
        print(f"[Recording] Stopped. Total frames: {frame_count}")
        # Get fresh reference to dataset from global state
        current_dataset = robot_instances.get("dataset")
        if current_dataset:
            print(f"[Recording] Dataset exists, checking buffer...")
            print(f"[Recording] Episode buffer is None: {current_dataset.episode_buffer is None}")
            print(f"[Recording] Episode buffer value: {current_dataset.episode_buffer}")
            if current_dataset.episode_buffer:
                print(f"[Recording] Buffer size: {current_dataset.episode_buffer.get('size', 'NO SIZE KEY')}")
            else:
                print(f"[Recording] WARNING: Buffer is None despite {frame_count} frames!")
        else:
            print(f"[Recording] WARNING: Dataset is None in robot_instances!")
            print(f"[Recording] Local dataset variable: {dataset}")


@app.post("/api/recording/set-task")
async def set_task(config: RecordingConfig):
    """Store task and config without starting recording (for pedal use).
    
    This allows you to set the task in advance, so the pedal can start recording
    without needing to interact with the UI.
    """
    if recording_state["is_recording"] or recording_state["is_initializing"]:
        raise HTTPException(status_code=400, detail="Cannot change task while recording")
    
    # Store the config and task for later use by pedal or start button
    recording_state["config"] = config.model_dump()
    recording_state["task"] = config.task
    
    print(f"[SetTask] Task set to: {config.task}")
    
    return {
        "status": "success",
        "task": config.task,
        "message": "Task stored. You can now start recording with pedal or button."
    }


@app.post("/api/recording/start")
async def start_recording(config: RecordingConfig):
    """Start recording an episode."""
    return do_start_recording(config, source="API")




@app.post("/api/recording/stop")
async def stop_recording():
    """Stop recording, encode, and upload episode."""
    return do_stop_recording(source="API")


def cleanup_robot_systems(keep_robots=False):
    """Clean up robot systems."""
    global robot_instances
    
    try:
        if not keep_robots:
            if robot_instances.get("leader"):
                leader_type = robot_instances.get("leader_type", "openarms")
                # Only disable torque for OpenArms (with gravity compensation)
                if leader_type == "openarms":
                    robot_instances["leader"].bus_right.disable_torque()
                    robot_instances["leader"].bus_left.disable_torque()
                    time.sleep(0.1)
                robot_instances["leader"].disconnect()
            
            if robot_instances.get("follower"):
                    robot_instances["follower"].disconnect()
            
            robot_instances["follower"] = None
            robot_instances["leader"] = None
            robot_instances["leader_type"] = None
            recording_state["robots_ready"] = False
            print(f"[Cleanup] Robots disconnected")
        
        # Always clean up dataset
        robot_instances["dataset"] = None
        robot_instances["dataset_features"] = None
        robot_instances["repo_id"] = None
    
    except Exception as e:
        print(f"[Cleanup] Error: {e}")


def do_start_recording(config: RecordingConfig, source: str = "API"):
    """Core logic for starting recording (used by both API and pedal)."""
    global recording_thread
    
    if recording_state["is_recording"] or recording_state["is_initializing"]:
        raise HTTPException(status_code=400, detail="Already recording or initializing")
    
    if not robot_instances.get("follower") or not robot_instances.get("leader"):
        raise HTTPException(
            status_code=400, 
            detail="Robots not initialized. Please click 'Setup Robots' first."
        )
    
    try:
        recording_state["config"] = config.model_dump()
        recording_state["task"] = config.task
        recording_state["error"] = None
        recording_state["is_initializing"] = True
        recording_state["status_message"] = "Initializing recording..."
        
        initialize_robot_systems(config)
        
        if robot_instances.get("dataset") is None:
            raise RuntimeError("Dataset not created!")
        
        recording_state["is_initializing"] = False
        recording_state["is_recording"] = True
        recording_state["start_time"] = time.time()
        recording_state["recording_started_time"] = None
        recording_state["ramp_up_remaining"] = RAMP_UP_DURATION
        stop_recording_flag.clear()
        
        recording_thread = threading.Thread(target=record_loop_with_compensation, daemon=True)
        recording_thread.start()
        time.sleep(0.1)
        
        if not recording_thread.is_alive():
            raise RuntimeError("Recording thread failed to start")
        
        print(f"[{source}] Recording started for: {config.task}")
        return {"status": "started", "task": config.task}
    
    except Exception as e:
        recording_state["is_recording"] = False
        recording_state["is_initializing"] = False
        recording_state["error"] = str(e)
        recording_state["status_message"] = f"Error: {str(e)}"
        cleanup_robot_systems(keep_robots=True)
        raise HTTPException(status_code=500, detail=str(e))


def do_stop_recording(source: str = "API"):
    """Core logic for stopping recording (used by both API and pedal)."""
    global recording_thread, stop_recording_flag
    
    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="Not recording")
    
    recording_state["is_recording"] = False
    recording_state["status_message"] = "Stopping recording..."
    recording_state["ramp_up_remaining"] = 0.0
    recording_state["recording_started_time"] = None
    
    stop_recording_flag.set()
    if recording_thread:
        recording_thread.join(timeout=5)
    
    dataset = robot_instances.get("dataset")
    dataset_name = robot_instances.get("repo_id", "").split("/")[-1] if robot_instances.get("repo_id") else ""
    
    print(f"[{source}] Recording stopped")
    
    try:
        if dataset is not None and dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
            buffer_size = dataset.episode_buffer.get("size", 0)
            print(f"[{source}] Buffer size: {buffer_size}")
            
            recording_state["is_encoding"] = True
            recording_state["is_uploading"] = False
            recording_state["status_message"] = f"Encoding videos ({buffer_size} frames)..."
            recording_state["upload_status"] = None
            print(f"[{source}] Saving episode...")
            dataset.save_episode()
            dataset.finalize()
            recording_state["is_encoding"] = False
            recording_state["status_message"] = "Encoding complete, uploading..."
            print(f"[{source}] Episode saved")
            
            recording_state["is_uploading"] = True
            recording_state["status_message"] = "Uploading to HuggingFace..."
            recording_state["upload_status"] = "Uploading..."
            print(f"[{source}] Uploading to hub...")
            dataset.push_to_hub(private=True)
            
            recording_state["is_uploading"] = False
            recording_state["upload_status"] = "✓ Upload successful!"
            recording_state["status_message"] = "Ready"
            recording_state["episode_count"] += 1
            print(f"[{source}] Upload complete. Episode count: {recording_state['episode_count']}")
        else:
            recording_state["status_message"] = "No data"
            recording_state["upload_status"] = "No data"
            print(f"[{source}] No data to save")
        
        cleanup_robot_systems(keep_robots=True)
    
    except Exception as e:
        recording_state["is_encoding"] = False
        recording_state["is_uploading"] = False
        recording_state["error"] = f"Upload failed: {str(e)}"
        recording_state["status_message"] = f"Error: {str(e)}"
        recording_state["upload_status"] = "✗ Upload failed"
        cleanup_robot_systems(keep_robots=True)
        print(f"[{source}] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "status": "stopped",
        "dataset_name": dataset_name,
        "episode_count": recording_state["episode_count"]
    }


def do_move_to_zero(source: str = "API"):
    """Core logic for moving to zero (used by both API and pedal)."""
    if recording_state["is_recording"] or recording_state["is_initializing"]:
        raise HTTPException(status_code=400, detail="Cannot move to zero while recording")
    
    if not robot_instances.get("follower"):
        raise HTTPException(status_code=400, detail="Robots not initialized. Please setup robots first.")
    
    if recording_state["moving_to_zero"]:
        raise HTTPException(status_code=400, detail="Already moving to zero position")
    
    try:
        recording_state["moving_to_zero"] = True
        recording_state["status_message"] = "Moving to zero position..."
        
        follower = robot_instances["follower"]
        leader = robot_instances.get("leader")
        leader_type = robot_instances.get("leader_type", "openarms")
        
        # Handle leader based on type
        if leader and leader_type == "openarms":
            # OpenArms: Disable torque (with gravity compensation)
            print(f"[{source}] Disabling leader torque (OpenArms with gravity compensation)...")
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
        elif leader and leader_type == "openarms_mini":
            # OpenArms Mini: Send to zero position command (will be executed in parallel with follower)
            print(f"[{source}] Moving OpenArms Mini leader to zero position...")
        
        motor_index = {
            "joint_1": 0, "joint_2": 1, "joint_3": 2, "joint_4": 3,
            "joint_5": 4, "joint_6": 5, "joint_7": 6, "gripper": 7,
        }
        
        # Prepare follower zero action
        follower_zero_action = {}
        follower_custom_kp = {}
        follower_custom_kd = {}
        
        for motor in follower.bus_right.motors:
            full_name = f"right_{motor}"
            follower_zero_action[f"{full_name}.pos"] = 0.0
            idx = motor_index.get(motor, 0)
            kp_base = follower.config.position_kp[idx] if isinstance(follower.config.position_kp, list) else follower.config.position_kp
            kd_base = follower.config.position_kd[idx] if isinstance(follower.config.position_kd, list) else follower.config.position_kd
            follower_custom_kp[full_name] = kp_base * 0.05
            follower_custom_kd[full_name] = kd_base
        
        for motor in follower.bus_left.motors:
            full_name = f"left_{motor}"
            follower_zero_action[f"{full_name}.pos"] = 0.0
            idx = motor_index.get(motor, 0)
            kp_base = follower.config.position_kp[idx] if isinstance(follower.config.position_kp, list) else follower.config.position_kp
            kd_base = follower.config.position_kd[idx] if isinstance(follower.config.position_kd, list) else follower.config.position_kd
            follower_custom_kp[full_name] = kp_base * 0.05
            follower_custom_kd[full_name] = kd_base
        
        # Prepare leader zero action (for OpenArms Mini)
        leader_zero_action = None
        if leader and leader_type == "openarms_mini":
            leader_zero_action = {}
            # For OpenArms Mini, send all joints (including gripper) to zero
            # Gripper zero = 0 (closed)
            for motor in leader.bus_right.motors:
                full_name = f"right_{motor}"
                leader_zero_action[f"{full_name}.pos"] = 0.0
            for motor in leader.bus_left.motors:
                full_name = f"left_{motor}"
                leader_zero_action[f"{full_name}.pos"] = 0.0
        
        duration = 2.0
        fps = 30
        dt = 1 / fps
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            loop_start = time.perf_counter()
            
            # Send follower to zero
            follower.send_action(follower_zero_action, custom_kp=follower_custom_kp, custom_kd=follower_custom_kd)
            
            # Send leader (OpenArms Mini) to zero if applicable
            if leader_zero_action:
                # Use Feetech write commands directly (no custom gains needed)
                for motor_name in leader.bus_right.motors:
                    full_name = f"right_{motor_name}"
                    pos_key = f"{full_name}.pos"
                    if pos_key in leader_zero_action:
                        # Write zero position to leader motor
                        leader.bus_right.write("Goal_Position", motor_name, leader_zero_action[pos_key])
                
                for motor_name in leader.bus_left.motors:
                    full_name = f"left_{motor_name}"
                    pos_key = f"{full_name}.pos"
                    if pos_key in leader_zero_action:
                        # Write zero position to leader motor
                        leader.bus_left.write("Goal_Position", motor_name, leader_zero_action[pos_key])
            
            loop_duration = time.perf_counter() - loop_start
            sleep_time = dt - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Re-enable torque only for OpenArms (with gravity compensation)
        if leader and leader_type == "openarms":
            print(f"[{source}] Re-enabling leader torque (OpenArms with gravity compensation)...")
            leader.bus_right.enable_torque()
            leader.bus_left.enable_torque()
            time.sleep(0.1)
        elif leader and leader_type == "openarms_mini":
            print(f"[{source}] OpenArms Mini leader moved to zero position")
        
        recording_state["moving_to_zero"] = False
        recording_state["status_message"] = "Moved to zero position" if recording_state["robots_ready"] else "Ready"
        
        print(f"[{source}] Move to zero complete (follower + leader)")
        return {"status": "success", "message": "Robots moved to zero position"}
    
    except Exception as e:
        recording_state["moving_to_zero"] = False
        recording_state["status_message"] = f"Error: {str(e)}"
        
        # Try to re-enable torque on error (only for OpenArms with gravity compensation)
        leader = robot_instances.get("leader")
        leader_type = robot_instances.get("leader_type", "openarms")
        if leader and leader_type == "openarms":
            try:
                print(f"[{source}] Attempting to re-enable leader torque after error...")
                leader.bus_right.enable_torque()
                leader.bus_left.enable_torque()
            except Exception as torque_error:
                print(f"[{source}] Failed to re-enable leader torque: {torque_error}")
        
        print(f"[{source}] Move to zero failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def pedal_reader():
    """Read foot pedal events and trigger actions.
    
    Key mappings (configurable):
    - KEY_B (left pedal) -> Start recording
    - KEY_N (middle pedal) -> Move to zero
    - KEY_M (right pedal) -> Stop recording
    """
    if not PEDAL_AVAILABLE:
        print(f"[Pedal] evdev not available, pedal support disabled")
        return
    
    # Key mapping - adjust these if your pedal sends different keys
    KEY_START_RECORDING = "KEY_A"    # Left pedal
    KEY_MOVE_TO_ZERO = "KEY_B"       # Middle pedal
    KEY_STOP_RECORDING = "KEY_C"     # Right pedal
    
    try:
        dev = InputDevice(PEDAL_DEVICE)
        print(f"[Pedal] Using {dev.path} ({dev.name})")
        print(f"[Pedal] Key mapping: {KEY_START_RECORDING}=Start | {KEY_MOVE_TO_ZERO}=Zero | {KEY_STOP_RECORDING}=Stop")
        
        for ev in dev.read_loop():
            if stop_pedal_flag.is_set():
                break
            
            if ev.type != ecodes.EV_KEY:
                continue
            
            key = categorize(ev)
            code = key.keycode
            if isinstance(code, (list, tuple)):
                code = code[0]
            
            # Only trigger on key down (not up or hold)
            if key.keystate != 1:  # 1 = DOWN
                continue
            
            print(f"[Pedal] Key pressed: {code}")
            
            # Use lock to prevent concurrent pedal actions
            with pedal_action_lock:
                try:
                    if code == KEY_START_RECORDING:
                        # Check if task is set before starting
                        if not recording_state.get("task"):
                            print(f"[Pedal] Cannot start: no task set")
                            recording_state["error"] = "No task set. Please enter task in web UI."
                            continue
                        config = RecordingConfig(**recording_state["config"])
                        do_start_recording(config, source="Pedal")
                    
                    elif code == KEY_MOVE_TO_ZERO:
                        do_move_to_zero(source="Pedal")
                    
                    elif code == KEY_STOP_RECORDING:
                        do_stop_recording(source="Pedal")
                    
                    else:
                        print(f"[Pedal] Unmapped key: {code}")
                
                except HTTPException as e:
                    # Handle validation errors gracefully (don't crash pedal thread)
                    print(f"[Pedal] Action blocked: {e.detail}")
                except Exception as e:
                    print(f"[Pedal] Action error: {e}")
    
    except FileNotFoundError:
        print(f"[Pedal] Device not found: {PEDAL_DEVICE}")
        print(f"[Pedal] Pedal support disabled. Connect pedal and restart server.")
    except PermissionError:
        print(f"[Pedal] Permission denied: {PEDAL_DEVICE}")
        print(f"[Pedal] Run: sudo setfacl -m u:$USER:rw {PEDAL_DEVICE}")
    except Exception as e:
        print(f"[Pedal] Error: {e}")
        import traceback
        traceback.print_exc()


@app.post("/api/robots/setup")
async def setup_robots(config: RobotSetupConfig):
    """Pre-initialize robots for faster recording start."""
    global recording_state, robot_instances

    if recording_state["is_recording"] or recording_state["is_initializing"]:
        raise HTTPException(status_code=400, detail="Cannot setup robots while recording")

    if recording_state["robots_ready"] and robot_instances.get("follower") and robot_instances.get("leader"):
        raise HTTPException(status_code=400, detail="Robots already initialized")

    # Clean up any existing robots first
    if robot_instances.get("follower") or robot_instances.get("leader"):
        cleanup_robot_systems()

    try:
        recording_state["error"] = None
        recording_state["status_message"] = "Setting up robots..."

        # Update config (add empty task for compatibility)
        config_dict = config.model_dump()
        config_dict["task"] = ""  # Task not needed for setup
        recording_state["config"] = config_dict

        # Initialize robot systems (without dataset creation)
        initialize_robots_only(config)

        recording_state["robots_ready"] = True
        recording_state["status_message"] = "Robots ready for recording"

        return {"status": "ready", "message": "Robots initialized successfully"}

    except Exception as e:
        recording_state["error"] = str(e)
        recording_state["status_message"] = f"Setup failed"
        recording_state["robots_ready"] = False
        print(f"[Setup] Error: {e}")
        cleanup_robot_systems()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/robots/disconnect")
async def disconnect_robots():
    """Disconnect pre-initialized robots."""
    global recording_state

    if recording_state["is_recording"] or recording_state["is_initializing"]:
        raise HTTPException(status_code=400, detail="Cannot disconnect while recording")

    recording_state["robots_ready"] = False
    recording_state["status_message"] = "Robots disconnected"
    cleanup_robot_systems(keep_robots=False)  # Actually disconnect robots

    return {"status": "disconnected"}


@app.post("/api/robots/move-to-zero")
async def move_to_zero():
    """Move follower robot to zero position with reduced gains (for 2 seconds).
    
    Also disables leader torque during the movement for safety and ease of repositioning.
    """
    return do_move_to_zero(source="API")


@app.get("/api/status")
async def get_status():
    """Get current recording status."""
    elapsed = 0
    # Only show elapsed time after ramp-up completes
    if recording_state["is_recording"] and recording_state["recording_started_time"]:
        elapsed = time.time() - recording_state["recording_started_time"]

    return {
        "is_recording": recording_state["is_recording"],
        "is_initializing": recording_state["is_initializing"],
        "is_encoding": recording_state["is_encoding"],
        "is_uploading": recording_state["is_uploading"],
        "robots_ready": recording_state["robots_ready"],
        "elapsed_time": elapsed,
        "current_fps": recording_state["current_fps"],
        "loop_fps": recording_state["loop_fps"],
        "task": recording_state["task"],
        "episode_count": recording_state["episode_count"],
        "error": recording_state["error"],
        "status_message": recording_state["status_message"],
        "upload_status": recording_state["upload_status"],
        "ramp_up_remaining": recording_state["ramp_up_remaining"],
        "moving_to_zero": recording_state["moving_to_zero"],
        "config": recording_state["config"]
    }


@app.post("/api/counter/reset")
async def reset_counter():
    """Reset the episode counter."""
    recording_state["episode_count"] = 0
    return {"episode_count": 0}


@app.post("/api/counter/set")
async def set_counter(update: CounterUpdate):
    """Set the episode counter value."""
    recording_state["episode_count"] = update.value
    return {"episode_count": update.value}


@app.get("/api/camera/stream/{camera_name}")
async def stream_camera(camera_name: str):
    """Stream camera feed from robot at 10 FPS.
    
    During recording: streams raw frames from the recording loop (no extra camera reads!)
    When not recording: streams directly from cameras.
    
    Encoding happens here at 10 FPS to avoid slowing down 30 Hz control loop.
    """
    def generate():
        try:
            while True:
                frame_rgb = None
                
                # If recording, get raw frames from recording loop
                if recording_state["is_recording"]:
                    current_time = time.perf_counter()
                    
                    # Quick read of raw frame with minimal lock time
                    with camera_frames_lock:
                        frame_rgb = camera_frames_raw.get(camera_name)
                        frame_timestamp = camera_frames_timestamp.get(camera_name, 0.0)
                    
                    # Check frame freshness
                    frame_age = current_time - frame_timestamp
                    
                    if frame_rgb is not None and frame_age <= MAX_FRAME_AGE:
                        # Frame is fresh - encode it now (only when sending to client)
                        try:
                            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                            time.sleep(0.1)  # 10 FPS streaming
                        except Exception as encode_error:
                            print(f"[Camera] Encode error: {encode_error}")
                            time.sleep(0.01)
                    else:
                        # Frame is stale or not available - wait for next frame
                        time.sleep(0.01)
                        continue
                        
                else:
                    # Not recording: stream directly from camera at 10 FPS
                    follower = robot_instances.get("follower")
                    if not follower or not follower.cameras:
                        break
                    
                    if camera_name not in follower.cameras:
                        break
                    
                    camera = follower.cameras[camera_name]
                    frame = camera.async_read(timeout_ms=50)
                    
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    
                    # Convert BGR to RGB and encode
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    time.sleep(0.1)  # 10 FPS when not recording
                    
        except Exception as e:
            print(f"[Camera] Stream error: {e}")
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize pedal monitoring on startup."""
    global pedal_thread
    
    if PEDAL_ENABLED and PEDAL_AVAILABLE:
        print(f"[Startup] Starting pedal monitor...")
        stop_pedal_flag.clear()
        pedal_thread = threading.Thread(target=pedal_reader, daemon=True)
        pedal_thread.start()
        print(f"[Startup] Pedal monitor started")
    else:
        if not PEDAL_ENABLED:
            print(f"[Startup] Pedal disabled in configuration")
        elif not PEDAL_AVAILABLE:
            print(f"[Startup] Pedal unavailable (evdev not installed)")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global stop_recording_flag, stop_pedal_flag
    
    print(f"[Shutdown] Stopping recording...")
    stop_recording_flag.set()
    
    print(f"[Shutdown] Stopping pedal monitor...")
    stop_pedal_flag.set()
    if pedal_thread and pedal_thread.is_alive():
        pedal_thread.join(timeout=2)
    
    print(f"[Shutdown] Cleaning up robots...")
    cleanup_robot_systems()
    print(f"[Shutdown] Complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
