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

# LeRobot imports
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader

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
# Store pre-encoded JPEG bytes with timestamps to track freshness and discard stale frames
camera_frames_jpeg = {
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
MAX_FRAME_AGE = 0.2 
# Recording control
recording_thread = None
stop_recording_flag = threading.Event()
FPS = 30
FRICTION_SCALE = 1.0
RAMP_UP_DURATION = 3.0  # Seconds to ramp up PID/torque from 0 to full


class RecordingConfig(BaseModel):
    task: str
    leader_left: str
    leader_right: str
    follower_left: str
    follower_right: str
    left_wrist: str
    right_wrist: str
    base: str


class RobotSetupConfig(BaseModel):
    """Configuration for robot setup (no task required)."""
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


@app.get("/api/cameras/discover")
async def discover_cameras():
    """Discover available cameras."""
    cameras = discover_cameras_sync()
    return {"cameras": cameras}


@app.get("/api/can/interfaces")
async def get_can_interfaces():
    """Get available CAN interfaces."""
    return {"interfaces": ["can0", "can1", "can2", "can3"]}


def initialize_robots_only(config: RobotSetupConfig):
    """Initialize robots only (no dataset) for pre-setup."""
    global robot_instances, recording_state
    
    print(f"[Setup] Initializing robots...")

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

    # Configure leader teleoperator
    recording_state["status_message"] = "Configuring leader teleoperator..."
    leader_config = OpenArmsLeaderConfig(
        port_left=config.leader_left,
        port_right=config.leader_right,
        can_interface="socketcan",
        id="openarms_leader",
        manual_control=False,
    )

    # Initialize and connect
    recording_state["status_message"] = "Connecting to follower robot..."
    follower = OpenArmsFollower(follower_config)
    follower.connect(calibrate=False)  # Skip calibration in web mode

    recording_state["status_message"] = "Connecting to leader teleoperator..."
    leader = OpenArmsLeader(leader_config)
    leader.connect(calibrate=False)

    # Verify URDF is loaded
    recording_state["status_message"] = "Loading URDF model for gravity compensation..."
    if leader.pin_robot is None:
        raise RuntimeError("URDF model not loaded on leader. Gravity compensation not available.")

    # Enable gravity compensation
    recording_state["status_message"] = "Enabling gravity compensation..."
    leader.bus_right.enable_torque()
    leader.bus_left.enable_torque()
    time.sleep(0.1)

    robot_instances["follower"] = follower
    robot_instances["leader"] = leader


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

        # Configure leader teleoperator
        recording_state["status_message"] = "Configuring leader teleoperator..."
        leader_config = OpenArmsLeaderConfig(
            port_left=config.leader_left,
            port_right=config.leader_right,
            can_interface="socketcan",
            id="openarms_leader",
            manual_control=False,
        )

        # Initialize and connect
        recording_state["status_message"] = "Connecting to follower robot..."
        follower = OpenArmsFollower(follower_config)
        follower.connect(calibrate=False)  # Skip calibration in web mode

        recording_state["status_message"] = "Connecting to leader teleoperator..."
        leader = OpenArmsLeader(leader_config)
        leader.connect(calibrate=False)

        # Verify URDF is loaded
        recording_state["status_message"] = "Loading URDF model for gravity compensation..."
        if leader.pin_robot is None:
            raise RuntimeError("URDF model not loaded on leader. Gravity compensation not available.")

        # Enable gravity compensation
        recording_state["status_message"] = "Enabling gravity compensation..."
        leader.bus_right.enable_torque()
        leader.bus_left.enable_torque()
        time.sleep(0.1)

    # Configure dataset features
    recording_state["status_message"] = "Configuring dataset features..."
    action_features_hw = {}
    for key, value in follower.action_features.items():
        if key.endswith(".pos"):
            action_features_hw[key] = value

    action_features = hw_to_dataset_features(action_features_hw, "action")
    obs_features = hw_to_dataset_features(follower.observation_features, "observation")
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
            image_writer_processes=0,  # Use threads only (faster for single process)
            image_writer_threads=8,  # More threads for 3 cameras (2-3 per camera)
        )
        print(f"[Dataset] Created: {repo_id}")
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
    """Main recording loop with compensation."""
    global recording_state, stop_recording_flag
    
    follower = robot_instances.get("follower")
    leader = robot_instances.get("leader")
    dataset = robot_instances.get("dataset")
    dataset_features = robot_instances.get("dataset_features")
    task = recording_state.get("task", "")
    
    if follower is None or leader is None or dataset is None:
        recording_state["error"] = "Robot or dataset not initialized"
        print(f"[Recording] Error: Missing components")
        return
    
    print(f"[Recording] Starting recording loop with ramp-up...")
    
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
            
            # More aggressive skip logic: skip encoding if loop is taking >90% of frame time
            # OR if previous loop was slow
            if hasattr(record_loop_with_compensation, '_last_loop_duration'):
                skip_encoding = record_loop_with_compensation._last_loop_duration > dt * 0.9
            else:
                skip_encoding = False
            
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
            leader_action = leader.get_action()
            
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
            leader_gravity_torques_nm = leader._gravity_from_q(leader_positions_rad)
            
            leader_velocities_rad_per_sec = {k: np.deg2rad(v) for k, v in leader_velocities_deg_per_sec.items()}
            leader_friction_torques_nm = leader._friction_from_velocity(
                leader_velocities_rad_per_sec,
                friction_scale=FRICTION_SCALE
            )
            
            # Combine torques
            leader_total_torques_nm = {}
            for motor_name in leader_gravity_torques_nm:
                gravity = leader_gravity_torques_nm.get(motor_name, 0.0)
                friction = leader_friction_torques_nm.get(motor_name, 0.0)
                leader_total_torques_nm[motor_name] = gravity + friction
            
            # Apply compensation to RIGHT arm with ramp-up scaling
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
            
            # Send positions to follower with ramped PID gains using send_action
            follower_action = {}
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
                follower.send_action(follower_action, custom_kp=custom_kp, custom_kd=custom_kd)
            
            # Get observation
            observation = follower.get_observation()
            
            # Get current timestamp for frame freshness tracking
            frame_timestamp = time.perf_counter()
            
            # Pre-encode camera frames as JPEG for streaming (minimizes lock time and encoding overhead)
            # This avoids encoding the same frame multiple times for multiple stream endpoints
            # OpenCV cameras configured with ColorMode.RGB return RGB frames (default config)
            # Skip encoding if we're falling behind to maintain 30 FPS
            encoded_frames = {}
            
            # Only encode if we're not falling behind
            # PRIORITY: 30 Hz control loop > streaming quality
            if not skip_encoding:
                for cam_name in ["left_wrist", "right_wrist", "base"]:
                    if cam_name in observation:
                        frame = observation[cam_name]
                        if frame is not None and len(frame.shape) == 3:
                            # Frame is already RGB from OpenCVCamera (ColorMode.RGB by default)
                            # Encode once, use many times - eliminates per-stream encoding overhead
                            # Use low quality (40) for fast streaming - control loop is priority
                            encode_start = time.perf_counter()
                            success, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                            encode_duration = time.perf_counter() - encode_start
                            
                            # Only use encoded frame if encoding was very fast (< 5ms = 15% of frame time)
                            if success and encode_duration < 0.005:
                                encoded_frames[cam_name] = jpeg_bytes.tobytes()
                            # If encoding took too long, skip this camera for this frame
            
            # Store encoded frames with timestamps with minimal lock time
            with camera_frames_lock:
                for cam_name, jpeg_bytes in encoded_frames.items():
                    camera_frames_jpeg[cam_name] = jpeg_bytes
                    camera_frames_timestamp[cam_name] = frame_timestamp
            
            # Add to dataset ONLY after ramp-up completes
            if is_ramp_up_complete:
                try:
                    obs_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
                    action_frame = build_dataset_frame(dataset_features, follower_action, prefix="action")
                    frame = {**obs_frame, **action_frame}
                    frame["task"] = task
                    dataset.add_frame(frame)
                    frame_count += 1
                    
                    # Log progress every 5 seconds (150 frames @ 30 FPS)
                    if frame_count % 150 == 0:
                        print(f"[Recording] {frame_count} frames | Loop: {recording_state['loop_fps']} Hz | Recording: {recording_state['current_fps']} FPS")
                except Exception as frame_error:
                    print(f"[Recording] Frame error: {frame_error}")
                    # Continue recording even if one frame fails
            
            # Maintain loop rate - don't wait if behind, just continue
            loop_duration = time.perf_counter() - loop_start
            
            # Track loop duration for next iteration's encoding skip decision
            record_loop_with_compensation._last_loop_duration = loop_duration
            
            sleep_time = dt - loop_duration
            
            # Only sleep if we're ahead of schedule (more than 2ms remaining)
            if sleep_time > 0.002:
                time.sleep(min(sleep_time, dt * 0.5))  # Never sleep more than half a frame time
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


@app.post("/api/recording/start")
async def start_recording(config: RecordingConfig):
    """Start recording an episode."""
    global recording_state, recording_thread, stop_recording_flag
    
    if recording_state["is_recording"] or recording_state["is_initializing"]:
        raise HTTPException(status_code=400, detail="Already recording or initializing")
    
    # Check if robots are available (either from setup or previous recording)
    if not robot_instances.get("follower") or not robot_instances.get("leader"):
        raise HTTPException(
            status_code=400, 
            detail="Robots not initialized. Please click 'Setup Robots' first."
        )
    
    try:
        # Update config
        recording_state["config"] = config.model_dump()
        recording_state["task"] = config.task
        recording_state["error"] = None
        recording_state["is_initializing"] = True
        recording_state["status_message"] = "Initializing recording..."
        
        # Initialize robot systems (will reuse pre-initialized robots if available)
        initialize_robot_systems(config)
        
        if robot_instances.get("dataset") is None:
            raise RuntimeError("Dataset not created!")
        
        # Start recording
        recording_state["is_initializing"] = False
        recording_state["is_recording"] = True
        recording_state["start_time"] = time.time()
        recording_state["recording_started_time"] = None  # Will be set after ramp-up
        recording_state["ramp_up_remaining"] = RAMP_UP_DURATION  # Initialize ramp-up
        stop_recording_flag.clear()  # Clear the stop flag
        
        # Start recording in background thread
        recording_thread = threading.Thread(target=record_loop_with_compensation, daemon=True)
        recording_thread.start()
        time.sleep(0.1)
        
        if not recording_thread.is_alive():
            raise RuntimeError("Recording thread failed to start")
        
        print(f"[Start] Recording started for: {config.task}")
        return {"status": "started", "task": config.task}
    
    except Exception as e:
        recording_state["is_recording"] = False
        recording_state["is_initializing"] = False
        recording_state["error"] = str(e)
        recording_state["status_message"] = f"Error: {str(e)}"
        # Clean up dataset if initialization failed, but keep robots
        cleanup_robot_systems(keep_robots=True)
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/recording/stop")
async def stop_recording():
    """Stop recording, encode, and upload episode."""
    global recording_state, recording_thread, stop_recording_flag
    
    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="Not recording")
    
    recording_state["is_recording"] = False
    recording_state["status_message"] = "Stopping recording..."
    recording_state["ramp_up_remaining"] = 0.0  # Reset ramp-up
    recording_state["recording_started_time"] = None  # Reset recording start time
    
    # Stop the recording thread
    stop_recording_flag.set()
    if recording_thread:
        recording_thread.join(timeout=5)
    
        dataset = robot_instances.get("dataset")
    dataset_name = robot_instances.get("repo_id", "").split("/")[-1] if robot_instances.get("repo_id") else ""
    
    print(f"[Stop] Recording stopped")
    print(f"[Stop] Dataset is None: {dataset is None}")
    if dataset:
        print(f"[Stop] Dataset type: {type(dataset)}")
        print(f"[Stop] Episode buffer is None: {dataset.episode_buffer is None}")
        print(f"[Stop] Episode buffer value: {dataset.episode_buffer}")
        if dataset.episode_buffer:
            print(f"[Stop] Episode buffer keys: {list(dataset.episode_buffer.keys())}")
            print(f"[Stop] Episode buffer type: {type(dataset.episode_buffer)}")
            print(f"[Stop] Episode buffer size: {dataset.episode_buffer.get('size', 'NO SIZE KEY')}")
    
    # Process episode immediately (blocking)
    try:
        # Check buffer the same way as reference implementation
        if dataset is not None and dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
            buffer_size = dataset.episode_buffer.get("size", 0)
            print(f"[Stop] Buffer size: {buffer_size}")
            
            # Encode videos
            recording_state["is_encoding"] = True
            recording_state["is_uploading"] = False  # Ensure upload flag is clear
            recording_state["status_message"] = f"Encoding videos ({buffer_size} frames)..."
            recording_state["upload_status"] = None  # Clear upload status
            print(f"[Stop] Saving episode...")
            dataset.save_episode()
            dataset.finalize()
            recording_state["is_encoding"] = False
            recording_state["status_message"] = "Encoding complete, uploading..."
            print(f"[Stop] Episode saved")
            
            # Upload to hub
            recording_state["is_uploading"] = True
            recording_state["status_message"] = "Uploading to HuggingFace..."
            recording_state["upload_status"] = "Uploading..."
            print(f"[Stop] Uploading to hub...")
            dataset.push_to_hub(private=True)
            
            recording_state["is_uploading"] = False
            recording_state["upload_status"] = "✓ Upload successful!"
            recording_state["status_message"] = "Ready"
            recording_state["episode_count"] += 1
            print(f"[Stop] Upload complete. Episode count: {recording_state['episode_count']}")
        else:
            recording_state["status_message"] = "No data"
            recording_state["upload_status"] = "No data"
            print(f"[Stop] No dataset or buffer (dataset={dataset is not None}, buffer={dataset.episode_buffer is not None if dataset else 'N/A'}, size={dataset.episode_buffer.get('size', 0) if dataset and dataset.episode_buffer else 'N/A'})")
        
        # Clean up dataset, keep robots
        cleanup_robot_systems(keep_robots=True)
    
    except Exception as e:
        recording_state["is_encoding"] = False
        recording_state["is_uploading"] = False
        recording_state["error"] = f"Upload failed: {str(e)}"
        recording_state["status_message"] = f"Error: {str(e)}"
        recording_state["upload_status"] = f"✗ Upload failed"
        cleanup_robot_systems(keep_robots=True)
        print(f"[Stop] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "status": "stopped",
        "dataset_name": dataset_name,
        "episode_count": recording_state["episode_count"]
    }


def cleanup_robot_systems(keep_robots=False):
    """Clean up robot systems."""
    global robot_instances
    
    try:
        if not keep_robots:
            if robot_instances.get("leader"):
                    robot_instances["leader"].bus_right.disable_torque()
                    robot_instances["leader"].bus_left.disable_torque()
                    time.sleep(0.1)
                    robot_instances["leader"].disconnect()
            
            if robot_instances.get("follower"):
                    robot_instances["follower"].disconnect()
            
            robot_instances["follower"] = None
            robot_instances["leader"] = None
            recording_state["robots_ready"] = False
            print(f"[Cleanup] Robots disconnected")
        
        # Always clean up dataset
        robot_instances["dataset"] = None
        robot_instances["dataset_features"] = None
        robot_instances["repo_id"] = None
    
    except Exception as e:
        print(f"[Cleanup] Error: {e}")


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
    global recording_state
    
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
        
        # Disable leader torque for safety and ease of repositioning
        if leader:
            print(f"[MoveToZero] Disabling leader torque...")
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
        
        # Motor name to index mapping
        motor_index = {
            "joint_1": 0, "joint_2": 1, "joint_3": 2, "joint_4": 3,
            "joint_5": 4, "joint_6": 5, "joint_7": 6, "gripper": 7,
        }
        
        # Build zero position action with reduced gains (20% of original)
        zero_action = {}
        custom_kp = {}
        custom_kd = {}
        
        # Right arm - all motors to zero
        for motor in follower.bus_right.motors:
            full_name = f"right_{motor}"
            zero_action[f"{full_name}.pos"] = 0.0
            
            idx = motor_index.get(motor, 0)
            kp_base = follower.config.position_kp[idx] if isinstance(follower.config.position_kp, list) else follower.config.position_kp
            kd_base = follower.config.position_kd[idx] if isinstance(follower.config.position_kd, list) else follower.config.position_kd
            
            custom_kp[full_name] = kp_base * 0.05
            custom_kd[full_name] = kd_base
        
        # Left arm - all motors to zero
        for motor in follower.bus_left.motors:
            full_name = f"left_{motor}"
            zero_action[f"{full_name}.pos"] = 0.0
            
            idx = motor_index.get(motor, 0)
            kp_base = follower.config.position_kp[idx] if isinstance(follower.config.position_kp, list) else follower.config.position_kp
            kd_base = follower.config.position_kd[idx] if isinstance(follower.config.position_kd, list) else follower.config.position_kd
            
            custom_kp[full_name] = kp_base * 0.05
            custom_kd[full_name] = kd_base
        
        # Send zero position commands for 2 seconds at 30 FPS
        duration = 2.0
        fps = 30
        dt = 1 / fps
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            loop_start = time.perf_counter()
            
            # Send action with reduced gains
            follower.send_action(zero_action, custom_kp=custom_kp, custom_kd=custom_kd)
            
            # Maintain loop rate
            loop_duration = time.perf_counter() - loop_start
            sleep_time = dt - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Re-enable leader torque for gravity compensation
        if leader:
            print(f"[MoveToZero] Re-enabling leader torque...")
            leader.bus_right.enable_torque()
            leader.bus_left.enable_torque()
            time.sleep(0.1)
        
        recording_state["moving_to_zero"] = False
        recording_state["status_message"] = "Moved to zero position" if recording_state["robots_ready"] else "Ready"
        
        print(f"[MoveToZero] Complete - leader torque re-enabled")
        return {"status": "success", "message": "Robot moved to zero position"}
    
    except Exception as e:
        recording_state["moving_to_zero"] = False
        recording_state["status_message"] = f"Error: {str(e)}"
        
        # Try to re-enable leader torque even on error
        leader = robot_instances.get("leader")
        if leader:
            try:
                leader.bus_right.enable_torque()
                leader.bus_left.enable_torque()
            except Exception as torque_error:
                print(f"[MoveToZero] Failed to re-enable leader torque: {torque_error}")
        
        raise HTTPException(status_code=500, detail=str(e))


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
    """Stream camera feed from robot.
    
    During recording: streams frames from the recording loop (no extra camera reads = no contention!)
    When not recording: streams directly from cameras.
    """
    def generate():
        try:
            while True:
                frame = None
                
                # If recording, use pre-encoded JPEG frames from recording loop (zero encoding, minimal lock time!)
                if recording_state["is_recording"]:
                    # Acquire lock only to read the reference and timestamp, not to encode
                    jpeg_bytes = None
                    frame_timestamp = 0.0
                    current_time = time.perf_counter()
                    
                    with camera_frames_lock:
                        jpeg_bytes = camera_frames_jpeg.get(camera_name)
                        frame_timestamp = camera_frames_timestamp.get(camera_name, 0.0)
                    
                    # Only serve frames that are fresh (not stale)
                    frame_age = current_time - frame_timestamp
                    if jpeg_bytes is not None and frame_age <= MAX_FRAME_AGE:
                        # Frame is fresh and available - yield it
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                        time.sleep(0.1)  # ~10 FPS for streaming (independent of recording FPS)
                    elif jpeg_bytes is not None and frame_age > MAX_FRAME_AGE:
                        # Frame is stale - skip it, wait a bit and try next frame
                        time.sleep(0.01)
                        continue
                    else:
                        # No frame available yet - wait a bit
                        time.sleep(0.01)
                        continue
                else:
                    # Not recording: stream directly from camera
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
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    time.sleep(0.05)  # ~20 FPS when not recording
                    
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


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global stop_recording_flag
    
    stop_recording_flag.set()
    cleanup_robot_systems()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
