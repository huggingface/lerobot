#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MCP server to control LeRobot robots with intuitive commands.

Example usage:
```shell
python mcp_robot_server.py \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=/dev/ttyUSB0 \
    --robot.right_arm_port=/dev/ttyUSB1
```
"""

import os
import atexit
import io
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import draccus
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from mcp.server.fastmcp import FastMCP, Image
from lerobot.robots.bi_koch_follower.bi_koch_follower import BiKochFollower
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.bi_koch_follower.config_bi_koch_follower import (
    make_bimanual_koch_robot_processors,
    BiKochFollowerConfig,
)
from lerobot.teleoperators.bi_koch_leader.config_bi_koch_leader import make_bimanual_koch_teleop_processors
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.async_inference.helpers import (
    get_logger,
)
from lerobot.async_inference.bimanual_koch_utils import (
    INITIAL_EE_POSE,
    action_dict_to_tensor,
    action_tensor_to_dict,
    compute_current_ee,
    generate_linear_trajectory,
    get_bimanual_action_features,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, log_rerun_action_chunk

# Configure logging
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class MCPRobotServerConfig:
    """Configuration for MCP Robot Server."""

    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    robot_description: str = field(
        default="This is a bimanual Koch robot arm with inverse kinematics control. "
        "You can control the end effector position and gripper using intuitive commands.",
        metadata={"help": "Robot description for LLM context"},
    )
    transport: str = field(
        default="stdio",
        metadata={"help": "Transport protocol to use (stdio or sse)"},
    )
    environment_dt: float = field(
        default=0.05,
        metadata={"help": "Control loop time step in seconds (default: 0.05 = 20 Hz)"},
    )


# -----------------------------------------------------------------------------
# Initialize FastMCP server
# -----------------------------------------------------------------------------

mcp = FastMCP(name="LeRobot MCP Controller", port=3765)

# -----------------------------------------------------------------------------
# Robot Controller using LeRobot infrastructure
# -----------------------------------------------------------------------------

# ============================================================================
# COORDINATE FRAME DEFINITIONS
# ============================================================================
# We use three coordinate frames:
#
# 1. INTUITIVE FRAME (external, for LLM interface):
#    - forward: toward camera/front of robot
#    - right: to the robot's right side
#    - up: against gravity, toward ceiling
#
# 2. WORLD FRAME (internal, for IK/FK/bounds):
#    - X: points BACKWARD (away from camera) - workspace is X: -0.25 to 0.0
#    - Y: points RIGHT
#    - Z: points UP
#
#    Note: The world X-axis is INVERTED from intuitive forward direction!
#
# 3. GRIPPER FRAME (internal, for gripper-relative movements):
#    - X: points backward (behind gripper)
#    - Y: points down (through gripper palm)
#    - Z: points right (to gripper's right)
#
# All transformations between frames are handled by functions below.
# ============================================================================


def intuitive_to_world_translation(forward: float, right: float, up: float) -> tuple[float, float, float]:
    """Transform intuitive frame translation to world frame.

    Args:
        forward: Movement toward camera (mm)
        right: Movement to robot's right (mm)
        up: Movement upward against gravity (mm)

    Returns:
        Tuple of (x, y, z) in world frame (mm)
    """
    # BUG FIX: World X is inverted - negative X is toward camera (workspace is X: -0.25 to 0.0)
    # forward (intuitive, toward camera) = -X (world)
    # right (intuitive) = Y (world)
    # up (intuitive) = Z (world)
    return -forward, right, up


def world_to_intuitive_position(x: float, y: float, z: float) -> dict[str, str]:
    """Transform world frame position to intuitive frame for display.

    Args:
        x: World X coordinate (meters)
        y: World Y coordinate (meters)
        z: World Z coordinate (meters)

    Returns:
        Dictionary with forward/right/up positions formatted as strings
    """
    # BUG FIX: World X is inverted - negative X is toward camera
    # forward (intuitive, toward camera) = -X (world)
    # right (intuitive) = Y (world)
    # up (intuitive) = Z (world)
    return {
        "forward": f"{-x:.4f}m",  # world -X is forward (intuitive)
        "right": f"{y:.4f}m",     # world +Y is right
        "up": f"{z:.4f}m"         # world +Z is up
    }


def intuitive_to_gripper_translation(forward: float, right: float, up: float) -> np.ndarray:
    """Transform intuitive frame translation to gripper frame.

    Args:
        forward: Movement in gripper's approach direction (mm)
        right: Movement to gripper's right (mm)
        up: Movement perpendicular to gripper palm (mm)

    Returns:
        np.ndarray of [x, y, z] in gripper frame (mm)
    """
    x = -forward  # forward in intuitive = -X in gripper
    y = -up       # up in intuitive = -Y in gripper
    z = right     # right in intuitive = +Z in gripper
    return np.array([x, y, z])


class LeRobotController:
    """Robot controller using LeRobot's IK and control infrastructure."""

    def __init__(self, config: MCPRobotServerConfig):
        self.config = config
        self.environment_dt = config.environment_dt
        logger.info("Initializing robot connection...")

        # Create and connect robot
        config.robot.id = "bimanual_follower"
        self.robot = BiKochFollower(config.robot)
        self.robot.connect()
        logger.info("Robot connected successfully")

        # Initialize rerun for visualization
        init_rerun(session_name="mcp_robot_server")

        # Create FK/IK processors
        self.robot_action_processor = make_bimanual_koch_robot_processors(self.robot, False)
        self.teleop_action_processor = make_bimanual_koch_teleop_processors(self.robot, True)

        # Action features (EE coordinates)
        self.action_features = get_bimanual_action_features(self.robot, self.teleop_action_processor)

        # Mutex for protecting get_observation() calls
        self._observation_lock = threading.Lock()

        # Debug thread for continuous observation monitoring
        self._debug_thread: Optional[threading.Thread] = None
        self._debug_running = False
        self._debug_frequency = 10.0  # Hz

        # Start debug thread automatically
        self.start_debug_thread(frequency=10.0)

    def execute_ee_action(self, ee_pose_dict: dict[str, float]):
        """Execute an end effector action using inverse kinematics.

        Args:
            ee_pose_dict: Dictionary with EE pose values (e.g., {"left_ee.x": 0.1, ...})
        """
        with self._observation_lock:
            observation = self.robot.get_observation()

        # Convert EE pose dict to tensor for logging
        ee_pose_tensor = action_dict_to_tensor(ee_pose_dict, self.action_features)
        log_rerun_action_chunk(ee_pose_tensor.unsqueeze(0))

        # Convert EE pose to joint angles using IK
        joint_action = self.robot_action_processor((ee_pose_dict, observation))

        # Send to robot
        with self._observation_lock:
            performed_action = self.robot.send_action(joint_action)

        # Log observation and performed action
        log_rerun_data(observation, performed_action)

        return performed_action

    def _get_arm_indices(self, arm: Literal["left", "right"]):
        """Get tensor indices for specified arm."""
        if arm == "left":
            return 0, 1, 2, 3, 4, 5  # x, y, z, wx, wy, wz
        else:
            return 7, 8, 9, 10, 11, 12

    def _apply_rotation_delta(
        self,
        ee_tensor: torch.Tensor,
        wx_idx: int,
        wy_idx: int,
        wz_idx: int,
        tilt_down_deg: float,
        rotate_gripper_ccw_deg: float,
        rotate_base_left_deg: float,
    ):
        """Apply rotation deltas to EE pose tensor."""
        if tilt_down_deg == 0 and rotate_gripper_ccw_deg == 0 and rotate_base_left_deg == 0:
            return

        current_rotvec = ee_tensor[[wx_idx, wy_idx, wz_idx]].numpy()
        current_rot = Rotation.from_rotvec(current_rotvec)

        # Apply rotations (in degrees, convert to radians)
        tilt_rad = np.deg2rad(tilt_down_deg)
        gripper_rot_rad = np.deg2rad(rotate_gripper_ccw_deg)
        base_rot_rad = np.deg2rad(rotate_base_left_deg)

        # Tilt is rotation around Y axis, Gripper rotation around X, Base rotation around Z
        delta_rot = Rotation.from_euler("yxz", [tilt_rad, gripper_rot_rad, base_rot_rad])

        # Compose rotations
        new_rot = current_rot * delta_rot
        new_rotvec = new_rot.as_rotvec()

        ee_tensor[wx_idx] = new_rotvec[0]
        ee_tensor[wy_idx] = new_rotvec[1]
        ee_tensor[wz_idx] = new_rotvec[2]

    def move_in_world_frame(
        self,
        arm: Literal["left", "right"] = "left",
        move_up_mm: float = 0.0,
        move_forward_mm: float = 0.0,
        move_right_mm: float = 0.0,
        tilt_down_deg: float = 0.0,
        rotate_gripper_ccw_deg: float = 0.0,
        rotate_base_left_deg: float = 0.0,
    ):
        """Move end-effector in the world/base_link coordinate frame.

        Movements are relative to the world axes, NOT the gripper's orientation:
        - move_forward_mm: Movement toward camera (maps to world -X axis)
        - move_right_mm: Movement to robot's right (maps to world +Y axis)
        - move_up_mm: Movement upward (maps to world +Z axis)

        Args:
            arm: Which arm to move ("left" or "right")
            move_up_mm: Distance to move along world Z axis (+ up, - down)
            move_forward_mm: Distance to move along world X axis (+ forward, - backward)
            move_right_mm: Distance to move along world Y axis (+ right, - left)
            tilt_down_deg: Tilt gripper down in degrees
            rotate_gripper_ccw_deg: Rotate gripper counter-clockwise in degrees
            rotate_base_left_deg: Rotate arm base left in degrees
        """
        # Get current EE pose (protected by mutex)
        with self._observation_lock:
            observation = self.robot.get_observation()
        current_ee_tensor = compute_current_ee(observation, self.teleop_action_processor, self.action_features)

        # Get indices for this arm
        x_idx, y_idx, z_idx, wx_idx, wy_idx, wz_idx = self._get_arm_indices(arm)

        # Transform intuitive frame movements to world frame
        world_x_mm, world_y_mm, world_z_mm = intuitive_to_world_translation(
            move_forward_mm, move_right_mm, move_up_mm
        )

        # Apply world-frame translations
        new_ee_tensor = current_ee_tensor.clone()
        new_ee_tensor[x_idx] += world_x_mm / 1000.0
        new_ee_tensor[y_idx] += world_y_mm / 1000.0
        new_ee_tensor[z_idx] += world_z_mm / 1000.0

        # Apply rotations
        self._apply_rotation_delta(
            new_ee_tensor, wx_idx, wy_idx, wz_idx, tilt_down_deg, rotate_gripper_ccw_deg, rotate_base_left_deg
        )

        # Execute action
        ee_pose_dict = action_tensor_to_dict(new_ee_tensor, self.action_features)
        self.execute_ee_action(ee_pose_dict)
        return ee_pose_dict

    def move_relative_to_gripper(
        self,
        arm: Literal["left", "right"] = "left",
        forward_mm: float = 0.0,
        right_mm: float = 0.0,
        up_mm: float = 0.0,
        tilt_down_deg: float = 0.0,
        rotate_gripper_ccw_deg: float = 0.0,
        rotate_base_left_deg: float = 0.0,
    ):
        """Move end-effector relative to the gripper's current orientation.

        This provides intuitive gripper-relative movements where directions are
        relative to how the gripper is currently oriented, not the world frame.

        Args:
            arm: Which arm to move ("left" or "right")
            forward_mm: Move in gripper's approach direction (+ forward, - backward)
            right_mm: Move to gripper's right (+ right, - left)
            up_mm: Move perpendicular to gripper palm (+ up, - down)
            tilt_down_deg: Tilt gripper down (+ down, - up) in degrees
            rotate_gripper_ccw_deg: Rotate gripper counter-clockwise in degrees
            rotate_base_left_deg: Rotate arm base left in degrees

        The Koch gripper frame is handled internally:
        - Actual Koch X = back (so forward maps to -X)
        - Actual Koch Y = down (so up maps to -Y)
        - Actual Koch Z = right (so right maps to +Z)
        """
        # Get current EE pose (protected by mutex)
        with self._observation_lock:
            observation = self.robot.get_observation()
        current_ee_tensor = compute_current_ee(observation, self.teleop_action_processor, self.action_features)

        # Get indices for this arm
        x_idx, y_idx, z_idx, wx_idx, wy_idx, wz_idx = self._get_arm_indices(arm)

        # Transform intuitive frame movements to gripper frame, then rotate to world frame
        gripper_frame_movement_mm = intuitive_to_gripper_translation(forward_mm, right_mm, up_mm)
        gripper_frame_movement_m = gripper_frame_movement_mm / 1000.0

        current_rotvec = current_ee_tensor[[wx_idx, wy_idx, wz_idx]].numpy()
        current_rot = Rotation.from_rotvec(current_rotvec)
        world_frame_movement = current_rot.apply(gripper_frame_movement_m)

        # Apply gripper-relative translations in world frame
        new_ee_tensor = current_ee_tensor.clone()
        new_ee_tensor[x_idx] += world_frame_movement[0]
        new_ee_tensor[y_idx] += world_frame_movement[1]
        new_ee_tensor[z_idx] += world_frame_movement[2]

        # Apply rotations
        self._apply_rotation_delta(
            new_ee_tensor, wx_idx, wy_idx, wz_idx, tilt_down_deg, rotate_gripper_ccw_deg, rotate_base_left_deg
        )

        # Execute action
        ee_pose_dict = action_tensor_to_dict(new_ee_tensor, self.action_features)
        self.execute_ee_action(ee_pose_dict)
        return ee_pose_dict

    def set_gripper(self, arm: Literal["left", "right"], openness_pct: float):
        """Set gripper openness.

        Args:
            arm: "left" or "right"
            openness_pct: 0-100 (0 = closed, 100 = open)
        """
        # Get current EE pose (protected by mutex)
        with self._observation_lock:
            observation = self.robot.get_observation()
        current_ee_tensor = compute_current_ee(observation, self.teleop_action_processor, self.action_features)

        # Update gripper value
        new_ee_tensor = current_ee_tensor.clone()
        new_ee_tensor[6 if arm == "left" else 13] = openness_pct

        # Convert to dictionary and execute
        ee_pose_dict = action_tensor_to_dict(new_ee_tensor, self.action_features)
        self.execute_ee_action(ee_pose_dict)

        return ee_pose_dict

    def reset_to_initial_position(self, num_steps: int = 30):
        """Reset the robot to the initial/home position using smooth trajectory.

        Args:
            num_steps: Number of interpolation steps for smooth movement (default: 30)

        Returns:
            The initial EE pose dictionary that was executed
        """
        logger.info(f"Resetting robot to initial position with {num_steps} steps...")

        # Get current position (protected by mutex)
        with self._observation_lock:
            observation = self.robot.get_observation()
        current_ee = compute_current_ee(observation, self.teleop_action_processor, self.action_features)

        # Generate smooth trajectory from current to initial position
        trajectory = generate_linear_trajectory(
            start=current_ee,
            target=INITIAL_EE_POSE,
            num_steps=num_steps
        )

        # Execute trajectory with proper control frequency
        for step_idx, ee_pose in enumerate(trajectory):
            step_start_time = time.perf_counter()

            ee_pose_dict = action_tensor_to_dict(ee_pose, self.action_features)
            self.execute_ee_action(ee_pose_dict)

            if step_idx < num_steps - 1:  # Don't sleep after last step
                # Dynamically adjust sleep time to maintain desired control frequency
                elapsed_time = time.perf_counter() - step_start_time
                time.sleep(max(0, self.environment_dt - elapsed_time))

        logger.info("Robot reset to initial position complete")
        return action_tensor_to_dict(INITIAL_EE_POSE, self.action_features)

    def get_state_dict(self):
        """Get human-readable robot state in intuitive coordinate frame."""
        with self._observation_lock:
            observation = self.robot.get_observation()
        ee_pose = compute_current_ee(observation, self.teleop_action_processor, self.action_features)

        # Transform world frame positions to intuitive frame
        left_position = world_to_intuitive_position(
            ee_pose[0].item(), ee_pose[1].item(), ee_pose[2].item()
        )
        right_position = world_to_intuitive_position(
            ee_pose[7].item(), ee_pose[8].item(), ee_pose[9].item()
        )

        state = {
            "left_arm": {
                "position": left_position,
                "orientation": {
                    "wx": f"{ee_pose[3].item():.4f}rad",
                    "wy": f"{ee_pose[4].item():.4f}rad",
                    "wz": f"{ee_pose[5].item():.4f}rad",
                },
                "gripper": f"{ee_pose[6].item():.1f}%",
            },
            "right_arm": {
                "position": right_position,
                "orientation": {
                    "wx": f"{ee_pose[10].item():.4f}rad",
                    "wy": f"{ee_pose[11].item():.4f}rad",
                    "wz": f"{ee_pose[12].item():.4f}rad",
                },
                "gripper": f"{ee_pose[13].item():.1f}%",
            },
        }

        return state

    def get_camera_images(self):
        """Get images from all cameras."""
        with self._observation_lock:
            observation = self.robot.get_observation()
        images = {}

        for key, value in observation.items():
            # Camera images are numpy arrays
            if isinstance(value, np.ndarray) and value.ndim == 3:
                images[key] = value

        return images

    def disconnect(self):
        """Disconnect from robot."""
        # Stop debug thread if running
        self.stop_debug_thread()

        if self.robot is not None:
            self.robot.disconnect()
            logger.info("Robot disconnected")

    def _debug_loop(self):
        """Internal debug loop that continuously reads observation and computes EE pose."""
        logger.info(f"Debug thread started at {self._debug_frequency} Hz")
        sleep_time = 1.0 / self._debug_frequency

        while self._debug_running:
            try:
                start_time = time.time()

                # Get observation and compute current EE (protected by mutex)
                with self._observation_lock:
                    observation = self.robot.get_observation()
                current_ee_tensor = compute_current_ee(
                    observation, self.teleop_action_processor, self.action_features
                )
                log_rerun_action_chunk(current_ee_tensor.unsqueeze(0), name="current_ee_")

                # Log the data for debugging
                logger.debug(f"[DEBUG] EE State: {current_ee_tensor.tolist()}")

                # Sleep to maintain frequency
                elapsed = time.time() - start_time
                if elapsed < sleep_time:
                    time.sleep(sleep_time - elapsed)
                else:
                    logger.warning(f"Debug thread running slower than {self._debug_frequency} Hz (took {elapsed:.3f}s)")

            except Exception as e:
                logger.error(f"Error in debug loop: {e}")
                logger.error(traceback.format_exc())

        logger.info("Debug thread stopped")

    def start_debug_thread(self, frequency: float = 10.0):
        """Start the debug thread for continuous observation monitoring.

        Args:
            frequency: Update frequency in Hz (default: 10.0)
        """
        if self._debug_running:
            logger.warning("Debug thread is already running")
            return

        self._debug_frequency = frequency
        self._debug_running = True
        self._debug_thread = threading.Thread(target=self._debug_loop, daemon=True)
        self._debug_thread.start()
        logger.info(f"Started debug thread at {frequency} Hz")

    def stop_debug_thread(self):
        """Stop the debug thread."""
        if not self._debug_running:
            return

        self._debug_running = False
        if self._debug_thread is not None:
            self._debug_thread.join(timeout=2.0)
            self._debug_thread = None
        logger.info("Stopped debug thread")


# -----------------------------------------------------------------------------
# Global robot instance
# -----------------------------------------------------------------------------

_robot_controller: Optional[LeRobotController] = None
_server_config: Optional[MCPRobotServerConfig] = None


def get_robot() -> LeRobotController:
    """Lazy-initialize the global robot controller instance."""
    global _robot_controller
    if _robot_controller is None:
        if _server_config is None:
            raise RuntimeError("Server config not initialized")

        try:
            _robot_controller = LeRobotController(_server_config)
            logger.info("LeRobotController initialized.")
        except Exception as e:
            logger.error(f"MCP: FATAL - Error initializing robot: {e}", exc_info=True)
            raise SystemExit(f"MCP Server cannot start: LeRobotController failed to initialize ({e})") from e

    return _robot_controller

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _np_to_mcp_image(arr_rgb: np.ndarray) -> Image:
    """Convert a numpy RGB image to MCP image format."""
    pil_img = PILImage.fromarray(arr_rgb.astype(np.uint8))
    with io.BytesIO() as buf:
        pil_img.save(buf, format="JPEG")
        raw_data = buf.getvalue()
    return Image(data=raw_data, format="jpeg")


def get_state_with_images(result_json: dict, is_movement: bool = False) -> List[Union[Image, dict]]:
    """Combine robot state with camera images into a unified response format."""
    robot = get_robot()
    try:
        if is_movement:
            time.sleep(0.1)  # Small delay for images to update

        raw_imgs = robot.get_camera_images()

        if not raw_imgs:
            logger.warning("MCP: No camera images returned from robot controller.")
            return [result_json, "Warning: No camera images available."]

        mcp_images = [_np_to_mcp_image(img) for img in raw_imgs.values()]

        # Return combined response
        return [result_json] + mcp_images
    except Exception as e:
        logger.error(f"Error getting camera images: {str(e)}")
        logger.error(traceback.format_exc())
        return [result_json, "Error getting camera images"]


# -----------------------------------------------------------------------------
# MCP Tools - Read-only
# -----------------------------------------------------------------------------


@mcp.tool(
    description="Get a description of the robot and instructions. Run this before using any other tool."
)
def get_initial_instructions() -> str:
    return _server_config.robot_description if _server_config else "Bimanual robot arm with IK control"


@mcp.tool(description="Get current robot state with images from all cameras.")
def get_robot_state():
    robot = get_robot()
    try:
        state_dict = robot.get_state_dict()
        result_json = {"status": "success", "robot_state": state_dict}
        logger.info("MCP: get_robot_state successful")
        return get_state_with_images(result_json, is_movement=False)
    except Exception as e:
        logger.error(f"Error getting robot state: {str(e)}")
        return {"status": "error", "message": str(e)}


# -----------------------------------------------------------------------------
# MCP Tools - Actuation
# -----------------------------------------------------------------------------


@mcp.tool(
    description="""
    Move the robot arm in the world coordinate frame.

    Movements are relative to the fixed world axes, NOT the gripper's current orientation.
    Use this when you want to move along absolute directions (e.g., "move straight up in world space").
    For gripper-relative movements, use move_robot_gripper_relative instead.

    Args:
        arm (str): Which arm to move - "left" or "right"
        move_gripper_up_mm (float, optional): Distance along world Z axis (+ up, - down) in mm
        move_gripper_forward_mm (float, optional): Distance along world X axis (+ forward, - backward) in mm
        move_gripper_right_mm (float, optional): Distance along world Y axis (+ right, - left) in mm
        tilt_gripper_down_angle (float, optional): Angle to tilt gripper down (positive) or up (negative) in degrees
        rotate_gripper_counterclockwise_angle (float, optional): Angle to rotate gripper counterclockwise (positive) or clockwise (negative) in degrees
        rotate_robot_left_angle (float, optional): Angle to rotate entire arm base left (positive) or right (negative) in degrees

    Expected input format:
    {
        "arm": "left",
        "move_gripper_up_mm": "10",
        "move_gripper_forward_mm": "-5",
        "tilt_gripper_down_angle": "10",
        "rotate_gripper_counterclockwise_angle": "-15",
        "rotate_robot_left_angle": "15"
    }

    Returns:
        list: JSON with status and robot state, plus camera images
    """
)
def move_robot_world_frame(
    arm: Literal["left", "right"] = "left",
    move_gripper_up_mm=None,
    move_gripper_forward_mm=None,
    move_gripper_right_mm=None,
    tilt_gripper_down_angle=None,
    rotate_gripper_counterclockwise_angle=None,
    rotate_robot_left_angle=None,
):
    robot = get_robot()
    logger.info(
        f"MCP Tool: move_robot_world_frame received for {arm} arm: up={move_gripper_up_mm}, fwd={move_gripper_forward_mm}, "
        f"right={move_gripper_right_mm}, tilt={tilt_gripper_down_angle}, "
        f"grip_rot={rotate_gripper_counterclockwise_angle}, robot_rot={rotate_robot_left_angle}"
    )

    try:
        # Convert parameters
        move_params = {
            "arm": arm,
            "move_up_mm": float(move_gripper_up_mm) if move_gripper_up_mm is not None else 0.0,
            "move_forward_mm": float(move_gripper_forward_mm) if move_gripper_forward_mm is not None else 0.0,
            "move_right_mm": float(move_gripper_right_mm) if move_gripper_right_mm is not None else 0.0,
            "tilt_down_deg": float(tilt_gripper_down_angle) if tilt_gripper_down_angle is not None else 0.0,
            "rotate_gripper_ccw_deg": float(rotate_gripper_counterclockwise_angle)
            if rotate_gripper_counterclockwise_angle is not None
            else 0.0,
            "rotate_base_left_deg": float(rotate_robot_left_angle)
            if rotate_robot_left_angle is not None
            else 0.0,
        }

        # Check if any movement requested
        if all(abs(v) < 1e-6 for k, v in move_params.items() if k != "arm"):
            result_json = {
                "status": "success",
                "message": "No movement parameters provided",
                "robot_state": robot.get_state_dict(),
            }
            return get_state_with_images(result_json, is_movement=False)

        # Execute movement
        robot.move_in_world_frame(**move_params)

        result_json = {
            "status": "success",
            "message": f"Successfully moved {arm} arm in world frame",
            "robot_state": robot.get_state_dict(),
        }

        logger.info(f"MCP: move_robot_world_frame successful for {arm} arm")
        return get_state_with_images(result_json, is_movement=True)

    except Exception as e:
        logger.error(f"Error in move_robot_world_frame: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}


@mcp.tool(
    description="""
    Move the robot arm relative to the gripper's current orientation.

    This is useful when you want the gripper to move in the direction it's facing,
    rather than along the world frame axes. For example, "forward" moves in the
    direction the gripper is pointing.

    Args:
        arm (str): Which arm to move - "left" or "right"
        move_forward_mm (float, optional): Move in gripper's pointing direction (+ forward, - backward) in mm
        move_right_mm (float, optional): Move to gripper's right (+ right, - left) in mm
        move_up_mm (float, optional): Move perpendicular to gripper (+ up, - down) in mm
        tilt_gripper_down_angle (float, optional): Angle to tilt gripper down (positive) or up (negative) in degrees
        rotate_gripper_counterclockwise_angle (float, optional): Angle to rotate gripper counterclockwise (positive) or clockwise (negative) in degrees
        rotate_robot_left_angle (float, optional): Angle to rotate entire arm base left (positive) or right (negative) in degrees

    Expected input format:
    {
        "arm": "left",
        "move_forward_mm": "20",
        "move_right_mm": "10",
        "move_up_mm": "-5",
        "tilt_gripper_down_angle": "10",
        "rotate_gripper_counterclockwise_angle": "-15",
        "rotate_robot_left_angle": "15"
    }

    Returns:
        list: JSON with status and robot state, plus camera images
    """
)
def move_robot_gripper_relative(
    arm: Literal["left", "right"] = "left",
    move_forward_mm=None,
    move_right_mm=None,
    move_up_mm=None,
    tilt_gripper_down_angle=None,
    rotate_gripper_counterclockwise_angle=None,
    rotate_robot_left_angle=None,
):
    robot = get_robot()
    logger.info(
        f"MCP Tool: move_robot_gripper_relative received for {arm} arm: "
        f"fwd={move_forward_mm}, right={move_right_mm}, up={move_up_mm}, "
        f"tilt={tilt_gripper_down_angle}, grip_rot={rotate_gripper_counterclockwise_angle}, "
        f"robot_rot={rotate_robot_left_angle}"
    )

    try:
        # Convert parameters
        move_params = {
            "arm": arm,
            "forward_mm": float(move_forward_mm) if move_forward_mm is not None else 0.0,
            "right_mm": float(move_right_mm) if move_right_mm is not None else 0.0,
            "up_mm": float(move_up_mm) if move_up_mm is not None else 0.0,
            "tilt_down_deg": float(tilt_gripper_down_angle) if tilt_gripper_down_angle is not None else 0.0,
            "rotate_gripper_ccw_deg": float(rotate_gripper_counterclockwise_angle)
            if rotate_gripper_counterclockwise_angle is not None
            else 0.0,
            "rotate_base_left_deg": float(rotate_robot_left_angle)
            if rotate_robot_left_angle is not None
            else 0.0,
        }

        # Check if any movement requested
        if all(abs(v) < 1e-6 for k, v in move_params.items() if k != "arm"):
            result_json = {
                "status": "success",
                "message": "No movement parameters provided",
                "robot_state": robot.get_state_dict(),
            }
            return get_state_with_images(result_json, is_movement=False)

        # Execute gripper-relative movement
        robot.move_relative_to_gripper(**move_params)

        result_json = {
            "status": "success",
            "message": f"Successfully moved {arm} arm relative to gripper orientation",
            "robot_state": robot.get_state_dict(),
        }

        logger.info(f"MCP: move_robot_gripper_relative successful for {arm} arm")
        return get_state_with_images(result_json, is_movement=True)

    except Exception as e:
        logger.error(f"Error in move_robot_gripper_relative: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}


@mcp.tool(
    description="""
    Control the robot's gripper openness from 0% (completely closed) to 100% (completely open).
    Args:
        arm (str): Which arm's gripper - "left" or "right"
        gripper_openness_pct (float): Gripper openness 0-100

    Expected input format:
    {
        "arm": "left",
        "gripper_openness_pct": "50"
    }
    """
)
def control_gripper(arm: Literal["left", "right"] = "left", gripper_openness_pct=None):
    robot = get_robot()

    try:
        if gripper_openness_pct is None:
            return {"status": "error", "message": "gripper_openness_pct is required"}

        openness = float(gripper_openness_pct)
        logger.info(f"MCP Tool: control_gripper called for {arm} arm with openness={openness}%")

        robot.set_gripper(arm, openness)

        result_json = {
            "status": "success",
            "message": f"Successfully set {arm} gripper to {openness}%",
            "robot_state": robot.get_state_dict(),
        }

        logger.info(f"MCP: control_gripper successful for {arm} arm")
        return get_state_with_images(result_json, is_movement=True)

    except (ValueError, TypeError) as e:
        logger.error(f"MCP: control_gripper received invalid input: {gripper_openness_pct}, error: {str(e)}")
        return {"status": "error", "message": f"Invalid gripper openness value: {str(e)}"}


@mcp.tool(
    description="""
    Reset the robot to its initial position that was captured when the server started.
    This is useful for returning to a known safe state or starting position.

    No arguments required.

    Returns:
        list: JSON with status and robot state, plus camera images
    """
)
def reset_robot():
    robot = get_robot()
    logger.info("MCP Tool: reset_robot called")

    try:
        robot.reset_to_initial_position()

        result_json = {
            "status": "success",
            "message": "Successfully reset robot to initial position",
            "robot_state": robot.get_state_dict(),
        }

        logger.info("MCP: reset_robot successful")
        return get_state_with_images(result_json, is_movement=True)

    except Exception as e:
        logger.error(f"Error in reset_robot: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}


# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------


def _cleanup():
    """Disconnect from hardware on server shutdown."""
    global _robot_controller
    if _robot_controller is not None:
        try:
            _robot_controller.disconnect()
        except Exception as e_disc:
            logger.error(f"MCP: Exception during disconnect: {e_disc}", exc_info=True)


atexit.register(_cleanup)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: MCPRobotServerConfig):
    """Main entry point for MCP robot server."""
    global _server_config
    _server_config = cfg
    # Initialize robot controller
    _ = get_robot()

    logger.info("Starting MCP Robot Server...")
    logger.info(f"Note: MCP servers communicate via {cfg.transport}")
    logger.info("They are typically launched by MCP clients like Claude Desktop")
    try:
        mcp.run(transport=cfg.transport)
    except SystemExit as e:
        logger.error(f"MCP Server failed to start: {e}")
    except Exception as e_main:
        logger.error(f"MCP Server CRITICAL RUNTIME ERROR: {e_main}", exc_info=True)


if __name__ == "__main__":
    main()
