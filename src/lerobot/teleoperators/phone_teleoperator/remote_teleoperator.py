#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import scipy

# Check scipy version for scalar_first compatibility
_scipy_version = tuple(map(int, scipy.__version__.split('.')[:2]))
_supports_scalar_first = _scipy_version >= (1, 7)


def _rotation_from_quat(quat: np.ndarray, scalar_first: bool = True) -> R:
    """
    Create a Rotation object from quaternion with backward compatibility.
    
    Args:
        quat: Quaternion array
        scalar_first: Whether the first element is the scalar component (w,x,y,z vs x,y,z,w)
    
    Returns:
        Rotation object
    """
    if _supports_scalar_first:
        return R.from_quat(quat, scalar_first=scalar_first)
    else:
        # For older scipy versions, convert quaternion format if needed
        if scalar_first:
            # Convert from (w,x,y,z) to (x,y,z,w) for older scipy
            quat_converted = np.array([quat[1], quat[2], quat[3], quat[0]])
        else:
            quat_converted = quat
        return R.from_quat(quat_converted)


def _quat_as_scalar_first(rotation: R) -> np.ndarray:
    """
    Get quaternion in scalar-first format (w,x,y,z) with backward compatibility.
    
    Args:
        rotation: Rotation object
        
    Returns:
        Quaternion as (w,x,y,z)
    """
    if _supports_scalar_first:
        return rotation.as_quat(scalar_first=True)
    else:
        # For older scipy versions, convert from (x,y,z,w) to (w,x,y,z)
        quat = rotation.as_quat()  # Returns (x,y,z,w)
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w,x,y,z)


try:
    import pyroki as pk
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError as e:
    raise ImportError(
        "Phone teleoperator requires additional dependencies. "
        "Please install with: pip install pyroki viser yourdfpy"
    ) from e

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_remote_teleoperator import PhoneTeleoperatorConfig

logger = logging.getLogger(__name__)


class PhoneTeleoperator(Teleoperator):
    """
    Phone-based teleoperator that receives pose data from mobile phone via gRPC
    and converts it to robot control commands using inverse kinematics.
    
    This teleoperator integrates with the VirtualManipulator system from the daxie package
    to provide phone-based robot control.
    """

    config_class = PhoneTeleoperatorConfig
    name = "phone_teleoperator"

    def __init__(self, config: PhoneTeleoperatorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        
        # Initialize robot model for IK
        self.urdf = None
        self.robot = None
        self.urdf_vis = None
        self.server = None
        
        # Phone connection state
        self._is_connected = False
        self._phone_connected = False
        self.start_teleop = False
        self.prev_is_resetting = False
        
        # Reset position holding - store the position when reset starts
        self.reset_hold_position = None
        
        # Pose tracking
        self.current_t_R = np.array(self.config.initial_position)
        self.current_q_R = np.array(self.config.initial_wxyz)
        self.initial_phone_quat = None
        self.initial_phone_pos = None
        self.last_precision_mode = False
        
        # Mapping parameters
        self.quat_RP = None
        self.translation_RP = None
        
        # gRPC server and pose service (to be initialized in connect())
        self.grpc_server = None
        self.pose_service = None
        
        # Timer for reading motor positions after 5 seconds
        self.teleop_start_time = None
        self.motor_positions_read = False
        
        # Flag to show initial motor positions on first get_action call
        self.initial_positions_shown = False
        
        # Temporal smoothing state
        self._prev_q = None
        # Elbow soft stop threshold (radians), computed from URDF limits when available
        self._elbow_soft_stop = None
        # Elbow backward block baseline (radians) captured at start of teleop
        self._elbow_back_limit = None
        # Direction to block as "backwards": 'increase' or 'decrease'
        self._elbow_block_direction = getattr(self.config, "elbow_block_direction", "increase")

        # Tuning parameters (can be edited in-script)
        # Joint indices: 0 shoulder_pan, 1 shoulder_lift, 2 elbow_flex, 3 wrist_flex, 4 wrist_roll, 5 gripper
        self.tune = {
            "lowpass_alpha": 0.25,  # 0..1; lower = smoother
            "delta_scale": {  # per-joint delta multipliers (post-IK, relative to previous)
                0: 0.5,  # shoulder_pan
                1: 1.0,
                2: 1.0,
                3: 1.0,
                4: 1.0,
                5: 1.0,
            },
            "wrist_roll_overhand_bias": {
                "enabled": True,
                "index": 4,
                "target": 0.0,   # radians
                "blend": 0.05,   # 0..1 small bias
            },
            "elbow_soft_stop": {
                "enabled": True,
                "index": 2,
                "fraction_from_lower": 0.25,  # 0..1; 0.25 ~ 6:00 soft stop
                "below_small_step_deg": 3.0,
                "below_margin_deg": 8.0,
                "above_max_down_deg": 25.0,
            },
            "elbow_back_block": {
                "enabled": True,
                "index": 2,
                "direction": getattr(self.config, "elbow_block_direction", "increase"),
                "tolerance_deg": 2.0,
            },
            "fixed_rate": {
                "enabled": True,
                "joints": {
                    # shoulder_pan moves at a fixed max step per update
                    0: {"step_deg": 2.0, "deadband_deg": 0.2},
                },
            },
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Features for the actions produced by this teleoperator."""
        # Assuming 6 DOF arm + gripper (adjust based on your robot)
        motor_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos", 
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ]
        return {name: float for name in motor_names}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Features for the feedback actions sent to this teleoperator."""
        # Phone teleoperator doesn't typically need feedback
        return {}

    @property
    def is_connected(self) -> bool:
        """Whether the phone teleoperator is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Phone teleoperator doesn't require calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Establish connection with phone via gRPC and initialize robot model."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # Initialize robot model
            urdf_path = self.config.urdf_path
            mesh_path = self.config.mesh_path
            
            # Resolve relative paths to absolute paths
            if urdf_path and mesh_path:
                from pathlib import Path
                import lerobot
                
                # Get the lerobot package root directory
                lerobot_root = Path(lerobot.__file__).parent.parent
                
                # Resolve relative paths
                if not Path(urdf_path).is_absolute():
                    urdf_path = str(lerobot_root / urdf_path)
                if not Path(mesh_path).is_absolute():
                    mesh_path = str(lerobot_root / mesh_path)
                
                logger.info(f"Using SO100 paths - URDF: {urdf_path}, Mesh: {mesh_path}")
            else:
                # Fallback auto-detection if paths are empty
                try:
                    from pathlib import Path
                    import lerobot
                    
                    # Get the lerobot package root directory
                    lerobot_root = Path(lerobot.__file__).parent.parent
                    so100_model_path = lerobot_root / "lerobot" / "common" / "robots" / "so100_follower" / "model"
                    
                    if so100_model_path.exists():
                        auto_urdf_path = str(so100_model_path / "so100.urdf")
                        auto_mesh_path = str(so100_model_path / "meshes")
                        urdf_path = urdf_path or auto_urdf_path
                        mesh_path = mesh_path or auto_mesh_path
                        logger.info(f"Auto-detected SO100 paths - URDF: {urdf_path}, Mesh: {mesh_path}")
                    else:
                        raise FileNotFoundError(f"Could not find SO100 model directory at {so100_model_path}")
                except Exception as e:
                    logger.warning(f"Could not auto-detect SO100 paths: {e}")
                    if not urdf_path or not mesh_path:
                        raise ValueError("URDF path and mesh path must be provided in config or SO100 model must be available in lerobot/common/robots/so100_follower/model/")
                
            self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_path)
            self.robot = pk.Robot.from_urdf(self.urdf)
            # Compute elbow soft stop from URDF limits if available
            frac = float(self.tune["elbow_soft_stop"]["fraction_from_lower"]) if self.tune["elbow_soft_stop"]["enabled"] else 0.25
            self._compute_elbow_soft_stop(frac)
            
            # Initialize visualization if enabled
            if self.config.enable_visualization:
                self._init_visualization()
            
            # Start gRPC server for phone communication
            self._start_grpc_server()
            
            self._is_connected = True
            
            logger.info(f"{self} connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect {self}: {e}")
            raise

    def _compute_elbow_soft_stop(self, fraction: float = 0.25) -> None:
        """Compute elbow soft-stop threshold from URDF joint limits.

        By default uses 25% from the lower limit to allow roughly twice the range
        compared to the prior halfway clamp (closer to 6:00 vs 9:00).
        """
        try:
            joint_name = "elbow_flex"
            lower = None
            upper = None
            jm = getattr(self.urdf, "joint_map", None)
            if jm and joint_name in jm:
                limit = getattr(jm[joint_name], "limit", None)
                lower = getattr(limit, "lower", None)
                upper = getattr(limit, "upper", None)
            if (lower is None or upper is None) and hasattr(self.urdf, "joints"):
                for j in getattr(self.urdf, "joints", []):
                    if getattr(j, "name", None) == joint_name:
                        limit = getattr(j, "limit", None)
                        lower = getattr(limit, "lower", None)
                        upper = getattr(limit, "upper", None)
                        break
            if lower is not None and upper is not None:
                low = float(lower)
                up = float(upper)
                # Ensure ordering
                if up < low:
                    low, up = up, low
                self._elbow_soft_stop = float(low + fraction * (up - low))
            else:
                self._elbow_soft_stop = None
        except Exception:
            self._elbow_soft_stop = None

    def calibrate(self) -> None:
        """Phone teleoperator doesn't require calibration."""
        pass

    def configure(self) -> None:
        """Configure the phone teleoperator (no-op for phone teleoperator)."""
        pass

    def _init_visualization(self) -> None:
        """Initialize the Viser visualization server and URDF model."""
        self.server = viser.ViserServer(port=self.config.viser_port)
        self.server.scene.add_grid("/ground", width=2.0, height=2.0)
        self.urdf_vis = ViserUrdf(self.server, self.urdf, root_node_name="/base")

    def _start_grpc_server(self) -> None:
        """Start the gRPC server for phone pose streaming."""
        try:
            # Import from local transport module
            from lerobot.transport.phone_teleop_grpc.pos_grpc_server import start_grpc_server
            
            self.grpc_server, self.pose_service = start_grpc_server(port=self.config.grpc_port)
            self.hz_grpc = 0.0
            self.pose_service.get_latest_pose(block=False)
            logger.info("gRPC server started for phone communication")
        except ImportError as e:
            logger.error(f"Could not import gRPC server: {e}")
            raise ImportError(f"Failed to import phone teleop gRPC server: {e}")

    def _open_phone_connection(self, curr_qpos_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wait for phone to connect and set initial mapping."""
        # Use the initial target pose, not the current robot joint positions
        # The current joint positions are used elsewhere, but the target pose is what we map to
        init_rot_robot = _rotation_from_quat(self.current_q_R, scalar_first=True)
        self.current_t_R = np.array(self.config.initial_position)
        self.current_q_R = np.array(self.config.initial_wxyz)

        logger.info("Getting initial phone data for mapping setup...")
        logger.info(f"gRPC server listening on port {self.config.grpc_port}")
        
        # Get phone data once to set up mapping - don't wait for start signal
        data = self.pose_service.get_latest_pose(block=True, timeout=self.config.grpc_timeout)
        if data is not None:
            self.start_teleop = data["switch"]
        else:
            # Use default data if no phone connected yet
            data = {
                "position": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],  # w,x,y,z
                "gripper_value": 0.0,
                "switch": False
            }
            self.start_teleop = False

        pos, quat, gripper_value = data["position"], data["rotation"], data["gripper_value"]
        
        initial_rot_phone = _rotation_from_quat(quat, scalar_first=True)
        initial_pos_phone = np.array(pos)

        self.initial_phone_quat = quat.copy()
        self.initial_phone_pos = initial_pos_phone.copy()

        quat_RP = init_rot_robot * initial_rot_phone.inv()
        translation_RP = self.current_t_R - quat_RP.apply(initial_pos_phone)
        
        logger.info("Phone connection established successfully!")
        return quat_RP, translation_RP

    def _reset_mapping(self, phone_pos: np.ndarray, phone_quat: np.ndarray) -> None:
        """Reset mapping parameters when precision mode toggles."""
        self.initial_phone_pos = phone_pos.copy()
        self.initial_phone_quat = phone_quat.copy()

        rot_init = _rotation_from_quat(self.initial_phone_quat, scalar_first=True)
        rot_curr = _rotation_from_quat(self.current_q_R, scalar_first=True)
        self.quat_RP = rot_curr * rot_init.inv()
        self.translation_RP = self.current_t_R - self.quat_RP.apply(self.initial_phone_pos)

    def _map_phone_to_robot(
        self, phone_pos: np.ndarray, phone_quat: np.ndarray, precision_mode: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map phone translation and rotation to robot's coordinate frame."""
        
        phone_pos = np.array(phone_pos, float)
        phone_quat = np.array(phone_quat, float)

        if precision_mode != self.last_precision_mode:
            self._reset_mapping(phone_pos, phone_quat)

        self.last_precision_mode = precision_mode
        scale = (
            self.config.sensitivity_precision if precision_mode 
            else self.config.sensitivity_normal
        ) * float(getattr(self.config, "mapping_gain", 1.0))

        # Translate
        delta = (phone_pos - self.initial_phone_pos) * scale
        scaled_pos = self.initial_phone_pos + delta

        # Rotate
        init_rot = _rotation_from_quat(self.initial_phone_quat, scalar_first=True)
        curr_rot = _rotation_from_quat(phone_quat, scalar_first=True)
        relative_rot = init_rot.inv() * curr_rot
        rotvec = relative_rot.as_rotvec() * (self.config.rotation_sensitivity * float(getattr(self.config, "mapping_gain", 1.0)))
        scaled_rot = R.from_rotvec(rotvec)
        quat_scaled = init_rot * scaled_rot

        # Apply mapping
        quat_robot = self.quat_RP * quat_scaled
        pos_robot = self.quat_RP.apply(scaled_pos) + self.translation_RP

        self.current_q_R = _quat_as_scalar_first(quat_robot)
        self.current_t_R = pos_robot
        
        return pos_robot, self.current_q_R

    def get_action(self, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Get the current action from phone input.
        
        Args:
            observation: Current robot observation containing joint positions
        
        This method processes phone pose data, solves inverse kinematics,
        and returns the target joint positions.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Extract current robot position from observation
        current_joint_pos_deg = None
        if observation is not None:
            try:
                # Extract joint positions from observation (assumes motor names follow pattern)
                motor_keys = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
                             "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
                current_joint_pos_deg = [observation.get(key, 0.0) for key in motor_keys]
            except Exception as e:
                logger.warning(f"Could not extract joint positions from observation: {e}")
        
        # If no observation or extraction failed, use rest pose
        if current_joint_pos_deg is None:
            # Phone teleoperator always works in degrees (robot is auto-configured)
            current_joint_pos_deg = list(np.rad2deg(self.config.rest_pose))
            logger.debug("Using rest pose as current position")

        # Show initial motor positions immediately on first call (before phone connection)
        if not self.initial_positions_shown:
            self._display_motor_positions_formatted(current_joint_pos_deg, "INITIAL ROBOT POSITION")
            self.initial_positions_shown = True

        try:
            # Handle phone connection
            if not self._phone_connected:
                # Pass current position to connection setup (IK solver always expects radians)
                # Phone teleoperator always works in degrees (robot is auto-configured)
                curr_qpos_rad = np.deg2rad(current_joint_pos_deg)
                self.quat_RP, self.translation_RP = self._open_phone_connection(curr_qpos_rad)
                self._phone_connected = True

            if not self.start_teleop:
                # Phone teleoperator always works in degrees (robot is auto-configured)
                current_joint_pos_deg = list(np.rad2deg(self.config.rest_pose))
                self._phone_connected = False
                # Reset timer when teleop stops
                self.teleop_start_time = None
                self.motor_positions_read = False
                # Return current position when not teleoperating
                return self._format_action_dict(current_joint_pos_deg)
            
            # Start timer when teleop becomes active
            if self.teleop_start_time is None:
                self.teleop_start_time = time.time()
            
            # Check if 5 seconds have passed and we haven't read positions yet
            if not self.motor_positions_read and time.time() - self.teleop_start_time >= 5.0:
                self._read_and_display_motor_positions(current_joint_pos_deg)
                self.motor_positions_read = True

            # Get latest pose from gRPC
            data = self.pose_service.get_latest_pose(block=False)

            switch_state = data.get("switch", False)
            reset_mapping_pressed = data.get("reset_mapping", False)
            is_resetting_state = data.get("is_resetting", False)

            # Update reset state tracking - handle both is_resetting and reset_mapping
            current_is_resetting = is_resetting_state or reset_mapping_pressed
            
            # Check for reset transition (prev=False, current=True) - reset just started
            if self.prev_is_resetting == False and current_is_resetting == True:
                self.reset_hold_position = current_joint_pos_deg.copy()
            
            if current_is_resetting:
                self.prev_is_resetting = current_is_resetting
                # Return the captured hold position instead of current drifting position
                if self.reset_hold_position is not None:
                    return self._format_action_dict(self.reset_hold_position)
                else:
                    # Fallback if no hold position captured yet
                    return self._format_action_dict(current_joint_pos_deg)

            # Check for reset transition (prev=True, current=False) - reset just ended
            if self.prev_is_resetting == True and current_is_resetting == False:
                pos, quat = data["position"], data["rotation"]
                self._reset_mapping(pos, quat)
                self.reset_hold_position = None  # Clear the hold position

            self.prev_is_resetting = current_is_resetting

            pos, quat, gripper_value = data["position"], data["rotation"], data["gripper_value"]

            # Map phone pose to robot pose
            t_robot, q_robot = self._map_phone_to_robot(pos, quat, data["precision"])

            # Solve inverse kinematics (returns radians)
            solution_rad = self._solve_ik(t_robot, q_robot)
            
            # Temporal smoothing and posture shaping
            if self.tune.get("bypass_all_mods", False):
                # Bypass: use raw IK output, keep prev for continuity
                self._prev_q = solution_rad
                solution_final = np.rad2deg(solution_rad)
                self.start_teleop = switch_state
                return self._format_action_dict(solution_final)

            try:
                import numpy as _np
            except Exception:
                _np = np
            alpha = float(self.tune["lowpass_alpha"])  # low-pass filter factor
            if self._prev_q is None:
                self._prev_q = solution_rad
            # Low-pass filter all joints
            solution_rad = alpha * solution_rad + (1.0 - alpha) * self._prev_q

            # Discourage elbow going down past soft stop (≈ quarter range from lower)
            ELBOW_IDX = int(self.tune["elbow_soft_stop"]["index"])  # shoulder_pan, shoulder_lift, elbow_flex, ...
            soft_stop = self._elbow_soft_stop
            if soft_stop is not None:
                if solution_rad[ELBOW_IDX] < soft_stop:
                    # Below soft stop: allow only small downward steps and clamp near soft stop
                    small_step = _np.deg2rad(float(self.tune["elbow_soft_stop"]["below_small_step_deg"]))
                    margin = _np.deg2rad(float(self.tune["elbow_soft_stop"]["below_margin_deg"]))
                    allowed = max(solution_rad[ELBOW_IDX], self._prev_q[ELBOW_IDX] - small_step)
                    solution_rad[ELBOW_IDX] = max(allowed, soft_stop - margin)
                else:
                    # Above soft stop: allow more generous downward motion for responsiveness
                    max_down_per_call = _np.deg2rad(float(self.tune["elbow_soft_stop"]["above_max_down_deg"]))
                    solution_rad[ELBOW_IDX] = max(
                        solution_rad[ELBOW_IDX],
                        self._prev_q[ELBOW_IDX] - max_down_per_call,
                    )
            else:
                # Fallback if no limits known
                max_down_per_call = _np.deg2rad(25.0)
                solution_rad[ELBOW_IDX] = max(
                    solution_rad[ELBOW_IDX],
                    self._prev_q[ELBOW_IDX] - max_down_per_call,
                )

            # Gentle overhand bias on wrist roll
            if self.tune["wrist_roll_overhand_bias"]["enabled"]:
                WRIST_ROLL_IDX = int(self.tune["wrist_roll_overhand_bias"]["index"]) 
                overhand_roll_target = float(self.tune["wrist_roll_overhand_bias"]["target"])  # rad
                blend = float(self.tune["wrist_roll_overhand_bias"]["blend"])  # 0..1
                keep = 1.0 - blend
                solution_rad[WRIST_ROLL_IDX] = keep * solution_rad[WRIST_ROLL_IDX] + blend * overhand_roll_target

            # Per-joint delta scaling (sensitivity)
            for j_idx, scale in self.tune["delta_scale"].items():
                j = int(j_idx)
                s = float(scale)
                if 0 <= j < len(solution_rad) and s != 1.0:
                    delta = solution_rad[j] - self._prev_q[j]
                    solution_rad[j] = self._prev_q[j] + s * delta

            # Per-joint fixed-rate limiter (velocity clamp per update)
            if self.tune.get("fixed_rate", {}).get("enabled", False):
                joints_cfg = self.tune["fixed_rate"].get("joints", {})
                for j_idx, cfg in joints_cfg.items():
                    j = int(j_idx)
                    if 0 <= j < len(solution_rad):
                        step = float(cfg.get("step_deg", 2.0))
                        deadband = float(cfg.get("deadband_deg", 0.0))
                        step_rad = _np.deg2rad(step)
                        deadband_rad = _np.deg2rad(deadband)
                        target = float(solution_rad[j])
                        prev = float(self._prev_q[j])
                        error = target - prev
                        if abs(error) <= deadband_rad:
                            solution_rad[j] = prev
                        else:
                            direction = 1.0 if error > 0 else -1.0
                            solution_rad[j] = prev + direction * min(step_rad, abs(error))

            # Strongly prevent elbow from moving "backwards" from initial (12:00) position
            if self.tune["elbow_back_block"]["enabled"]:
                ELBOW_IDX = int(self.tune["elbow_back_block"]["index"]) 
                if self._elbow_back_limit is None:
                    self._elbow_back_limit = float(solution_rad[ELBOW_IDX])
                back_margin = _np.deg2rad(float(self.tune["elbow_back_block"]["tolerance_deg"]))
                direction = str(self.tune["elbow_back_block"]["direction"]).lower()
                if direction == "decrease":
                    solution_rad[ELBOW_IDX] = max(solution_rad[ELBOW_IDX], self._elbow_back_limit - back_margin)
                else:
                    solution_rad[ELBOW_IDX] = min(solution_rad[ELBOW_IDX], self._elbow_back_limit + back_margin)

            self._prev_q = solution_rad

            # Update visualization (expects radians)
            if self.config.enable_visualization and self.urdf_vis:
                self.urdf_vis.update_cfg(solution_rad)

            # Convert to degrees (phone teleoperator always uses degrees, robot is auto-configured)
            solution_final = np.rad2deg(solution_rad)

            # Apply backward compatibility transformations for old calibration system
            # Based on PR #777 backward compatibility documentation
            
            # For SO100/SO101 backward compatibility (applied in degrees):

            # ORIGINAL SO100 CALIBRATIONS

            # shoulder_lift (index 1): direction reversal + 90° offset
            if len(solution_final) > 1:
                solution_final[1] = -(solution_final[1] - 90)
            
            # elbow_flex (index 2): 90° offset
            if len(solution_final) > 2:
                solution_final[2] -= 90
            
           # wrist_roll (index 4): direction reversal + 90° offset
            if len(solution_final) > 4:
                solution_final[4] = -(solution_final[4] + 90)

            # Sourccey Math Additions 

            # shoulder_pan (index 0): direction reversal
        #     if len(solution_final) > 0:
        #         solution_final[0] = solution_final[0]
            
        #    # shoulder_lift (index 1): direction reversal + 90° offset
        #     if len(solution_final) > 1:
        #         # solution_final[1] = (solution_final[1]) + 180
        #         solution_final[1] = (solution_final[1] + 180)

        #     # elbow_flex (index 2): direction reversal + 90° offset
        #     if len(solution_final) > 2:
        #         # solution_final[2] = -solution_final[2] - 180
        #         solution_final[2] = (solution_final[2] - 90)
            
        #     # wrist_flex (index 3): direction reversal
        #     if len(solution_final) > 3:
        #         solution_final[3] = solution_final[3]

        #     # wrist_roll (index 4): direction reversal + 90° offset
        #     if len(solution_final) > 4:
        #         solution_final[4] = -(solution_final[4] - 90)

            # CHATGPT's advice

            # # shoulder_lift (index 1): sign flip only
            # if len(solution_final) > 1:
            #     solution_final[1] = -solution_final[1]

            # # elbow_flex (index 2): no fixed offset or sign change
            # #   (axis is +X and rpy now embeds the old –90° roll)

            # # wrist_roll (index 4): sign flip only
            # if len(solution_final) > 4:
            #     solution_final[4] = -solution_final[4]


            # Update gripper state - convert percentage (0-100) to gripper position
            # gripper_value is 0-100, we need to map it to configured range
            gripper_range = self.config.gripper_max_pos - self.config.gripper_min_pos
            gripper_position = self.config.gripper_min_pos + (gripper_value / 100.0) * gripper_range
            solution_final[-1] = gripper_position
            
            # Update teleop state
            self.start_teleop = switch_state

            return self._format_action_dict(solution_final)

        except Exception as e:
            logger.error(f"Error getting action from {self}: {e}")
            # Return current position on error (safer than rest pose)
            return self._format_action_dict(current_joint_pos_deg)

    def _solve_ik(self, target_position: np.ndarray, target_wxyz: np.ndarray) -> list[float]:
        """Solve inverse kinematics for target pose. Returns solution in radians."""
        try:
            # Import IK solver from local module
            from .solve_ik import solve_ik
            
            solution = solve_ik(
                robot=self.robot,
                target_link_name=self.config.target_link_name,
                target_position=target_position,
                target_wxyz=target_wxyz,
            )
            
            return solution  # Always return radians
        except ImportError as e:
            logger.error(f"Could not import IK solver: {e}")
            # Return rest pose in radians
            return list(self.config.rest_pose)

    def _format_action_dict(self, joint_positions: list[float]) -> dict[str, Any]:
        """Format joint positions into action dictionary."""
        action_keys = list(self.action_features.keys())
        if len(joint_positions) != len(action_keys):
            logger.warning(
                f"Joint positions length ({len(joint_positions)}) doesn't match "
                f"action features length ({len(action_keys)})"
            )
            # Pad or truncate as needed
            joint_positions = joint_positions[:len(action_keys)]
            while len(joint_positions) < len(action_keys):
                joint_positions.append(0.0)
        
        return {key: pos for key, pos in zip(action_keys, joint_positions)}

    def _read_and_display_motor_positions(self, current_joint_pos: list[float]) -> None:
        """Read and display current motor positions in rest_pose format (radians)."""
        self._display_motor_positions_formatted(current_joint_pos, "5-SECOND TELEOP READING")
        
        # Also log to logger (phone teleoperator always works in degrees)
        current_joint_pos_rad = np.deg2rad(current_joint_pos)
        logger.info(f"Motor positions after 5 seconds - Degrees: {current_joint_pos}")
        logger.info(f"Motor positions after 5 seconds - Radians: {current_joint_pos_rad}")
        
        # rest_pose is always stored in radians for consistency with IK solver
        logger.info(f"rest_pose format: {tuple(current_joint_pos_rad)}")

    def _display_motor_positions_formatted(self, current_joint_pos: list[float], context: str) -> None:
        """Display motor positions in rest_pose format with given context."""
        # Convert to radians for rest_pose format (rest_pose is always stored in radians)
        # Phone teleoperator always works in degrees
        current_joint_pos_rad = np.deg2rad(current_joint_pos)
        
        # Format as tuple like rest_pose in config
        position_tuple = tuple(current_joint_pos_rad)
        
        formatted_values = ", ".join([f"{pos:.6f}" for pos in current_joint_pos_rad])

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to phone teleoperator (no-op for phone teleoperator)."""
        pass

    def disconnect(self) -> None:
        """Disconnect from phone and cleanup resources."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        try:
            if self.grpc_server:
                self.grpc_server.stop(0)
            
            if self.server:
                self.server.stop()
                
            self._is_connected = False
            self._phone_connected = False
            logger.info(f"{self} disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting {self}: {e}")

 