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

import json
import logging
import time
from pathlib import Path
from functools import cached_property
from typing import Any, Optional, Sequence, Tuple

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
from lerobot.motors.feetech.tables import MODEL_RESOLUTION

try:
    from .config_remote_teleoperator_sourccey import PhoneTeleoperatorSourcceyConfig
except ImportError:  # pragma: no cover - legacy name fallback
    from .config_phone_teleoperator_sourccey import PhoneTeleoperatorSourcceyConfig  # type: ignore[import]

try:
    from lerobot.teleoperators.occulus.normalization import (  # type: ignore[import]
        extract_joint_limits_deg_from_urdf,
        normalize_values_to_0_100,
    )
except ImportError:  # pragma: no cover - fallback when module not available
    def _get_joint_limit_rad(urdf, joint_name: str) -> Tuple[float | None, float | None]:
        lower = None
        upper = None
        try:
            jm = getattr(urdf, "joint_map", None)
            if jm and joint_name in jm:
                limit = getattr(jm[joint_name], "limit", None)
                lower = getattr(limit, "lower", None)
                upper = getattr(limit, "upper", None)
            if (lower is None or upper is None) and hasattr(urdf, "joints"):
                for j in getattr(urdf, "joints", []):
                    if getattr(j, "name", None) == joint_name:
                        limit = getattr(j, "limit", None)
                        lower = getattr(limit, "lower", None)
                        upper = getattr(limit, "upper", None)
                        break
        except Exception:
            lower, upper = None, None
        return lower, upper

    def extract_joint_limits_deg_from_urdf(
        urdf,
        joint_names_in_order: Sequence[str],
        *,
        default_limits_deg: Tuple[float, float] = (-180.0, 180.0),
    ) -> list[Tuple[float, float]]:
        limits: list[Tuple[float, float]] = []
        for name in joint_names_in_order:
            lo_rad, hi_rad = _get_joint_limit_rad(urdf, name)
            if lo_rad is None or hi_rad is None:
                limits.append(default_limits_deg)
                continue
            lo_deg = np.degrees(float(lo_rad))
            hi_deg = np.degrees(float(hi_rad))
            if hi_deg < lo_deg:
                lo_deg, hi_deg = hi_deg, lo_deg
            limits.append((lo_deg, hi_deg))
        return limits

    def normalize_values_to_0_100(
        values_deg: Sequence[float],
        limits_deg: Sequence[Tuple[float, float]],
    ) -> list[float]:
        out: list[float] = []
        for val, (mn, mx) in zip(values_deg, limits_deg):
            if mx <= mn:
                out.append(50.0)
                continue
            norm = 100.0 * (float(val) - float(mn)) / (float(mx) - float(mn))
            if norm < 0.0:
                norm = 0.0
            elif norm > 100.0:
                norm = 100.0
            out.append(norm)
        return out

logger = logging.getLogger(__name__)


class PhoneTeleoperatorSourccey(Teleoperator):
    """
    Phone-based teleoperator that receives pose data from mobile phone via gRPC
    and converts it to robot control commands using inverse kinematics.
    
    This teleoperator integrates with the VirtualManipulator system from the daxie package
    to provide phone-based robot control for Sourccey robots.
    """

    config_class = PhoneTeleoperatorSourcceyConfig
    name = "phone_teleoperator_sourccey"

    def __init__(self, config: PhoneTeleoperatorSourcceyConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.arm_side = getattr(self.config, "arm_side", "left").lower()
        
        # Set default joint offsets based on arm side if not explicitly provided
        if hasattr(self.config, "joint_offsets_deg"):
            if self.config.joint_offsets_deg is None:
                # Initialize with default values if None
                self.config.joint_offsets_deg = {
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                }
            
            # Set shoulder_pan offset based on arm_side if it's still at default
            if self.config.joint_offsets_deg.get("shoulder_pan", 0.0) == 0.0:
                offset_value = 30.0 if self.arm_side == "right" else -30.0
                self.config.joint_offsets_deg["shoulder_pan"] = offset_value
        
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
        
        # Store last valid arm position for reset functionality
        self.last_valid_arm_position = None
        
        # Pose tracking (per-arm initial)
        if self.arm_side == "right":
            # Store original right arm initial pose for later mirroring when teleop starts
            self._original_right_position = np.array(getattr(self.config, "initial_position_right", self.config.initial_position))
            self._original_right_quat = np.array(getattr(self.config, "initial_wxyz_right", self.config.initial_wxyz))
            self.current_t_R = self._original_right_position.copy()
            self.current_q_R = self._original_right_quat.copy()
        else:
            self.current_t_R = np.array(self.config.initial_position)
            self.current_q_R = np.array(self.config.initial_wxyz)
        self.initial_phone_quat = None
        self.initial_phone_pos = None
        self.last_precision_mode = False

        self._joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        self._joint_limits_deg: list[tuple[float, float]] | None = None
        self._observation_uses_degrees = bool(getattr(self.config, "observation_uses_degrees", False))
        self._motor_models = dict(getattr(self.config, "motor_models", {}))
        self._calibration_helpers: dict[str, dict[str, float]] | None = self._load_joint_calibration()
        
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
        
        # Temporal smoothing state (per controlled arm)
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
                    0: {"step_deg": 2.0, "deadband_deg": 0.2},
                },
            },
        }
        
        # Connection timeout tracking
        self.last_phone_data_time = None
        self.phone_disconnection_timeout = 3.0  # seconds without data before considering disconnected

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Features for the actions produced by this teleoperator."""
        # Controlled arm depends on arm_side
        joints = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        prefix = "left_" if self.arm_side == "left" else "right_"
        motor_names = [f"{prefix}{j}" for j in joints]
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
            mesh_path = getattr(self.config, "mesh_path", None)
            
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
                
                logger.info(f"Using Sourccey paths - URDF: {urdf_path}, Mesh: {mesh_path}")
            else:
                # Fallback auto-detection if paths are empty
                try:
                    from pathlib import Path
                    import lerobot
                    
                    # Get the lerobot package root directory
                    lerobot_root = Path(lerobot.__file__).parent.parent
                    sourccey_model_path = lerobot_root / "lerobot" / "robots" / "sourccey" / "sourccey_v2beta" / "model"
                    
                    if sourccey_model_path.exists():
                        auto_urdf_path = str(sourccey_model_path / "Arm.urdf")
                        urdf_path = urdf_path or auto_urdf_path
                        if mesh_path:
                            logger.info(f"Auto-detected Sourccey URDF: {urdf_path}")
                        else:
                            logger.info(f"Auto-detected Sourccey URDF (no meshes): {urdf_path}")
                    else:
                        raise FileNotFoundError(f"Could not find Sourccey model directory at {sourccey_model_path}")
                except Exception as e:
                    logger.warning(f"Could not auto-detect Sourccey paths: {e}")
                    if not urdf_path:
                        raise ValueError("URDF path must be provided in config or model must exist in lerobot/robots/sourccey/sourccey_v2beta/model/")
                
            # Load URDF; meshes optional
            if mesh_path:
                self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_path)
            else:
                self.urdf = yourdfpy.URDF.load(urdf_path)
            self.robot = pk.Robot.from_urdf(self.urdf)
            try:
                self._joint_limits_deg = extract_joint_limits_deg_from_urdf(
                    self.urdf,
                    self._joint_names,
                    default_limits_deg=(-180.0, 180.0),
                )
            except Exception:
                self._joint_limits_deg = [(-180.0, 180.0)] * len(self._joint_names)
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
                if up < low:
                    low, up = up, low
                self._elbow_soft_stop = float(low + fraction * (up - low))
            else:
                self._elbow_soft_stop = None
        except Exception:
            self._elbow_soft_stop = None

    def _resolve_asset_path(self, path_str: str | None) -> Path | None:
        if not path_str:
            return None
        candidate = Path(path_str)
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            try:
                import lerobot

                lerobot_root = Path(lerobot.__file__).parent.parent
                candidate = lerobot_root / path_str
                if candidate.is_file():
                    return candidate
            except Exception:
                return None
        return candidate if candidate.is_file() else None

    def _load_joint_calibration(self) -> dict[str, dict[str, float]] | None:
        path_attr = "calibration_path_left" if self.arm_side == "left" else "calibration_path_right"
        cal_path = self._resolve_asset_path(getattr(self.config, path_attr, None))
        if cal_path is None:
            return None

        try:
            calibration_dict = json.loads(cal_path.read_text())
        except Exception:
            return None

        helpers: dict[str, dict[str, float]] = {}
        for joint in self._joint_names:
            entry = calibration_dict.get(joint)
            if not entry:
                continue
            model = self._motor_models.get(joint)
            if not model:
                continue
            max_res = float(MODEL_RESOLUTION.get(model, 4096)) - 1.0
            try:
                range_min = float(entry["range_min"])
                range_max = float(entry["range_max"])
            except Exception:
                continue
            if range_max <= range_min:
                continue
            mid = (range_min + range_max) / 2.0
            drive_mode = int(entry.get("drive_mode", 0))
            helpers[joint] = {
                "range_min": range_min,
                "range_max": range_max,
                "mid": mid,
                "drive_mode": drive_mode,
                "max_res": max_res,
            }

        return helpers or None

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
        if self.arm_side == "right":
            self.current_t_R = np.array(getattr(self.config, "initial_position_right", self.config.initial_position))
            self.current_q_R = np.array(getattr(self.config, "initial_wxyz_right", self.config.initial_wxyz))
        else:
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

        # Extract current robot positions from observation (controlled arm only)
        current_left_arm_pos_deg = None
        if observation is not None:
            try:
                # Define joint name patterns to look for (without any prefix requirements)
                joint_base_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                
                # Try to find keys that contain these joint names with .pos suffix
                found_keys = []
                for joint_name in joint_base_names:
                    matching_key = None
                    # Look for any key that contains the joint name and ends with .pos
                    for obs_key in observation.keys():
                        if joint_name in obs_key and obs_key.endswith(".pos"):
                            matching_key = obs_key
                            break
                    
                    if matching_key:
                        found_keys.append(matching_key)
                    else:
                        # If we can't find a key for this joint, we can't proceed
                        found_keys = []
                        break
                
                # If we found all required joint keys, extract the values
                if len(found_keys) == len(joint_base_names):
                    raw_joint_values = [observation[key] for key in found_keys]
                    if self._observation_uses_degrees:
                        current_left_arm_pos_deg = raw_joint_values
                    else:
                        current_left_arm_pos_deg = self._denormalize_observation_values(raw_joint_values)
                    logger.debug(f"Found joint keys: {found_keys}")
                else:
                    logger.debug(f"Could not find all required joint keys. Available keys: {list(observation.keys())}")
                    current_left_arm_pos_deg = None
                        
            except Exception as e:
                logger.warning(f"Could not extract left arm joint positions from observation: {e}")
        
        # If we successfully extracted position, store it as last valid position
        if current_left_arm_pos_deg is not None and not all(pos == 0.0 for pos in current_left_arm_pos_deg):
            self.last_valid_arm_position = current_left_arm_pos_deg.copy()
        
        # If no valid observation, use last valid position or return early to avoid unwanted movement
        if current_left_arm_pos_deg is None:
            if self.last_valid_arm_position is not None:
                current_left_arm_pos_deg = self.last_valid_arm_position.copy()
                logger.debug("Using last valid arm position")
            else:
                # No valid observation and no last position - avoid sending arm commands
                logger.warning("No valid observation data available - skipping arm control to avoid unwanted movement")
                # Still try to get phone data for base controls if available
                try:
                    data = self.pose_service.get_latest_pose(block=False) if self.pose_service else None
                    if data and getattr(self.config, "enable_base_from_phone", True):
                        base = data.get("base")
                        if base:
                            return self._merge_base_with_action({}, base=base)
                except Exception:
                    pass
                # Return empty action - no arm movement, no base movement
                return {}

        # Show initial motor positions immediately on first call (before phone connection)
        if not self.initial_positions_shown:
            self._display_motor_positions_formatted(current_left_arm_pos_deg, "INITIAL LEFT ARM POSITION")
            self.initial_positions_shown = True

        try:
            # FIRST: Check for phone disconnection by trying to get fresh data with a short timeout
            # This must happen before any other logic to ensure continuous rest position commands
            current_time = time.time()
            
            # Try to get fresh data with a short timeout to detect disconnection
            fresh_data = self.pose_service.get_latest_pose(block=True, timeout=0.1)
            
            if fresh_data is not None:
                # We received fresh data - phone is connected
                self.last_phone_data_time = current_time
                data = fresh_data
                
                # Log reconnection if we were previously disconnected
                if not self._phone_connected and not self.start_teleop:
                    logger.info("Phone reconnected - resuming normal operation")
            else:
                # No fresh data received within timeout
                if self.last_phone_data_time is None:
                    # First time, initialize the timer
                    self.last_phone_data_time = current_time
                
                # If we haven't received fresh data for the timeout period, consider phone disconnected
                # Keep returning rest position continuously until we get fresh data again
                if (current_time - self.last_phone_data_time > self.phone_disconnection_timeout):
                    
                    if self.start_teleop:  # Only log once when we first detect disconnection
                        logger.info("Phone disconnected - continuously returning to rest position until reconnection")
                    
                    # Set start_teleop to False and reset connection state
                    self.start_teleop = False
                    self._phone_connected = False
                    self.teleop_start_time = None
                    self.motor_positions_read = False
                    
                    # Continuously return rest position until phone reconnects
                    if self.arm_side == "right":
                        # Right arm: use rest_pose_right with shoulder_lift flip for hardware compatibility
                        rest_pose_deg = list(np.rad2deg(getattr(self.config, "rest_pose_right", self.config.rest_pose)))
                        if len(rest_pose_deg) > 1:
                            rest_pose_deg[1] = -rest_pose_deg[1]  # Flip shoulder_lift for right arm hardware
                    else:
                        # Left arm: use rest_pose with shoulder_lift flip for hardware compatibility
                        rest_pose_deg = list(np.rad2deg(self.config.rest_pose))
                        if len(rest_pose_deg) > 1:
                            rest_pose_deg[1] = -rest_pose_deg[1]  # Flip shoulder_lift for left arm hardware
                    formatted_rest = self._format_action_dict(
                        rest_pose_deg,
                        gripper_percent=self._extract_gripper_percent(rest_pose_deg),
                    )
                    return formatted_rest
                
                # Get the last known data (may be stale) for continued operation
                data = self.pose_service.get_latest_pose(block=False)

            # Handle phone connection
            if not self._phone_connected:
                # Pass current left arm position to connection setup (IK solver always expects radians)
                # Phone teleoperator always works in degrees (robot is auto-configured)
                curr_qpos_rad = np.deg2rad(current_left_arm_pos_deg)
                self.quat_RP, self.translation_RP = self._open_phone_connection(curr_qpos_rad)
                self._phone_connected = True

            if not self.start_teleop:
                # Teleop inactive: keep arms at rest, but allow base commands if configured
                if self.arm_side == "right":
                    # Right arm: use rest_pose_right with shoulder_lift flip for hardware compatibility
                    rest_pose_deg = list(np.rad2deg(getattr(self.config, "rest_pose_right", self.config.rest_pose)))
                    if len(rest_pose_deg) > 1:
                        rest_pose_deg[1] = -rest_pose_deg[1]  # Flip shoulder_lift for right arm hardware
                else:
                    # Left arm: use rest_pose with shoulder_lift flip for hardware compatibility
                    rest_pose_deg = list(np.rad2deg(self.config.rest_pose))
                    if len(rest_pose_deg) > 1:
                        rest_pose_deg[1] = -rest_pose_deg[1]  # Flip shoulder_lift for left arm hardware
                self._phone_connected = False
                self.teleop_start_time = None
                self.motor_positions_read = False
                base = data.get("base") if data is not None else None
                if getattr(self.config, "base_allow_when_inactive", True):
                    formatted_rest = self._format_action_dict(
                        rest_pose_deg,
                        gripper_percent=self._extract_gripper_percent(rest_pose_deg),
                    )
                    return self._merge_base_with_action(formatted_rest, base=base)
                else:
                    return self._format_action_dict(
                        rest_pose_deg,
                        gripper_percent=self._extract_gripper_percent(rest_pose_deg),
                    )
            
            # Start timer when teleop becomes active
            if self.teleop_start_time is None:
                self.teleop_start_time = time.time()
            
            # Check if 5 seconds have passed and we haven't read positions yet
            if not self.motor_positions_read and time.time() - self.teleop_start_time >= 5.0:
                self._read_and_display_motor_positions(current_left_arm_pos_deg)
                self.motor_positions_read = True

            switch_state = data.get("switch", False) if data is not None else False
            reset_mapping_pressed = data.get("reset_mapping", False) if data is not None else False
            is_resetting_state = data.get("is_resetting", False) if data is not None else False

            # Update reset state tracking - handle both is_resetting and reset_mapping
            current_is_resetting = is_resetting_state or reset_mapping_pressed
            
            # Check for reset transition (prev=False, current=True) - reset just started
            if self.prev_is_resetting == False and current_is_resetting == True:
                # Use the last valid arm position if available, otherwise current position
                if self.last_valid_arm_position is not None:
                    self.reset_hold_position = self.last_valid_arm_position.copy()
                    logger.info("Reset started - holding arm at last valid position")
                else:
                    self.reset_hold_position = current_left_arm_pos_deg.copy()
                    logger.info("Reset started - holding arm at current position")
            
            if current_is_resetting:
                self.prev_is_resetting = current_is_resetting
                # Return the captured hold position instead of current drifting position
                if self.reset_hold_position is not None:
                    logger.debug(f"Reset active - returning hold position: {self.reset_hold_position}")
                    formatted_hold = self._format_action_dict(
                        self.reset_hold_position,
                        gripper_percent=self._extract_gripper_percent(self.reset_hold_position),
                    )
                    if getattr(self.config, "base_allow_when_resetting", True):
                        base = data.get("base") if data is not None else None
                        return self._merge_base_with_action(formatted_hold, base=base)
                    return formatted_hold
                else:
                    # Fallback if no hold position captured yet
                    logger.warning("Reset active but no hold position captured - using current position")
                    formatted_current = self._format_action_dict(
                        current_left_arm_pos_deg,
                        gripper_percent=self._extract_gripper_percent(current_left_arm_pos_deg),
                    )
                    if getattr(self.config, "base_allow_when_resetting", True):
                        base = data.get("base") if data is not None else None
                        return self._merge_base_with_action(formatted_current, base=base)
                    return formatted_current

            # Check for reset transition (prev=True, current=False) - reset just ended
            if self.prev_is_resetting == True and current_is_resetting == False:
                if data is not None:
                    pos, quat = data["position"], data["rotation"]
                    self._reset_mapping(pos, quat)
                    logger.info("Reset ended - phone mapping reset to current arm position")
                else:
                    logger.warning("Reset ended but no phone data available for remapping")
                self.reset_hold_position = None  # Clear the hold position

            self.prev_is_resetting = current_is_resetting

            # Ensure we have valid data before processing pose
            if data is None:
                # If no data available, return current positions to maintain stability
                formatted_current = self._format_action_dict(
                    current_left_arm_pos_deg,
                    gripper_percent=self._extract_gripper_percent(current_left_arm_pos_deg),
                )
                return self._merge_base_with_action(
                    formatted_current,
                    base=data.get("base") if data else None
                )

            pos, quat, gripper_value = data["position"], data["rotation"], data["gripper_value"]

            # Map phone pose to robot pose
            t_robot, q_robot = self._map_phone_to_robot(pos, quat, data["precision"])

            # Solve inverse kinematics (returns radians)
            solution_rad = self._solve_ik(t_robot, q_robot)

            # Temporal smoothing and posture shaping
            if getattr(self, "tune", None) and self.tune.get("bypass_all_mods", False):
                self._prev_q = solution_rad
                solution_final = np.rad2deg(solution_rad)
                # Update teleop state and apply right arm initial position reversal when teleop starts
                prev_start_teleop = getattr(self, 'start_teleop', False)
                self.start_teleop = switch_state
                
                # Apply right arm initial position mirroring when teleop becomes active
                if self.arm_side == "right" and not prev_start_teleop and self.start_teleop:
                    # Mirror the right arm's initial position when teleop starts
                    # Apply comprehensive right arm reversals to match movement reversals
                    self.current_t_R = self._original_right_position.copy()
                    # Mirror position: flip Y (left-right) and potentially Z (up-down)
                    self.current_t_R[1] = -self.current_t_R[1]  # Flip Y coordinate (left-right)
                    
                    # Also apply quaternion mirroring for right arm orientation
                    self.current_q_R = self._original_right_quat.copy()
                    # Mirror orientation by flipping Y and Z components of quaternion
                    self.current_q_R[2] = -self.current_q_R[2]  # Flip Y component
                    self.current_q_R[3] = -self.current_q_R[3]  # Flip Z component
                    
                    logger.info("Applied comprehensive right arm initial position and orientation mirroring for teleop (bypass mode)")
                action_ctrl = self._format_action_dict(
                    solution_final,
                    gripper_percent=float(gripper_value),
                )
                if self.arm_side == "right":
                    action_ctrl = {k.replace("left_", "right_"): v for k, v in action_ctrl.items()}
                if not getattr(self.config, "emit_both_arms", True):
                    return self._merge_base_with_action(action_ctrl, base=data.get("base"))
                # Only emit the controlled arm - no commands for the other arm
                return self._merge_base_with_action(action_ctrl, base=data.get("base"))

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
            WRIST_ROLL_IDX = 4
            overhand_roll_target = 0.0  # rad
            solution_rad[WRIST_ROLL_IDX] = 0.95 * solution_rad[WRIST_ROLL_IDX] + 0.05 * overhand_roll_target

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

            # Apply Sourccey-specific transformations for calibration system
            # Based on Sourccey V2 Beta robot configuration
            
            # For Sourccey V2 Beta compatibility (applied in degrees):

            # shoulder_lift: no sign flip (use IK direction)

            # elbow_flex (index 2): no fixed offset or sign change
            #   (axis is +X and rpy now embeds the old –90° roll)

            # Apply joint-level reversals based on arm side
            if self.arm_side == "right":
                # Right arm needs comprehensive joint reversals for proper mirroring
                if len(solution_final) > 0:  # shoulder_pan
                    solution_final[0] = -solution_final[0]
                if len(solution_final) > 1:  # shoulder_lift  
                    solution_final[1] = -solution_final[1]
                if len(solution_final) > 2:  # elbow_flex
                    solution_final[2] = -solution_final[2]
                if len(solution_final) > 3:  # wrist_flex
                    solution_final[3] = -solution_final[3]
                if len(solution_final) > 4:  # wrist_roll
                    solution_final[4] = -solution_final[4]
            else:
                # Left arm: only wrist_roll flip for consistency
                if len(solution_final) > 4:
                    solution_final[4] = -solution_final[4]

            # Apply optional per-joint offsets (degrees)
            if getattr(self.config, "joint_offsets_deg", None):
                offsets = self.config.joint_offsets_deg or {}
                name_by_index = [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ]
                for i, name in enumerate(name_by_index):
                    if i < len(solution_final) and name in offsets and name != "gripper":
                        solution_final[i] += float(offsets[name])


            # Update gripper state - convert percentage (0-100) to gripper position
            # gripper_value is 0-100, we need to map it to configured range
            gripper_range = self.config.gripper_max_pos - self.config.gripper_min_pos
            gripper_position = self.config.gripper_min_pos + (gripper_value / 100.0) * gripper_range
            solution_final[-1] = gripper_position
            
            # Update teleop state and apply right arm initial position reversal when teleop starts
            prev_start_teleop = getattr(self, 'start_teleop', False)
            self.start_teleop = switch_state
            
            # Apply right arm initial position mirroring when teleop becomes active
            if self.arm_side == "right" and not prev_start_teleop and self.start_teleop:
                # Mirror the right arm's initial position when teleop starts
                # Apply comprehensive right arm reversals to match movement reversals
                self.current_t_R = self._original_right_position.copy()
                # Mirror position: flip Y (left-right) and potentially Z (up-down)
                self.current_t_R[1] = -self.current_t_R[1]  # Flip Y coordinate (left-right)
                
                # Also apply quaternion mirroring for right arm orientation
                self.current_q_R = self._original_right_quat.copy()
                # Mirror orientation by flipping Y and Z components of quaternion
                self.current_q_R[2] = -self.current_q_R[2]  # Flip Y component
                self.current_q_R[3] = -self.current_q_R[3]  # Flip Z component
                
                logger.info("Applied comprehensive right arm initial position and orientation mirroring for teleop")

            # Format action for selected arm
            action_ctrl = self._format_action_dict(
                solution_final,
                gripper_percent=float(gripper_value),
            )
            if self.arm_side == "right":
                # Mirror left->right if not at rest or resetting handled below
                action_ctrl = {k.replace("left_", "right_"): v for k, v in action_ctrl.items()}

            # If only emitting controlled arm, return now
            if not getattr(self.config, "emit_both_arms", True):
                return self._merge_base_with_action(action_ctrl, base=data.get("base"))

            # Only emit the controlled arm - no commands for the other arm
            return self._merge_base_with_action(action_ctrl, base=data.get("base"))

        except Exception as e:
            logger.error(f"Error getting action from {self}: {e}")
            # Return current positions on error (safer than rest pose)
            formatted_current = self._format_action_dict(
                current_left_arm_pos_deg,
                gripper_percent=self._extract_gripper_percent(current_left_arm_pos_deg),
            )
            return self._merge_base_with_action(formatted_current, base=data.get("base") if 'data' in locals() else None)

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

    def _ensure_joint_limits_loaded(self) -> None:
        if self._joint_limits_deg is not None:
            return
        if self.urdf is None:
            self._joint_limits_deg = [(-180.0, 180.0)] * len(self._joint_names)
            return
        try:
            self._joint_limits_deg = extract_joint_limits_deg_from_urdf(
                self.urdf,
                self._joint_names,
                default_limits_deg=(-180.0, 180.0),
            )
        except Exception:
            self._joint_limits_deg = [(-180.0, 180.0)] * len(self._joint_names)

    def _normalize_joint_degrees_to_m100(self, joint_positions_deg: Sequence[float]) -> list[float]:
        if self._calibration_helpers:
            return self._normalize_with_calibration(joint_positions_deg)
        self._ensure_joint_limits_loaded()
        limits = self._joint_limits_deg or [(-180.0, 180.0)] * len(joint_positions_deg)
        norm0_100 = normalize_values_to_0_100(joint_positions_deg, limits)
        return [(float(val) * 2.0) - 100.0 for val in norm0_100]

    def _normalize_with_calibration(self, joint_positions_deg: Sequence[float]) -> list[float]:
        norms: list[float] = []
        for deg, joint in zip(joint_positions_deg, self._joint_names):
            helper = self._calibration_helpers.get(joint) if self._calibration_helpers else None
            if not helper:
                norms.append(0.0)
                continue
            raw = (float(deg) * helper["max_res"] / 360.0) + helper["mid"]
            raw = float(np.clip(raw, helper["range_min"], helper["range_max"]))
            norm = ((raw - helper["range_min"]) / (helper["range_max"] - helper["range_min"])) * 200.0 - 100.0
            if helper["drive_mode"]:
                norm = -norm
            norms.append(float(np.clip(norm, -100.0, 100.0)))
        return norms

    def _extract_gripper_percent(self, joint_positions: Sequence[float]) -> float:
        if not joint_positions:
            return 0.0
        return float(np.clip(joint_positions[-1], 0.0, 100.0))

    def _denormalize_observation_values(self, raw_joint_values: Sequence[float]) -> list[float]:
        if self._calibration_helpers:
            return self._denormalize_with_calibration(raw_joint_values)
        self._ensure_joint_limits_loaded()
        limits = self._joint_limits_deg or [(-180.0, 180.0)] * len(self._joint_names)
        joint_count = len(self._joint_names)
        degs: list[float] = []
        for norm_m100, (mn, mx) in zip(raw_joint_values[:joint_count], limits):
            norm_clamped = float(np.clip(norm_m100, -100.0, 100.0))
            norm_fraction = (norm_clamped + 100.0) / 200.0
            degs.append(mn + norm_fraction * (mx - mn))
        if len(raw_joint_values) > joint_count:
            gripper_percent = float(np.clip(raw_joint_values[joint_count], 0.0, 100.0))
            degs.append(gripper_percent)
        else:
            degs.append(0.0)
        return degs

    def _denormalize_with_calibration(self, raw_joint_values: Sequence[float]) -> list[float]:
        joint_count = len(self._joint_names)
        degs: list[float] = []
        for norm_m100, joint in zip(raw_joint_values[:joint_count], self._joint_names):
            helper = self._calibration_helpers.get(joint) if self._calibration_helpers else None
            if not helper:
                degs.append(0.0)
                continue
            norm = float(np.clip(norm_m100, -100.0, 100.0))
            if helper["drive_mode"]:
                norm = -norm
            raw = ((norm + 100.0) / 200.0) * (helper["range_max"] - helper["range_min"]) + helper["range_min"]
            raw = float(np.clip(raw, helper["range_min"], helper["range_max"]))
            deg = (raw - helper["mid"]) * 360.0 / helper["max_res"]
            degs.append(float(deg))

        if len(raw_joint_values) > joint_count:
            gripper_percent = float(np.clip(raw_joint_values[joint_count], 0.0, 100.0))
            degs.append(gripper_percent)
        else:
            degs.append(0.0)
        return degs

    def _format_action_dict(
        self,
        joint_positions_deg: list[float],
        *,
        gripper_percent: float | None = None,
    ) -> dict[str, Any]:
        """Format joint positions (degrees) into normalization-aware action dictionary."""
        joint_subset = list(joint_positions_deg[:len(self._joint_names)])
        norm_m100 = self._normalize_joint_degrees_to_m100(joint_subset)
        prefix = "left_" if self.arm_side == "left" else "right_"

        action = {
            f"{prefix}{name}.pos": float(np.clip(value, -100.0, 100.0))
            for name, value in zip(self._joint_names, norm_m100)
        }

        if gripper_percent is None:
            if len(joint_positions_deg) > len(self._joint_names):
                gripper_percent = joint_positions_deg[len(self._joint_names)]
            else:
                gripper_percent = 0.0

        gripper_percent_clamped = float(np.clip(gripper_percent, 0.0, 100.0))
        action[f"{prefix}gripper.pos"] = gripper_percent_clamped
        return action

    def _merge_base_with_action(self, action: dict[str, Any], base: Optional[dict] = None) -> dict[str, Any]:
        """Merge base velocities from phone data into action if enabled.

        Applies scaling factors from config and only includes base keys when active.
        """
        try:
            if not getattr(self.config, "enable_base_from_phone", True):
                return action
            if not base:
                return action

            # Read raw analog values (in [-1, 1]) and apply optional scaling
            x = float(base.get("x.vel", 0.0)) * float(getattr(self.config, "base_scale_x", 1.0))
            y = float(base.get("y.vel", 0.0)) * float(getattr(self.config, "base_scale_y", 1.0))
            theta = float(base.get("theta.vel", 0.0)) * float(getattr(self.config, "base_scale_theta", 1.0))

            # If base_active is False but we have non-zero inputs, treat as active
            is_active = bool(base.get("active", False)) or (abs(x) > 0.0 or abs(y) > 0.0 or abs(theta) > 0.0)
            if not is_active:
                return action

            # Debug: Log when phone wheel commands are applied
            if abs(x) > 0.0 or abs(y) > 0.0 or abs(theta) > 0.0:
                print(f"DEBUG TELEOP WHEELS: Applying phone wheel command - x={x:.3f}, y={y:.3f}, theta={theta:.3f}")

            # Emit raw analog values in [-1,1]; client will scale via _from_analog_to_base_action
            merged = {**action}
            merged["x.vel"] = x
            merged["y.vel"] = y
            merged["theta.vel"] = theta
            return merged
        except Exception:
            return action


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
            self.start_teleop = False
            
            # Reset phone disconnection tracking
            self.last_phone_data_time = None
            
            # Reset position tracking
            self.last_valid_arm_position = None
            self.reset_hold_position = None
            
            logger.info(f"{self} disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting {self}: {e}")

 