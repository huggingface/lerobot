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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


# Default joint limits for R1D2 (from realman_r1d2_joint_limits.yaml)
DEFAULT_R1D2_JOINT_LIMITS = {
    "joint_1": (-178.0, 178.0),
    "joint_2": (-130.0, 130.0),
    "joint_3": (-135.0, 135.0),
    "joint_4": (-178.0, 178.0),
    "joint_5": (-128.0, 128.0),
    "joint_6": (-360.0, 360.0),
    "gripper": (1.0, 1000.0),
}


def load_joint_limits_from_yaml(yaml_path: str | Path) -> dict[str, tuple[float, float]]:
    """
    Load joint limits from a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML file containing joint limits
        
    Returns:
        Dictionary mapping joint names to (min, max) tuples
    """
    import yaml
    
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Joint limits file not found: {yaml_path}")
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    limits = {}
    joint_limits_data = config.get("joint_limits", config)
    
    for joint_name, limit_values in joint_limits_data.items():
        if isinstance(limit_values, list) and len(limit_values) == 2:
            limits[joint_name] = (float(limit_values[0]), float(limit_values[1]))
        elif isinstance(limit_values, dict):
            limits[joint_name] = (float(limit_values.get("min", -180)), 
                                   float(limit_values.get("max", 180)))
    
    return limits


@RobotConfig.register_subclass("realman_follower")
@dataclass
class RealManFollowerConfig(RobotConfig):
    """
    Configuration for RealMan robot follower (R1D2 and other models).
    
    Joint Mapping from SO101 Leader to RealMan R1D2:
    ================================================
    SO101 uses normalized values (-100 to 100 for arm, 0-100 for gripper)
    RealMan R1D2 uses degrees directly.
    
    The mapping is:
    - SO101 shoulder_pan  (-100..100) -> R1D2 joint_1 (-178..178°)
    - SO101 shoulder_lift (-100..100) -> R1D2 joint_2 (-130..130°)
    - SO101 elbow_flex    (-100..100) -> R1D2 joint_3 (-135..135°)
    - R1D2 joint_4 is FIXED at center (0°) - not mapped from SO101
    - SO101 wrist_flex    (-100..100) -> R1D2 joint_5 (-128..128°)
    - SO101 wrist_roll    (-100..100) -> R1D2 joint_6 (-360..360°)
    - SO101 gripper       (0..100)    -> R1D2 gripper (1..1000)
    """

    # Network configuration
    ip: str = "192.168.10.18"
    port: int = 8080

    # Robot model (R1D2, RM65, RM75, RML63, ECO65, GEN72)
    model: str = "R1D2"

    # Degrees of freedom for the arm (excluding gripper)
    # R1D2/RM65/RML63/ECO65: 6 DOF, RM75/GEN72: 7 DOF
    dof: int = 6

    # Motion velocity (1-100) - use high values (80-100) for responsive teleoperation
    velocity: int = 80

    # Collision detection level (0-8, higher = more sensitive)
    collision_level: int = 3

    # Whether to disable torque/servo on disconnect
    disable_torque_on_disconnect: bool = True

    # Fixed position for joint 4 when mapping from 5-joint leader (SO101)
    # This joint is held at center position during SO101 teleoperation
    fixed_joint_4_position: float = 0.0

    # Maximum relative target for safety (degrees per step)
    # Set to None to disable, or a float value to limit movement per step
    max_relative_target: float | dict[str, float] | None = 30.0

    # Cameras configuration
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Whether to use degrees for joint angles (always True for RealMan)
    use_degrees: bool = True

    # Gripper configuration
    gripper_speed: int = 500  # 1-1000
    gripper_force: int = 500  # 1-1000

    # Path to joint limits YAML file (optional, uses defaults if not provided)
    joint_limits_path: str | None = None

    # Joint limits (degrees) - R1D2 defaults from realman_r1d2_joint_limits.yaml
    # These are HARD LIMITS that will never be exceeded
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_R1D2_JOINT_LIMITS.copy()
    )

    # Whether to strictly enforce joint limits (recommended: True)
    enforce_joint_limits: bool = True

    # Invert mapping for specific joints (useful when SO101 and RealMan rotate in opposite directions)
    # Key is RealMan joint name (e.g., "joint_1"), value is True to invert
    invert_joints: dict[str, bool] = field(default_factory=dict)

    # Minimum Z position (meters) for end effector safety
    # Set to None to disable, or a float value (e.g., 0.05 = 5cm above table)
    # This prevents the arm from going below this Z height relative to base
    min_z_position: float | None = None
    
    # Action to take when Z limit would be violated: "clamp" or "reject"
    z_limit_action: str = "clamp"

    def __post_init__(self):
        super().__post_init__()
        
        # Load joint limits from YAML file if path is provided
        if self.joint_limits_path is not None:
            self.joint_limits = load_joint_limits_from_yaml(self.joint_limits_path)

