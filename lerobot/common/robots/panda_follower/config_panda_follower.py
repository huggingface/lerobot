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
from typing import Any

from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig

@RobotConfig.register_subclass("panda_follower")
@dataclass
class PandaConfig(RobotConfig):
    """Configuration for Franka Emika Panda robot."""
    
    # gRPC server configuration
    ip: str 
    
    # Safety limits for relative motion (radians)
    max_relative_target: float | dict[str, float] | None = 0.1  # 0.1 radians ≈ 5.7 degrees
    
    # Control parameters
    control_frequency: float = 20.0  # Hz
    timeout: float = 5.0  # seconds
    
    # Joint limits (in radians) - Panda's actual joint limits
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "panda_joint1": (-2.7437, 2.7437),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0421, -0.1518),
        "panda_joint5": (-2.8065, 2.8065),
        "panda_joint6": (0.5445, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
    })
    
    # Camera configurations (optional)
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": RealSenseCameraConfig(
                serial_number_or_name="817612070256",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    # Whether to use force/torque feedback
    use_force_feedback: bool = True
    
    # Whether to use Cartesian pose feedback
    use_cartesian_feedback: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate joint limits
        if len(self.joint_limits) != 7:
            raise ValueError("Panda robot must have exactly 7 joint limits defined")
            
        # Validate max_relative_target
        if isinstance(self.max_relative_target, dict):
            expected_joints = set(self.joint_limits.keys())
            if set(self.max_relative_target.keys()) != expected_joints:
                raise ValueError(
                    f"max_relative_target keys must match joint names: {expected_joints}"
                )
