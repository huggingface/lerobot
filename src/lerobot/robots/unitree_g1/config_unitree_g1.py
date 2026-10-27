#!/usr/bin/env python

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

from dataclasses import dataclass, field
from math import isfinite

from lerobot.cameras import CameraConfig
from lerobot.envs.configs import G1EndEffector, UnitreeG1MujocoEnv

from ..config import RobotConfig
from .g1_embodiments import get_g1_embodiment
from .g1_utils import NUM_MOTORS


@dataclass
class SuperMimicControllerConfig:
    """Settings for ``SuperMimicController``: reference-motion tracking with a SuperMimic actor."""

    # Local directory or Hugging Face Hub model repo holding actor.ts and manifest.json.
    export: str | None = None
    export_revision: str | None = None
    # A reference clip CSV, or a directory of them (cycled in file-name order). One row per frame:
    # [root_pos(3), root_quat_xyzw(4), 29 joint positions].
    motions: str | None = None
    motion_fps: float = 30.0
    # Clip (file stem) selected at startup; None selects the first one.
    start_clip: str | None = None
    # Play a clip as soon as it is selected instead of balancing at its first frame until
    # START (or `controller.play`) is pressed.
    autoplay: bool = False
    # Select the next clip when one finishes, wrapping around after the last.
    auto_advance: bool = False
    # Re-run the export's golden vectors when loading the actor.
    check_golden_vectors: bool = True
    # CPU threads for actor inference.
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.motion_fps <= 0:
            raise ValueError("supermimic.motion_fps must be positive")
        if self.num_threads < 1:
            raise ValueError("supermimic.num_threads must be at least 1")


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    kp: list[float] | None = None
    kd: list[float] | None = None

    # Default joint positions
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Launch mujoco simulation
    is_simulation: bool = True

    # Supports dummy, dex1, dex3
    end_effector: G1EndEffector = G1EndEffector.DEX1

    # Loads the lerobot/unitree-g1-mujoco environment
    sim_env: UnitreeG1MujocoEnv = field(init=False)

    # Where the sim's cameras are published, or its viewer instead when publishing is off.
    sim_publish_images: bool = True
    sim_camera_port: int = 5555

    # Toggle the viewer on or off
    sim_onscreen: bool | None = None

    # Hold the simulated robot up by its torso with the sim's elastic band. Without it, every
    # reset places the robot standing on the floor in the controller's start pose.
    sim_elastic_band: bool = True

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP

    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Controller class name, e.g. GrootLocomotionController / HolosomaLocomotionController /
    # SonicWholeBodyController / SuperMimicController. None disables it.
    controller: str | None = None

    # Used when controller=SuperMimicController.
    supermimic: SuperMimicControllerConfig = field(default_factory=SuperMimicControllerConfig)

    embodiment: str = "g1_29"

    def __post_init__(self):
        super().__post_init__()
        spec = get_g1_embodiment(self.embodiment)
        if self.controller in spec.unsupported_controllers:
            raise ValueError(f"{self.controller} is not supported for {self.embodiment}")
        if self.controller == "SuperMimicController" and (
            self.supermimic.export is None or self.supermimic.motions is None
        ):
            raise ValueError(
                "SuperMimicController needs --robot.supermimic.export and --robot.supermimic.motions"
            )
        if self.gravity_compensation and not spec.supports_gravity_compensation:
            raise ValueError(f"Gravity compensation is not implemented for {self.embodiment}")
        default_kp, default_kd = spec.default_gains()
        self.kp = list(default_kp if self.kp is None else self.kp)
        self.kd = list(default_kd if self.kd is None else self.kd)
        inactive = set(range(NUM_MOTORS)) - {joint.value for joint in spec.joint_index}
        for name in ("kp", "kd", "default_positions"):
            values = getattr(self, name)
            if len(values) != NUM_MOTORS or not all(isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {NUM_MOTORS} finite values in DDS slot order")
            if name in ("kp", "kd") and any(value < 0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
            if any(values[index] != 0 for index in inactive):
                raise ValueError(f"{name} must be zero at inactive {self.embodiment} DDS slots")
        self.end_effector = G1EndEffector(self.end_effector)  # from Python it is still a string
        self.sim_env = UnitreeG1MujocoEnv(
            publish_images=self.sim_publish_images,
            camera_port=self.sim_camera_port,
            onscreen=self.sim_onscreen,
            end_effector=self.end_effector,
        )
