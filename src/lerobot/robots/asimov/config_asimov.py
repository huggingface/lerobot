# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from math import degrees, isfinite

from lerobot.cameras import CameraConfig
from lerobot.envs.configs import AsimovMujocoEnvConfig

from ..config import RobotConfig

SIM_HOST = "127.0.0.1"


@RobotConfig.register_subclass("asimov")
@dataclass
class AsimovConfig(RobotConfig):
    """Asimov 1 locomotion or arm control over the Menlo SDK's UDP transport.

    The robot is calibrated by its firmware: LeRobot does not calibrate it and ignores `calibration_dir`.
    """

    # The robot's address; the simulation's with is_simulation.
    host: str = ""
    command_port: int = 8850
    # Receive LAN telemetry on all local interfaces; the SDK filters by robot source IP.
    state_host: str = "0.0.0.0"  # nosec B104
    state_port: int = 8851
    version_port: int = 3000
    connect_timeout_s: float = 5.0
    # Reject old observations, independently of the SDK's lost-link detection.
    max_state_age_s: float = 0.5
    # Each command expires unless refreshed. The SDK sends zero when this expires.
    command_timeout_s: float = 0.25
    # Body-frame forward/left limits (m/s), and counter-clockwise yaw (deg/s).
    max_linear_velocity: float = 0.3
    max_lateral_velocity: float = 0.2
    max_angular_velocity: float = 30.0
    # Cameras use the standard LeRobot backends; UDP does not carry onboard video.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # "velocity": walk with x.vel/y.vel/theta.vel on the onboard balancing policy.
    # "arms": stream arm joint targets (<joint>.pos, radians). Menlo joint control puts
    # every joint under position control with the walking policy OFF: legs, waist and
    # neck are held where they were, and the robot does not balance.
    # "whole_body": LeRobot runs the leg policy from controller_repo_id in its own 50 Hz
    # loop (onboard walking policy off); actions are x.vel/y.vel/theta.vel plus arm targets,
    # and each send_action() only updates the controller's input.
    control_mode: str = "velocity"
    # Required for "arms": the robot hangs from its gantry or sits on a stool or bench.
    robot_supported: bool = False
    # Largest change of any arm target per send_action() call (rad).
    max_arm_step_rad: float = 0.1

    # "whole_body" only. A Hub repo with a Menlo lower-body locomotion policy.onnx.
    controller_repo_id: str = "Menlo/asimov1-locomotion-26-10-26-lower-body-27dof"
    # Time to move from the measured pose to the policy's default pose before it starts.
    controller_ease_in_s: float = 2.0
    # The controller walks at zero velocity when send_action() is older than this.
    input_timeout_s: float = 0.5

    # Launch the lerobot/asimov-mujoco simulation and connect to it on 127.0.0.1 instead of
    # the robot ("whole_body" only: the sim has no onboard walking policy). The robot starts
    # held by an elastic band; press 9 in the viewer to release it.
    is_simulation: bool = False
    sim_env: AsimovMujocoEnvConfig = field(init=False)
    # Publish the sim's head and wrist cameras over ZMQ on sim_camera_port, or open its
    # viewer instead when publishing is off.
    sim_publish_images: bool = False
    sim_camera_port: int = 5555
    sim_onscreen: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_simulation:
            if self.host.strip() not in ("", SIM_HOST):
                raise ValueError(f"The Asimov simulation listens on {SIM_HOST}; leave host unset.")
            self.host = SIM_HOST
        if not self.host.strip():
            raise ValueError("host must be the Asimov robot's address.")
        if self.control_mode not in ("velocity", "arms", "whole_body"):
            raise ValueError("control_mode must be 'velocity', 'arms' or 'whole_body'.")
        if self.is_simulation and self.control_mode != "whole_body":
            raise ValueError("The Asimov simulation needs --robot.control_mode=whole_body.")
        for name in ("controller_ease_in_s", "input_timeout_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.control_mode == "arms" and not self.robot_supported:
            raise ValueError(
                "Arm control turns Asimov's walking policy off, so a free-standing robot falls. "
                "Support the robot (gantry or seated), then pass --robot.robot_supported=true."
            )
        if self.control_mode == "whole_body" and not self.is_simulation and not self.robot_supported:
            raise ValueError(
                "Whole-body control turns Asimov's walking policy off and is not yet validated on "
                "the robot; Edge DAMPs it when LeRobot stops. Keep it on its gantry and pass "
                "--robot.robot_supported=true."
            )
        if not isfinite(self.max_arm_step_rad) or not 0 < self.max_arm_step_rad <= 0.5:
            raise ValueError("max_arm_step_rad must be finite and in (0, 0.5].")
        for name in ("command_port", "state_port", "version_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be an integer between 1 and 65535.")
        for name in ("connect_timeout_s", "max_state_age_s", "command_timeout_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        # Keep recorded actions within the firmware's caps, not just the SDK's limits.
        for name, cap in (
            ("max_linear_velocity", 0.4),
            ("max_lateral_velocity", 0.4),
            ("max_angular_velocity", degrees(0.8)),
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= cap:
                raise ValueError(f"{name} must be finite and between 0 and {cap}.")
        self.sim_env = AsimovMujocoEnvConfig(
            publish_images=self.sim_publish_images,
            camera_port=self.sim_camera_port,
            onscreen=self.sim_onscreen,
            # The robot's own simulation repository, as for the G1.
            trust_remote_code=True,
        )
