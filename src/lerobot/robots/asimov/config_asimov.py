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

    # "arms" only. The settings below let a learned arm policy run from lerobot-rollout.
    # Instead of the gantry or a seat, the robot stands on its own feet in a leg stance held
    # by start_pose, under the lean watchdog, and is handed to Menlo's walking policy at the
    # end (exit_mode="balance" is required).
    free_standing: bool = False
    # Rate (Hz) of the setpoint loop that re-sends the latest targets between send_action()
    # calls. Asimov Edge DAMPs 2 s after the last setpoint, so a slow or paused caller
    # (a rollout reset phase, an interactive prompt) must not stop the setpoints.
    arm_loop_hz: float = 50.0
    # Joint targets (rad) to ease into when arm control starts, e.g. a leg stance, a neck aim or
    # an arm home pose. Keys are joint names, with each ankle given as "<L|R>_Ankle_Pitch" /
    # "<L|R>_Ankle_Roll" instead of its motors A/B; omitted joints keep their measured position.
    # When set, arm control starts in connect() (otherwise on the first action), easing over
    # start_pose_s.
    start_pose: dict[str, float] = field(default_factory=dict)
    start_pose_s: float = 3.0
    # PD gains sent with every setpoint, both or neither: 25 values each, in the firmware joint
    # order (ASIMOV_JOINTS). Without them Asimov Edge applies its own per-joint gain table.
    kp: list[float] | None = None
    kd: list[float] | None = None
    # What happens when arm control ends (stop(), disconnect(), a lost setpoint loop, the lean
    # watchdog):
    # "release": setpoints stop and Asimov Edge DAMPs 2 s later. For a robot hanging from its
    #   gantry or seated, under which the walking policy must not start.
    # "balance": the arms, waist and neck glide back to the pose arm control started from (over
    #   exit_glide_s, 0 to skip), then stand() and balance() hand the robot to Menlo's walking
    #   policy. For a robot with its feet on the ground (rope slack, or free-standing).
    exit_mode: str = "release"
    exit_glide_s: float = 4.0
    # Lean watchdog: extra tilt (deg) beyond the lean measured when arm control started that
    # ends arm control at once (no glide). None: 6 deg free-standing, off otherwise; 0: off.
    max_extra_lean_deg: float | None = None
    # Also observe joint velocities (<joint>.vel, rad/s, firmware order like the positions).
    observe_velocities: bool = False

    # Asimov's head camera, streamed over LiveKit through Asimov Manager (the SDK's "hybrid"
    # connection: control and state stay on UDP). Needs an SDK credential from the Developer
    # page of Asimov Manager in the MENLO_CREDENTIAL environment variable (kept off the
    # command line). Observed as "head" (RGB), "head.age_s" (seconds since capture, with
    # head_camera_latency_s added) and "head.frame" (counts distinct frames).
    head_camera: bool = False
    # Asimov Manager's address; empty: http://<host>.
    manager_url: str = ""
    # Frames of another size are resized to this one.
    head_camera_width: int = 1280
    head_camera_height: int = 720
    # Capture-to-reception delay (s) of the stream; measure it on your network (about 0.1 s
    # over Wi-Fi has been seen). Added to every frame's age.
    head_camera_latency_s: float = 0.0
    head_camera_timeout_s: float = 5.0

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
        if self.control_mode == "arms" and not (self.robot_supported or self.free_standing):
            raise ValueError(
                "Arm control turns Asimov's walking policy off, so a free-standing robot falls. "
                "Support the robot (gantry or seated), then pass --robot.robot_supported=true, or "
                "hold it in a leg stance with --robot.free_standing=true (see the Asimov docs)."
            )
        self._validate_arm_policy_options()
        self._validate_head_camera()
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

    @property
    def lean_limit_deg(self) -> float:
        """The lean watchdog's extra-tilt limit in degrees; 0 when it is off."""
        if self.max_extra_lean_deg is None:
            return 6.0 if self.free_standing else 0.0
        return self.max_extra_lean_deg

    def _validate_arm_policy_options(self) -> None:
        # Joint names and gain lengths are checked by the robot, which owns the joint list.
        if self.control_mode != "arms":
            arm_only = {
                "free_standing": self.free_standing,
                "start_pose": bool(self.start_pose),
                "kp": self.kp is not None,
                "kd": self.kd is not None,
                "exit_mode": self.exit_mode != "release",
                "max_extra_lean_deg": self.max_extra_lean_deg is not None,
            }
            used = [name for name, value in arm_only.items() if value]
            if used:
                raise ValueError(f"{', '.join(used)} only apply to --robot.control_mode=arms.")
            return
        if self.exit_mode not in ("release", "balance"):
            raise ValueError("exit_mode must be 'release' or 'balance'.")
        if not isfinite(self.arm_loop_hz) or not 5.0 <= self.arm_loop_hz <= 200.0:
            raise ValueError("arm_loop_hz must be between 5 and 200.")
        for name in ("start_pose_s", "exit_glide_s"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.max_extra_lean_deg is not None and (
            not isfinite(self.max_extra_lean_deg) or not 0 <= self.max_extra_lean_deg <= 45
        ):
            raise ValueError("max_extra_lean_deg must be between 0 (off) and 45.")
        if (self.kp is None) != (self.kd is None):
            raise ValueError("Pass both kp and kd, or neither (Asimov Edge ignores a lone gain).")
        if any(not isfinite(value) for value in self.start_pose.values()):
            raise ValueError("start_pose values must be finite.")
        if self.free_standing:
            if self.exit_mode != "balance":
                raise ValueError(
                    "A free-standing robot must be handed to the walking policy when arm control "
                    "ends: pass --robot.exit_mode=balance."
                )
            stance = {f"{side}_{joint}" for side in "LR" for joint in ("Hip_Pitch", "Knee", "Ankle_Pitch")}
            if missing := sorted(stance - self.start_pose.keys()):
                raise ValueError(
                    "Free-standing arm control needs a leg stance in start_pose that keeps the robot "
                    f"upright with the walking policy off; missing {', '.join(missing)}."
                )
            if self.lean_limit_deg <= 0:
                raise ValueError(
                    "Free-standing arm control needs the lean watchdog (max_extra_lean_deg > 0)."
                )

    def _validate_head_camera(self) -> None:
        if not self.head_camera:
            return
        if self.is_simulation:
            raise ValueError("The Asimov simulation has no LiveKit head camera; use sim_publish_images.")
        for name in ("head_camera_width", "head_camera_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isfinite(self.head_camera_latency_s) or self.head_camera_latency_s < 0:
            raise ValueError("head_camera_latency_s must be finite and non-negative.")
        if not isfinite(self.head_camera_timeout_s) or self.head_camera_timeout_s <= 0:
            raise ValueError("head_camera_timeout_s must be finite and positive.")
