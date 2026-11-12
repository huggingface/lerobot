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

from __future__ import annotations

import logging
import threading
import time
from functools import cached_property
from math import degrees, isfinite, radians
from typing import TYPE_CHECKING, cast

import numpy as np

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _menlo_sdk_available, require_package

from ..robot import Robot
from .asimov_kinematics import ASIMOV_ARM_JOINTS
from .config_asimov import AsimovConfig

if TYPE_CHECKING or _menlo_sdk_available:
    import menlo.asimov as menlo
    from menlo.asimov import robots as menlo_robots
else:
    menlo = None
    menlo_robots = None

if TYPE_CHECKING:
    from lerobot.controllers.asimov.locomotion import AsimovLocomotionController

logger = logging.getLogger(__name__)

# Fixed before connecting so dataset features can be built offline. Names/order from
# menlo-sdk 0.1.0, menlo/asimov/robots.py: ASIMOV_1_BIPED_JOINTS.
ASIMOV_JOINTS = (
    "L_Hip_Pitch",
    "L_Hip_Roll",
    "L_Hip_Yaw",
    "L_Knee",
    "L_Ankle_A",
    "L_Ankle_B",
    "R_Hip_Pitch",
    "R_Hip_Roll",
    "R_Hip_Yaw",
    "R_Knee",
    "R_Ankle_A",
    "R_Ankle_B",
    "L_Shoulder_Pitch",
    "L_Shoulder_Roll",
    "L_Shoulder_Yaw",
    "L_Elbow",
    "L_Wrist_Yaw",
    "R_Shoulder_Pitch",
    "R_Shoulder_Roll",
    "R_Shoulder_Yaw",
    "R_Elbow",
    "R_Wrist_Yaw",
    "Waist_Yaw",
    "Neck_Yaw",
    "Neck_Pitch",
)
BASE_ACTIONS = ("x.vel", "y.vel", "theta.vel")
ARM_JOINTS = (*ASIMOV_ARM_JOINTS["left"], *ASIMOV_ARM_JOINTS["right"])
ARM_ACTIONS = tuple(f"{name}.pos" for name in ARM_JOINTS)


def motors_to_joints(values) -> np.ndarray:
    """Reported ankle motors (A, B) -> ankle (pitch, roll), other entries unchanged. Linear,
    so it converts velocities too. menlo-sdk 0.1.0, menlo/asimov/robots.py."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    for a, b in menlo_robots.ASIMOV_1_BIPED_ANKLES:
        out[a] = (values[a] - values[b]) / (2 * menlo_robots.ANKLE_K_PITCH)
        out[b] = -(values[a] + values[b]) / (2 * menlo_robots.ANKLE_K_ROLL)
    return out


def joints_to_motors(values) -> np.ndarray:
    """Ankle (pitch, roll) -> motors (A, B), the frame ``trajectory()`` takes."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    for a, b in menlo_robots.ASIMOV_1_BIPED_ANKLES:
        out[a] = menlo_robots.ANKLE_K_PITCH * values[a] - menlo_robots.ANKLE_K_ROLL * values[b]
        out[b] = -menlo_robots.ANKLE_K_PITCH * values[a] - menlo_robots.ANKLE_K_ROLL * values[b]
    return out


def clip_ankles(joint_targets: np.ndarray) -> np.ndarray:
    """Joint-space targets with each ankle inside the pitch/roll range the firmware holds."""
    out = joint_targets.copy()
    for a, b in menlo_robots.ASIMOV_1_BIPED_ANKLES:
        out[a] = np.clip(out[a], -menlo_robots.ANKLE_PITCH_LIMIT_RAD, menlo_robots.ANKLE_PITCH_LIMIT_RAD)
        out[b] = np.clip(out[b], -menlo_robots.ANKLE_ROLL_LIMIT_RAD, menlo_robots.ANKLE_ROLL_LIMIT_RAD)
    return out


class Asimov(Robot):
    """Asimov 1 with onboard balancing and bounded velocity commands, or arm joint control.

    Connect only observes. In ``velocity`` mode an operator must put the robot in MOVE
    using Asimov Manager before sending actions; locomotion actions follow LeRobot's
    mobile-base convention (m/s, m/s, deg/s). In ``arms`` mode actions are arm joint
    targets in radians, streamed as full-body trajectories with the walking policy off.
    In ``whole_body`` mode LeRobot runs a leg policy on a background thread, eased in from
    the measured pose on connect; actions are both velocities and arm targets, and
    ``send_action`` only updates that thread's input. Joint positions are observations in
    radians.
    """

    config_class = AsimovConfig
    name = "asimov"
    record_sent_action = True

    def __init__(self, config: AsimovConfig) -> None:
        require_package("menlo-sdk", extra="asimov", import_name="menlo")
        super().__init__(config)
        self.config = config
        self._client: menlo.Robot | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._command_sent = False
        # Full-body trajectory targets in firmware order, set from the measured pose on the
        # first arm command and cleared by stop().
        self._joint_targets: list[float] | None = None
        # whole_body: the controller thread and its input, guarded by _input_lock.
        self._controller: AsimovLocomotionController | None = None
        self._controller_thread: threading.Thread | None = None
        self._controller_stop = threading.Event()
        self._controller_error: BaseException | None = None
        self._input_lock = threading.Lock()
        self._velocity = np.zeros(3)  # vx m/s, vy m/s, wz rad/s
        self._velocity_time = float("-inf")
        self._upper_targets: np.ndarray | None = None  # 25 joint-space targets; legs unused
        # Validate camera names before any resources are opened.
        _ = self.observation_features

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.control_mode == "arms":
            return dict.fromkeys(ARM_ACTIONS, float)
        if self.config.control_mode == "whole_body":
            return dict.fromkeys((*BASE_ACTIONS, *ARM_ACTIONS), float)
        return dict.fromkeys(BASE_ACTIONS, float)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = dict.fromkeys((f"{name}.pos" for name in ASIMOV_JOINTS), float)
        for group, axes in (("gravity", "xyz"), ("gyro", "xyz"), ("quat", "wxyz")):
            features.update(dict.fromkeys((f"imu.{group}.{axis}" for axis in axes), float))
        features.update({"mode": float, "faulted": float})
        for name, cfg in self.config.cameras.items():
            camera_features = {}
            if getattr(cfg, "use_rgb", True):
                camera_features[name] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                camera_features[f"{name}_depth"] = (cfg.height, cfg.width, 1)
            if features.keys() & camera_features.keys():
                raise ValueError(f"Camera {name!r} collides with another observation feature.")
            features.update(camera_features)
        return features

    @property
    def is_connected(self) -> bool:
        # Resource ownership, including after SDK link loss, so callers still clean up.
        # The SDK's live connection is checked before every observation and action.
        return self._client is not None or any(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True  # Calibration belongs to Asimov's firmware/Manager.

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Do not change posture, gains, or onboard calibration on connection."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            if self.config.is_simulation:
                from .asimov_sim import AsimovMujocoClient

                # Answers the trajectory/state subset of menlo.Robot that whole_body uses.
                self._client = cast("menlo.Robot", AsimovMujocoClient(onscreen=self.config.sim_onscreen))
                self._client.connect("sim")
            else:
                self._connect_sdk()
            assert self._client is not None
            if tuple(self._client.info.joint_names or ()) != ASIMOV_JOINTS:
                raise ValueError("Expected the Asimov 1 25-joint body reported by menlo-sdk 0.1.x.")
            self._observation_from_state(self._get_state())
            for camera in self.cameras.values():
                camera.connect()
            if self.config.control_mode == "whole_body":
                self._start_controller()
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Failed to clean up Asimov after connection failure")
            raise

    def _connect_sdk(self) -> None:
        connection = menlo.ConnectionConfig(
            udp=menlo.UdpConfig(
                host=self.config.host,
                command_port=self.config.command_port,
                state_bind=(self.config.state_host, self.config.state_port),
                state_source=self.config.host,
                version_port=self.config.version_port,
                version_timeout=self.config.connect_timeout_s,
            )
        )
        self._client = menlo.Robot(
            connection,
            limits=menlo.Limits(
                vx=self.config.max_linear_velocity,
                vy=self.config.max_lateral_velocity,
                vyaw=radians(self.config.max_angular_velocity),
            ),
        )
        self._client.connect("udp", timeout=self.config.connect_timeout_s, persist=False)

    def _start_controller(self) -> None:
        from lerobot.controllers.asimov.locomotion import AsimovLocomotionController

        state = self._get_state()
        # Like arm control, joint commands are followed from an armed STAND or MOVE only.
        if state.faulted or state.mode not in (menlo.Mode.STAND, menlo.Mode.MOVE):
            raise RuntimeError("Asimov must be in STAND or MOVE before whole-body control.")
        self._controller = AsimovLocomotionController(self.config.controller_repo_id)
        with self._input_lock:
            self._upper_targets = self._controller.default_pose.astype(np.float64)
            self._velocity[:] = 0.0
            self._velocity_time = float("-inf")
        self._controller_error = None
        self._controller_stop.clear()
        self._controller_thread = threading.Thread(
            target=self._controller_loop, name="asimov-controller", daemon=True
        )
        self._controller_thread.start()

    def _controller_loop(self) -> None:
        """Ease from the measured pose to the policy's default pose, then run the policy."""
        controller = self._controller
        assert controller is not None and self._client is not None
        dt = controller.control_dt
        ease_ticks = max(1, round(self.config.controller_ease_in_s / dt))
        start: np.ndarray | None = None
        tick = 0
        next_tick = time.monotonic()
        try:
            while not self._controller_stop.is_set():
                state = self._get_state()
                if state.faulted or state.mode in (menlo.Mode.DAMP, menlo.Mode.FAULT_DAMP):
                    raise RuntimeError("Asimov left whole-body control (DAMP or fault).")
                if any(joint.vel is None for joint in state.joints):
                    raise RuntimeError("Asimov telemetry is missing joint velocities.")
                q = motors_to_joints([joint.pos for joint in state.joints])
                dq = motors_to_joints([joint.vel for joint in state.joints])
                with self._input_lock:
                    assert self._upper_targets is not None
                    upper = self._upper_targets.copy()
                    fresh = time.monotonic() - self._velocity_time <= self.config.input_timeout_s
                    command = self._velocity.copy() if fresh else np.zeros(3)
                if tick < ease_ticks:
                    if start is None:
                        start = q
                    goal = upper.copy()
                    goal[: controller.leg_defaults.size] = controller.leg_defaults
                    targets = start + (tick + 1) / ease_ticks * (goal - start)
                    if tick + 1 == ease_ticks:
                        controller.reset()
                        logger.info("Asimov locomotion policy active")
                else:
                    targets = controller.run_step(q, dq, state.gyro, state.gravity, command, upper)
                self._client.trajectory(
                    joints_to_motors(clip_ankles(np.asarray(targets, dtype=np.float64))),
                    kp=controller.kp.tolist(),
                    kd=controller.kd.tolist(),
                    timeout=0.0,
                )
                tick += 1
                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._controller_stop.wait(delay)
                else:
                    next_tick = time.monotonic()
        except BaseException as exc:
            self._controller_error = exc
            logger.exception("Asimov controller stopped")

    def _stop_controller(self) -> None:
        self._controller_stop.set()
        if self._controller_thread is not None:
            self._controller_thread.join(timeout=2.0)
            self._controller_thread = None
        self._controller = None

    def _get_state(self) -> menlo.State:
        if self._client is None or not self._client.connected:
            raise DeviceNotConnectedError("Asimov's SDK connection is closed or lost. Reconnect the robot.")
        state = self._client.get_state()
        if not isfinite(state.age_s) or state.age_s > self.config.max_state_age_s:
            raise RuntimeError("Asimov telemetry is stale; refusing to use it.")
        return state

    def _observation_from_state(self, state: menlo.State) -> RobotObservation:
        if tuple(joint.name for joint in state.joints) != ASIMOV_JOINTS:
            raise ValueError("Asimov joint names/order changed; refusing an inconsistent observation.")
        observation: RobotObservation = {f"{joint.name}.pos": float(joint.pos) for joint in state.joints}
        for group, axes in (("gravity", "xyz"), ("gyro", "xyz"), ("quat", "wxyz")):
            values = getattr(state, group)
            if values is None or len(values) != len(axes):
                raise ValueError(f"Asimov telemetry is missing IMU {group}.")
            observation.update(
                {f"imu.{group}.{axis}": float(value) for axis, value in zip(axes, values, strict=True)}
            )
        if any(not isfinite(value) for value in observation.values()):
            raise ValueError("Asimov telemetry contains non-finite values.")
        observation.update({"mode": float(state.mode), "faulted": float(state.faulted)})
        return observation

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation = self._observation_from_state(self._get_state())
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[name] = camera.read_latest()
            if isinstance(camera, DepthCamera) and camera.use_depth:
                observation[f"{name}_depth"] = camera.read_latest_depth()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        unknown = action.keys() - self.action_features.keys()
        if unknown:
            raise ValueError(f"Unknown Asimov action keys: {sorted(unknown)}")
        if self.config.control_mode == "arms":
            return self._send_arm_action(action)
        if self.config.control_mode == "whole_body":
            return self._send_whole_body_action(action)
        # Omitted axes stop; never inherit a preceding nonzero command.
        values = {key: float(action.get(key, 0.0)) for key in BASE_ACTIONS}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("Asimov velocities must be finite.")
        state = self._get_state()
        if state.faulted or state.mode != menlo.Mode.MOVE:
            raise RuntimeError("Asimov must be in MOVE without faults. Prepare it using Asimov Manager.")
        assert self._client is not None
        sent = self._client.set_velocity(
            vx=values["x.vel"],
            vy=values["y.vel"],
            vyaw=radians(values["theta.vel"]),
            duration=self.config.command_timeout_s,
            wait=False,
            timeout=0.0,
        )
        self._command_sent = True
        # Sent.command includes SDK clamping. Sent is not a firmware acknowledgement.
        command = sent.command
        if not isinstance(command, menlo.Velocity):
            raise TypeError("Menlo SDK returned a non-velocity command.")
        return {"x.vel": command.vx, "y.vel": command.vy, "theta.vel": degrees(command.vyaw)}

    def _send_arm_action(self, action: RobotAction) -> RobotAction:
        # Omitted joints keep their previous target.
        requested = {key.removesuffix(".pos"): float(value) for key, value in action.items()}
        if any(not isfinite(value) for value in requested.values()):
            raise ValueError("Asimov joint targets must be finite.")
        state = self._get_state()
        if state.faulted:
            raise RuntimeError("Asimov reports a fault; refusing joint targets.")
        if self._joint_targets is None:
            # The firmware follows a trajectory from an armed STAND or MOVE, not from DAMP.
            if state.mode not in (menlo.Mode.STAND, menlo.Mode.MOVE):
                raise RuntimeError("Asimov must be in STAND or MOVE before arm control. Use Asimov Manager.")
            if tuple(joint.name for joint in state.joints) != ASIMOV_JOINTS:
                raise ValueError("Asimov joint names/order changed; refusing joint targets.")
            # Legs, waist and neck stay at the measured pose; arms ramp from it.
            self._joint_targets = [float(joint.pos) for joint in state.joints]
        elif state.mode in (menlo.Mode.DAMP, menlo.Mode.FAULT_DAMP):
            # Asimov Edge DAMPs when setpoints stop for 2 s, or an operator damped it.
            self._joint_targets = None
            raise RuntimeError("Asimov left arm control (DAMP). Re-prepare it in Asimov Manager.")

        step = self.config.max_arm_step_rad
        for name, value in requested.items():
            i = ASIMOV_JOINTS.index(name)
            current = self._joint_targets[i]
            self._joint_targets[i] = current + max(-step, min(step, value - current))
        assert self._client is not None
        self._client.trajectory(self._joint_targets, timeout=0.0)
        return {f"{name}.pos": self._joint_targets[ASIMOV_JOINTS.index(name)] for name in ARM_JOINTS}

    def _send_whole_body_action(self, action: RobotAction) -> RobotAction:
        if self._controller_error is not None:
            raise RuntimeError("Asimov's whole-body controller stopped.") from self._controller_error
        if self._controller_thread is None or not self._controller_thread.is_alive():
            raise RuntimeError("Asimov's whole-body controller is not running. Reconnect the robot.")
        values = {key: float(value) for key, value in action.items()}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("Asimov actions must be finite.")
        # Omitted axes stop; omitted joints keep their previous target.
        limits = (
            self.config.max_linear_velocity,
            self.config.max_lateral_velocity,
            self.config.max_angular_velocity,
        )
        velocity = [
            max(-limit, min(limit, values.get(key, 0.0)))
            for key, limit in zip(BASE_ACTIONS, limits, strict=True)
        ]
        step = self.config.max_arm_step_rad
        with self._input_lock:
            self._velocity[:] = (velocity[0], velocity[1], radians(velocity[2]))
            self._velocity_time = time.monotonic()
            targets = self._upper_targets
            assert targets is not None
            for name in ARM_JOINTS:
                value = values.get(f"{name}.pos")
                if value is not None:
                    i = ASIMOV_JOINTS.index(name)
                    targets[i] += max(-step, min(step, value - targets[i]))
            arms = {f"{name}.pos": float(targets[ASIMOV_JOINTS.index(name)]) for name in ARM_JOINTS}
        return {**dict(zip(BASE_ACTIONS, velocity, strict=True)), **arms}

    def stop(self) -> None:
        """End our velocity hold, keeping onboard balancing active.

        In arm mode this stops streaming setpoints; Asimov Edge DAMPs 2 s later. It never
        starts the walking policy under a robot that may be hanging or seated. In whole-body
        mode the controller keeps balancing at zero velocity.
        """
        if self.config.control_mode == "whole_body":
            with self._input_lock:
                self._velocity[:] = 0.0
            return
        if self.config.control_mode == "arms":
            self._joint_targets = None
            return
        if self._client is not None and self._command_sent:
            # An operator may already have changed mode. Do not re-enter MOVE.
            if self._client.get_state().mode == menlo.Mode.MOVE:
                self._client.balance(wait=False, timeout=0.0)
            self._command_sent = False

    def disconnect(self) -> None:
        """The SDK sends zero before closing; never STAND or DAMP a balancing biped.

        In whole-body mode the setpoints stop here, and Asimov Edge DAMPs 2 s later.
        """
        errors: list[Exception] = []
        self._stop_controller()
        if self._client is not None:
            try:
                self._client.close()
                self._client = None
                self._command_sent = False
                self._joint_targets = None
            except Exception as exc:
                errors.append(exc)
        for camera in self.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
