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
import os
import threading
import time
from functools import cached_property
from math import acos, degrees, isfinite, radians
from typing import TYPE_CHECKING, Any

import cv2
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
NUM_LEG_MOTORS = 12  # firmware slots 0-11; arms, waist and neck follow
# start_pose names each ankle in joint space (pitch, roll), not by its motors (A, B).
ANKLE_POSE_JOINTS = {
    f"{side}_Ankle_{axis}": ASIMOV_JOINTS.index(f"{side}_Ankle_{motor}")
    for side in "LR"
    for axis, motor in (("Pitch", "A"), ("Roll", "B"))
}
POSE_JOINTS = frozenset({name for name in ASIMOV_JOINTS if "_Ankle_" not in name} | ANKLE_POSE_JOINTS.keys())
HEAD_CAMERA = "head"
HEAD_CAMERA_FEATURES = (HEAD_CAMERA, f"{HEAD_CAMERA}.age_s", f"{HEAD_CAMERA}.frame")
# After stand() -> balance(), Asimov Edge can still drop a robot that followed a trajectory to
# DAMP 2 s later; the handover checks the mode again after this delay.
HANDOVER_CHECK_S = 2.5


class LeanWatchdogError(RuntimeError):
    """The robot tilted more than the configured margin under arm control."""


def lean_deg(gravity) -> float:
    """Body tilt from upright (deg), from the IMU's gravity direction in the body frame."""
    g = np.asarray(gravity, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if not isfinite(norm) or norm == 0.0:
        raise ValueError("Asimov telemetry has no usable gravity direction.")
    return degrees(acos(float(np.clip(-g[2] / norm, -1.0, 1.0))))


def frame_to_rgb(frame: Any, width: int, height: int) -> np.ndarray:
    """A menlo-sdk camera Frame as an RGB uint8 array of shape (height, width, 3)."""
    if frame.encoding == "jpeg":
        image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Asimov's head camera sent an undecodable JPEG frame.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif frame.encoding in ("rgb8", "bgr8", "gray8"):
        image = np.asarray(frame.to_numpy())
        if frame.encoding == "bgr8":
            image = image[..., ::-1]
        elif frame.encoding == "gray8":
            image = np.repeat(image.reshape(image.shape[0], image.shape[1], 1), 3, axis=2)
    else:
        raise ValueError(f"Asimov's head camera sent {frame.encoding!r} frames, which LeRobot cannot decode.")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image, dtype=np.uint8)


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
    targets in radians, streamed as full-body trajectories with the walking policy off; a
    background loop keeps the setpoints going between actions, and arm control can start
    from a configured pose (a leg stance for a free-standing robot) and end by handing the
    robot to Menlo's walking policy. In ``whole_body`` mode LeRobot runs a leg policy on a background thread, eased in from
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
        # arms: full-body trajectory targets in firmware (motor) order, set from the measured
        # pose when arm control starts and cleared when it ends. _arm_lock guards them, the
        # setpoint loop and the handover, so no setpoint follows a handover.
        self._arm_lock = threading.RLock()
        self._joint_targets: np.ndarray | None = None
        self._start_targets: np.ndarray | None = None  # the pose arm control started from
        self._start_lean: float | None = None
        self._last_setpoint = float("-inf")
        self._arm_thread: threading.Thread | None = None
        self._arm_stop = threading.Event()
        # Why arm control ended on its own. With exit_mode="balance" it stays until reconnect:
        # the robot was handed to the walking policy and must not be taken back silently.
        self._arm_error: BaseException | None = None
        # Head camera: the last frame decoded, and how many distinct frames were seen.
        self._head_frame: Any = None
        self._head_image: np.ndarray | None = None
        self._head_count = 0
        self._validate_arm_options()
        # whole_body: the controller thread and its input, guarded by _input_lock.
        self._controller: AsimovLocomotionController | None = None
        self._controller_thread: threading.Thread | None = None
        self._controller_stop = threading.Event()
        self._controller_error: BaseException | None = None
        self._input_lock = threading.Lock()
        self._velocity = np.zeros(3)  # vx m/s, vy m/s, wz rad/s
        self._velocity_time = float("-inf")
        self._upper_targets: np.ndarray | None = None  # 25 joint-space targets; legs unused
        self._sim_env: Any = None
        # Validate camera names before any resources are opened.
        _ = self.observation_features

    def _validate_arm_options(self) -> None:
        unknown = sorted(self.config.start_pose.keys() - POSE_JOINTS)
        if unknown:
            raise ValueError(
                f"Unknown start_pose joints {unknown}; use Asimov joint names, with ankles as "
                "<L|R>_Ankle_Pitch / <L|R>_Ankle_Roll."
            )
        for name in ("kp", "kd"):
            gains = getattr(self.config, name)
            if gains is not None and (
                len(gains) != len(ASIMOV_JOINTS) or any(not isfinite(g) or g <= 0 for g in gains)
            ):
                # Asimov Edge substitutes its DAMP gains for a gain of zero or less.
                raise ValueError(
                    f"{name} needs {len(ASIMOV_JOINTS)} finite positive values (firmware order)."
                )

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
        if self.config.observe_velocities:
            features.update(dict.fromkeys((f"{name}.vel" for name in ASIMOV_JOINTS), float))
        for group, axes in (("gravity", "xyz"), ("gyro", "xyz"), ("quat", "wxyz")):
            features.update(dict.fromkeys((f"imu.{group}.{axis}" for axis in axes), float))
        features.update({"mode": float, "faulted": float})
        if self.config.head_camera:
            height, width = self.config.head_camera_height, self.config.head_camera_width
            features.update(dict(zip(HEAD_CAMERA_FEATURES, ((height, width, 3), float, float), strict=True)))
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
        return (
            self._client is not None
            or self._sim_env is not None
            or any(camera.is_connected for camera in self.cameras.values())
        )

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
                self._start_sim()
            self._connect_sdk()
            assert self._client is not None
            if tuple(self._client.info.joint_names or ()) != ASIMOV_JOINTS:
                raise ValueError("Expected the Asimov 1 25-joint body reported by menlo-sdk 0.1.x.")
            self._observation_from_state(self._get_state())
            if self.config.head_camera:
                # Wait for the stream once, so every observation has a frame.
                self._client.camera.photo(timeout=self.config.head_camera_timeout_s)
            for camera in self.cameras.values():
                camera.connect()
            if self.config.control_mode == "whole_body":
                self._start_controller()
            if self.config.control_mode == "arms" and self.config.start_pose:
                # The policy's first observation is already in the start pose.
                self._send_arm_action({})
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Failed to clean up Asimov after connection failure")
            raise

    def _start_sim(self) -> None:
        """Launch the lerobot/asimov-mujoco sim, which emulates Asimov Edge on 127.0.0.1."""
        from lerobot.envs import make_env

        suites = make_env(self.config.sim_env)
        self._sim_env = next(iter(suites.values()))[0].envs[0]
        self._sim_env.reset()
        self._sim_env.simulator.start_as_thread()

    def _connect_sdk(self) -> None:
        media: dict[str, Any] = {}
        if self.config.head_camera:
            # Control and state stay on UDP; Asimov Manager hands out the LiveKit room for video.
            credential = os.environ.get("MENLO_CREDENTIAL", "")
            if not credential:
                raise RuntimeError(
                    "Asimov's head camera needs an SDK credential from the Developer page of Asimov "
                    "Manager in the MENLO_CREDENTIAL environment variable."
                )
            media["livekit"] = menlo.ManagerConfig(
                url=self.config.manager_url or f"http://{self.config.host}", credential=credential
            )
        connection = menlo.ConnectionConfig(
            udp=menlo.UdpConfig(
                host=self.config.host,
                command_port=self.config.command_port,
                state_bind=(self.config.state_host, self.config.state_port),
                state_source=self.config.host,
                version_port=self.config.version_port,
                version_timeout=self.config.connect_timeout_s,
            ),
            **media,
        )
        self._client = menlo.Robot(
            connection,
            limits=menlo.Limits(
                vx=self.config.max_linear_velocity,
                vy=self.config.max_lateral_velocity,
                vyaw=radians(self.config.max_angular_velocity),
            ),
        )
        if self.config.head_camera:
            self._client.connect(
                "hybrid",
                timeout=self.config.connect_timeout_s,
                media_timeout=self.config.head_camera_timeout_s,
                persist=False,
            )
        else:
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
        if self.config.observe_velocities:
            if any(joint.vel is None for joint in state.joints):
                raise ValueError("Asimov telemetry is missing joint velocities.")
            observation.update({f"{joint.name}.vel": float(joint.vel) for joint in state.joints})
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

    def _head_observation(self) -> RobotObservation:
        """The newest head frame, its age since capture and a count of distinct frames, so a
        policy can tell a new frame from a repeated one and place it in time."""
        assert self._client is not None
        frame = self._client.camera.latest()
        if frame is None:
            raise RuntimeError("Asimov's head camera has sent no frame.")
        if frame is not self._head_frame:
            config = self.config
            self._head_image = frame_to_rgb(frame, config.head_camera_width, config.head_camera_height)
            self._head_frame = frame
            self._head_count += 1
        age = float(frame.age_s) + self.config.head_camera_latency_s
        image, age_key, frame_key = HEAD_CAMERA_FEATURES
        return {image: self._head_image, age_key: age, frame_key: float(self._head_count)}

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation = self._observation_from_state(self._get_state())
        if self.config.head_camera:
            observation.update(self._head_observation())
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
        with self._arm_lock:
            if self._arm_error is not None:
                error = self._arm_error
                if self.config.exit_mode == "release":
                    self._arm_error = None  # Reported once; the next action starts afresh.
                raise RuntimeError(str(error)) from error
            try:
                if self._joint_targets is None:
                    self._begin_arm_control()
                else:
                    self._check_arm_state(self._get_state())
                assert self._joint_targets is not None
                step = self.config.max_arm_step_rad
                for name, value in requested.items():
                    i = ASIMOV_JOINTS.index(name)
                    current = self._joint_targets[i]
                    self._joint_targets[i] = current + max(-step, min(step, value - current))
                self._send_targets(self._joint_targets)
                return {
                    f"{name}.pos": float(self._joint_targets[ASIMOV_JOINTS.index(name)])
                    for name in ARM_JOINTS
                }
            except Exception as exc:
                failure = exc
        # Outside the lock: ending arm control joins the setpoint loop, which takes it.
        self._fail_arm_control(failure)
        raise failure

    # ------------------------------------------------------------------ arms: setpoints and handover

    def _safe_pose(self, motors) -> np.ndarray:
        """Motor-space targets with both ankles inside the firmware's pitch/roll limits, which
        ``trajectory()`` otherwise rejects (a measured pose can be just outside them)."""
        motors = np.asarray(motors, dtype=np.float64)
        if menlo_robots is None:
            return motors.copy()
        return joints_to_motors(clip_ankles(motors_to_joints(motors)))

    def _apply_start_pose(self, motors: np.ndarray) -> np.ndarray:
        out = motors.copy()
        ankles = {name: v for name, v in self.config.start_pose.items() if name in ANKLE_POSE_JOINTS}
        for name, value in self.config.start_pose.items():
            if name not in ankles:
                out[ASIMOV_JOINTS.index(name)] = value
        if ankles:
            joints = motors_to_joints(out)
            for name, value in ankles.items():
                joints[ANKLE_POSE_JOINTS[name]] = value
            out = joints_to_motors(joints)
        return self._safe_pose(out)

    def _send_targets(self, targets: np.ndarray) -> None:
        assert self._client is not None
        gains: dict[str, list[float]] = {}
        if self.config.kp is not None and self.config.kd is not None:
            gains = {"kp": list(self.config.kp), "kd": list(self.config.kd)}
        self._client.trajectory(targets.tolist(), timeout=0.0, **gains)
        self._last_setpoint = time.monotonic()

    def _check_arm_state(self, state: menlo.State) -> None:
        if state.faulted:
            raise RuntimeError("Asimov reports a fault; refusing joint targets.")
        if state.mode in (menlo.Mode.DAMP, menlo.Mode.FAULT_DAMP):
            # Asimov Edge DAMPs when setpoints stop for 2 s, or an operator damped it.
            raise RuntimeError("Asimov left arm control (DAMP). Re-prepare it in Asimov Manager.")
        limit = self.config.lean_limit_deg
        if limit > 0 and self._start_lean is not None:
            extra = lean_deg(state.gravity) - self._start_lean
            if extra > limit:
                raise LeanWatchdogError(
                    f"Asimov leans {extra:.1f} deg more than when arm control started (limit {limit:g} deg)."
                )

    def _begin_arm_control(self) -> None:
        """Take every joint under position control from the measured pose, ease into
        start_pose if one is set, and start the setpoint loop. Caller holds _arm_lock."""
        state = self._get_state()
        if state.faulted:
            raise RuntimeError("Asimov reports a fault; refusing joint targets.")
        # The firmware follows a trajectory from an armed STAND or MOVE, not from DAMP.
        if state.mode not in (menlo.Mode.STAND, menlo.Mode.MOVE):
            raise RuntimeError("Asimov must be in STAND or MOVE before arm control. Use Asimov Manager.")
        if tuple(joint.name for joint in state.joints) != ASIMOV_JOINTS:
            raise ValueError("Asimov joint names/order changed; refusing joint targets.")
        measured = self._safe_pose([joint.pos for joint in state.joints])
        self._start_targets = measured.copy()
        self._start_lean = lean_deg(state.gravity) if self.config.lean_limit_deg > 0 else None
        # Legs, waist and neck stay at the measured pose unless start_pose moves them.
        self._joint_targets = measured.copy()
        if self.config.start_pose:
            self._ease_to(self._apply_start_pose(measured), self.config.start_pose_s)
        self._arm_stop.clear()
        self._arm_thread = threading.Thread(target=self._arm_loop, name="asimov-arm-setpoints", daemon=True)
        self._arm_thread.start()

    def _ease_to(self, goal: np.ndarray, seconds: float) -> None:
        """Interpolate the targets to ``goal`` over ``seconds``, sending each setpoint and
        checking the robot (fault, DAMP, lean) at every one. Caller holds _arm_lock."""
        assert self._joint_targets is not None
        start = self._joint_targets.copy()
        period = 1.0 / self.config.arm_loop_hz
        steps = max(1, round(seconds / period))
        next_tick = time.monotonic()
        for i in range(1, steps + 1):
            self._check_arm_state(self._get_state())
            self._joint_targets = start + i / steps * (goal - start)
            self._send_targets(self._joint_targets)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _arm_loop(self) -> None:
        """Re-send the latest targets whenever the caller has not for one period, and watch the
        robot at every period. Any failure ends arm control (with the configured exit)."""
        period = 1.0 / self.config.arm_loop_hz
        try:
            while not self._arm_stop.wait(period):
                with self._arm_lock:
                    if self._joint_targets is None or self._arm_stop.is_set():
                        return
                    self._check_arm_state(self._get_state())
                    if time.monotonic() - self._last_setpoint >= period:
                        self._send_targets(self._joint_targets)
        except Exception as exc:
            logger.error("Asimov arm control stopped: %s", exc)
            with self._arm_lock:
                self._arm_error = exc
            self._fail_arm_control(exc)

    def _fail_arm_control(self, failure: BaseException) -> None:
        """End arm control after ``failure``; with exit_mode="balance" the failure is kept and
        refuses further actions until reconnect."""
        if self.config.exit_mode == "balance" and self._joint_targets is not None:
            with self._arm_lock:
                self._arm_error = failure
        self._end_arm_control(failure)

    def _end_arm_control(self, failure: BaseException | None = None) -> None:
        """Stop the setpoint loop and end arm control as exit_mode says. Never raises."""
        self._arm_stop.set()
        thread = self._arm_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=HANDOVER_CHECK_S + 20.0)
            self._arm_thread = None
        with self._arm_lock:
            if self._joint_targets is None:
                return
            try:
                if self.config.exit_mode == "balance":
                    self._hand_over(failure)
                else:
                    logger.info("Asimov arm control ended; Asimov Edge DAMPs it in 2 s (exit_mode=release).")
            finally:
                self._joint_targets = None
                self._start_targets = None
                self._start_lean = None

    def _hand_over(self, failure: BaseException | None) -> None:
        """Arms, waist and neck glide back to where arm control started, then stand() and
        balance() hand the robot to Menlo's walking policy. Caller holds _arm_lock."""
        client = self._client
        if client is None or not client.connected:
            logger.error("Asimov's link is lost: cannot hand it to the walking policy. Use Asimov Manager.")
            return
        try:
            state = client.get_state()
            if state.faulted or state.mode in (menlo.Mode.DAMP, menlo.Mode.FAULT_DAMP):
                logger.error("Asimov is DAMP or faulted: no handover. Use Asimov Manager.")
                return
        except Exception:
            logger.exception("Asimov state unavailable before the handover; trying it anyway")
        glide = self.config.exit_glide_s > 0 and not isinstance(failure, LeanWatchdogError)
        if glide and self._joint_targets is not None and self._start_targets is not None:
            goal = self._joint_targets.copy()
            goal[NUM_LEG_MOTORS:] = self._start_targets[NUM_LEG_MOTORS:]  # legs keep their stance
            try:
                self._ease_to(goal, self.config.exit_glide_s)
            except Exception as exc:
                logger.error("Asimov glide back interrupted (%s); handing over now", exc)
        for attempt in range(2):
            try:
                # A bare balance() after trajectories is accepted, then Edge DAMPs 2 s later;
                # stand() first arms the robot so balance() holds MOVE.
                try:
                    client.stand(timeout=5.0)
                except Exception as exc:
                    logger.warning("Asimov stand() before balance(): %s", exc)
                client.balance(timeout=5.0)
                time.sleep(HANDOVER_CHECK_S)
                if client.get_state().mode == menlo.Mode.MOVE:
                    logger.info("Asimov handed to Menlo's walking policy (MOVE)")
                    return
                logger.warning("Asimov left MOVE after the handover (attempt %d)", attempt + 1)
            except Exception:
                logger.exception("Asimov handover attempt %d failed", attempt + 1)
        logger.error("Asimov is not balancing after the handover. Use Asimov Manager now.")

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

        In arm mode this ends arm control as ``exit_mode`` says: with "release" the setpoints
        stop and Asimov Edge DAMPs 2 s later, never starting the walking policy under a robot
        that may be hanging or seated; with "balance" the robot is handed to the walking
        policy (stand() then balance()). In whole-body mode the controller keeps balancing at
        zero velocity.
        """
        if self.config.control_mode == "whole_body":
            with self._input_lock:
                self._velocity[:] = 0.0
            return
        if self.config.control_mode == "arms":
            self._end_arm_control()
            return
        if self._client is not None and self._command_sent:
            # An operator may already have changed mode. Do not re-enter MOVE.
            if self._client.get_state().mode == menlo.Mode.MOVE:
                self._client.balance(wait=False, timeout=0.0)
            self._command_sent = False

    def disconnect(self) -> None:
        """The SDK sends zero before closing; never STAND or DAMP a balancing biped.

        In whole-body mode the setpoints stop here, and Asimov Edge DAMPs 2 s later. In arm
        mode arm control ends first, as ``exit_mode`` says (see ``stop()``).
        """
        errors: list[Exception] = []
        self._stop_controller()
        self._end_arm_control()
        self._arm_error = None
        if self._client is not None:
            try:
                self._client.close()
                self._client = None
                self._command_sent = False
                self._head_frame = self._head_image = None
            except Exception as exc:
                errors.append(exc)
        for camera in self.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception as exc:
                    errors.append(exc)
        if self._sim_env is not None:
            try:
                self._sim_env.close()
                self._sim_env = None
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
