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
from collections.abc import Sequence
from functools import cached_property
from math import acos, degrees, isfinite, radians, sqrt
from typing import TYPE_CHECKING

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _menlo_sdk_available, require_package

from ..robot import Robot
from .config_asimov import ASIMOV_JOINTS, AsimovConfig

if TYPE_CHECKING or _menlo_sdk_available:
    import menlo.asimov as menlo
else:
    menlo = None

logger = logging.getLogger(__name__)

BASE_ACTIONS = ("x.vel", "y.vel", "theta.vel")
JOINT_INDEX = {name: i for i, name in enumerate(ASIMOV_JOINTS)}

# Each ankle is reported as two motors (A, B). Coupling from menlo-sdk 0.1.0,
# menlo/asimov/robots.py; limits kept just inside the firmware's 0.35 / 0.1 rad, which
# the SDK refuses to exceed in a trajectory.
_ANKLES = ((4, 5), (10, 11))
_ANKLE_K_PITCH, _ANKLE_K_ROLL = 2.02, 0.8
_ANKLE_PITCH_LIMIT, _ANKLE_ROLL_LIMIT = 0.33, 0.095

# Menlo's hardware-tuned Asimov 1 gains (asimov-1-amp), per joint without its side prefix.
_MENLO_AMP_KP = {
    "Hip_Pitch": 150.0,
    "Hip_Roll": 150.0,
    "Hip_Yaw": 150.0,
    "Knee": 150.0,
    "Ankle_A": 110.0,
    "Ankle_B": 110.0,
    "Shoulder_Pitch": 57.0,
    "Shoulder_Roll": 86.0,
    "Shoulder_Yaw": 96.0,
    "Elbow": 40.0,
    "Wrist_Yaw": 40.0,
    "Waist_Yaw": 65.0,
    "Neck_Yaw": 40.0,
    "Neck_Pitch": 40.0,
}
_MENLO_AMP_KD = {"Elbow": 2.0, "Wrist_Yaw": 2.0, "Neck_Yaw": 2.0, "Neck_Pitch": 2.0}


def _unsided(name: str) -> str:
    return name[2:] if name[:2] in ("L_", "R_") else name


MENLO_AMP_KP = tuple(_MENLO_AMP_KP[_unsided(name)] for name in ASIMOV_JOINTS)
MENLO_AMP_KD = tuple(_MENLO_AMP_KD.get(_unsided(name), 5.0) for name in ASIMOV_JOINTS)

# Asimov Edge damps a trajectory 2 s after its last setpoint; the watchdog runs this often.
_WATCH_PERIOD_S = 0.05
# After stand() -> balance(), check the robot still reports MOVE once Edge's timeout has passed.
_CONFIRM_BALANCE_S = 2.5


def _clip_ankles(pose: Sequence[float]) -> list[float]:
    """A reported pose with both ankles inside the pitch/roll a trajectory may command."""
    out = [float(value) for value in pose]
    for a, b in _ANKLES:
        pitch = (out[a] - out[b]) / (2 * _ANKLE_K_PITCH)
        roll = -(out[a] + out[b]) / (2 * _ANKLE_K_ROLL)
        pitch = min(max(pitch, -_ANKLE_PITCH_LIMIT), _ANKLE_PITCH_LIMIT)
        roll = min(max(roll, -_ANKLE_ROLL_LIMIT), _ANKLE_ROLL_LIMIT)
        out[a] = _ANKLE_K_PITCH * pitch - _ANKLE_K_ROLL * roll
        out[b] = -_ANKLE_K_PITCH * pitch - _ANKLE_K_ROLL * roll
    return out


def _with_stance(pose: Sequence[float], stance: Sequence[float]) -> list[float]:
    """``pose`` with both legs in ``stance`` (left hip pitch, knee, ankle pitch; the right leg
    mirrored), keeping each ankle's roll. Mirroring as tested on Asimov 1 hardware."""
    hip, knee, ankle = (float(value) for value in stance)
    out = [float(value) for value in pose]
    out[JOINT_INDEX["L_Hip_Pitch"]], out[JOINT_INDEX["L_Knee"]] = hip, knee
    out[JOINT_INDEX["R_Hip_Pitch"]], out[JOINT_INDEX["R_Knee"]] = -hip, -knee
    for (a, b), pitch in zip(_ANKLES, (ankle, -ankle), strict=True):
        roll = -(out[a] + out[b]) / (2 * _ANKLE_K_ROLL)
        out[a] = _ANKLE_K_PITCH * pitch - _ANKLE_K_ROLL * roll
        out[b] = -_ANKLE_K_PITCH * pitch - _ANKLE_K_ROLL * roll
    return _clip_ankles(out)


def _tilt_deg(state: menlo.State) -> float | None:
    """Body angle from vertical, from the IMU's projected gravity; None without a usable sample."""
    gravity = state.gravity
    if gravity is None or len(gravity) != 3 or not all(isfinite(value) for value in gravity):
        return None
    norm = sqrt(sum(value * value for value in gravity))
    if norm == 0:
        return None
    return degrees(acos(min(1.0, max(-1.0, -gravity[2] / norm))))


class Asimov(Robot):
    """Asimov 1 over the Menlo SDK, in one of two control modes.

    ``velocity``: bounded base velocities to the onboard walking policy, which keeps balancing.
    Locomotion actions follow LeRobot's mobile-base convention (m/s, m/s, deg/s).

    ``joints``: absolute targets (rad) for ``controlled_joints``. The first action takes the
    joints over (the walking policy turns off) and glides the legs into ``leg_stance`` if set;
    every joint not in the action keeps its target. A watchdog re-sends the targets while the
    caller pauses and hands the robot back on excessive tilt. Ending joint control (``stop()``,
    ``disconnect()``, watchdog) glides the controlled joints back, then ``stand()`` ->
    ``balance()``: a bare ``balance()`` after a trajectory is dropped by Asimov Edge 2 s later.

    Connect only observes. An operator prepares the robot using Asimov Manager: MOVE for
    velocity control; STAND or MOVE, with the robot supported, for joint control.
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
        self._joint_mode = config.control_mode == "joints"
        # Joint mode: hold() re-sends the targets in force (the watchdog also does).
        self.supports_command_hold = self._joint_mode
        self._lock = threading.RLock()
        self._handover = threading.Lock()
        self._target: list[float] | None = None  # every joint's target while we own the joints
        self._home: tuple[float, ...] | None = None  # measured pose when control was taken
        self._start_tilt = 0.0
        self._last_setpoint = 0.0
        self._ended: str | None = None  # why joint control ended, until reconnection
        self._watchdog: threading.Thread | None = None
        self._stop_watchdog = threading.Event()
        # Validate camera names before any resources are opened.
        _ = self.observation_features

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self._joint_mode:
            return dict.fromkeys((f"{name}.pos" for name in self.config.controlled_joints), float)
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
        self._ended = None
        try:
            self._client.connect("udp", timeout=self.config.connect_timeout_s, persist=False)
            if tuple(self._client.info.joint_names or ()) != ASIMOV_JOINTS:
                raise ValueError("Expected the Asimov 1 25-joint body reported by menlo-sdk 0.1.x.")
            self._observation_from_state(self._get_state())
            for camera in self.cameras.values():
                camera.connect()
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Failed to clean up Asimov after connection failure")
            raise

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
        if self._joint_mode:
            return self._send_joint_targets(action)
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

    # ── joint control ────────────────────────────────────────────────────────

    @property
    def _gains(self) -> dict[str, tuple[float, ...]]:
        if self.config.joint_gains == "menlo_amp":
            return {"kp": MENLO_AMP_KP, "kd": MENLO_AMP_KD}
        return {}  # Asimov Edge applies its own gain table.

    def _sent_targets(self, target: Sequence[float]) -> RobotAction:
        return {f"{name}.pos": target[JOINT_INDEX[name]] for name in self.config.controlled_joints}

    def _send_joint_targets(self, action: RobotAction) -> RobotAction:
        # Omitted joints keep their target: a position, unlike a velocity, is safe to keep.
        values = {key.removesuffix(".pos"): float(value) for key, value in action.items()}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("Asimov joint targets must be finite.")
        state = self._get_state()
        with self._lock:
            self._check_joint_control()
            taking_over = self._target is None
            if taking_over:
                self._take_over(state)
            elif state.faulted or state.mode not in (menlo.Mode.MOVE, menlo.Mode.STAND):
                self._ended = f"robot reports {getattr(state.mode, 'name', state.mode)}"
                self._target = None
                self._stop_watchdog.set()
                raise RuntimeError(
                    f"Asimov joint control ended: {self._ended}. The operator or firmware took over."
                )
        if taking_over:
            # Outside the lock, so the tilt watchdog can hand back during the glide.
            time.sleep(self.config.take_over_s)
            logger.info("Asimov joint control taken (tilt %.1f deg).", self._start_tilt)
        with self._lock:
            self._check_joint_control()
            assert self._target is not None and self._client is not None
            target = list(self._target)
            step = self.config.max_joint_step
            for joint, value in values.items():
                i = JOINT_INDEX[joint]
                low, high = self.config.joint_limits.get(joint, (float("-inf"), float("inf")))
                value = min(max(value, low), high)
                target[i] = min(max(value, target[i] - step), target[i] + step)
            self._client.trajectory(target, **self._gains)
            self._target = target
            self._last_setpoint = time.monotonic()
            return self._sent_targets(target)

    def _check_joint_control(self) -> None:
        if self._ended is not None:
            raise RuntimeError(f"Asimov joint control ended ({self._ended}). Reconnect to take it again.")

    def _take_over(self, state: menlo.State) -> None:
        """Put every joint under position control, gliding into the held pose."""
        if state.faulted or state.mode not in (menlo.Mode.MOVE, menlo.Mode.STAND):
            raise RuntimeError(
                "Asimov must be in STAND or MOVE without faults before joint control. Prepare it "
                "using Asimov Manager, with the robot supported."
            )
        tilt = _tilt_deg(state)
        if tilt is None:
            raise RuntimeError("Asimov reports no IMU gravity; the tilt watchdog cannot run.")
        assert self._client is not None
        pose = _clip_ankles([joint.pos for joint in state.joints])
        start = _with_stance(pose, self.config.leg_stance) if self.config.leg_stance is not None else pose
        self._home = tuple(pose)
        self._start_tilt = tilt
        # Owned from here: any failure below still ends with the handover in disconnect().
        self._target = list(start)
        self._client.set_joints(start, duration=self.config.take_over_s, wait=False, **self._gains)
        # The SDK glides and then holds the pose; the watchdog's keepalive waits for the glide.
        self._last_setpoint = time.monotonic() + self.config.take_over_s
        self._stop_watchdog = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, name="asimov-joint-watchdog", daemon=True)
        self._watchdog.start()

    def _watch(self) -> None:
        """Re-send the targets while the caller pauses, and hand back on excessive tilt."""
        stop, failing = self._stop_watchdog, False
        while not stop.wait(_WATCH_PERIOD_S):
            client = self._client
            if client is None:
                return
            try:
                state = client.get_state()
            except Exception:  # noqa: BLE001 - a missing sample is not a trip
                state = None
            if state is not None and isfinite(state.age_s) and state.age_s <= self.config.max_state_age_s:
                tilt = _tilt_deg(state)
                if tilt is not None and (
                    tilt > self.config.max_tilt_deg
                    or tilt > self._start_tilt + self.config.max_extra_tilt_deg
                ):
                    logger.error(
                        "Asimov leans %.1f deg (%.1f at take-over): handing back.", tilt, self._start_tilt
                    )
                    self._end_joint_control(f"tilt {tilt:.1f} deg")
                    return
            with self._lock:
                if stop.is_set() or self._target is None:
                    return
                if time.monotonic() - self._last_setpoint < self.config.keepalive_s:
                    continue
                try:
                    client.trajectory(self._target, **self._gains)
                    self._last_setpoint = time.monotonic()
                    failing = False
                except Exception as exc:  # noqa: BLE001 - retried next period
                    if not failing:
                        logger.warning("Asimov keepalive setpoint not sent: %s", exc)
                    failing = True

    @check_if_not_connected
    def hold(self) -> None:
        """Re-send the joint targets in force (joint mode); nothing before the first action."""
        with self._lock:
            if self._target is not None and self._client is not None:
                self._client.trajectory(self._target, **self._gains)
                self._last_setpoint = time.monotonic()

    def _end_joint_control(self, reason: str) -> None:
        """Glide the controlled joints back, then stand() -> balance(). Never raises.
        A second caller (disconnect() during a watchdog handover) waits for the first."""
        with self._handover:
            self._hand_back(reason)

    def _hand_back(self, reason: str) -> None:
        with self._lock:
            if self._target is None:
                return
            target, home = self._target, self._home or tuple(self._target)
            self._target, self._ended = None, reason
        self._stop_watchdog.set()
        watchdog, self._watchdog = self._watchdog, None
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=1.0)
        client = self._client
        if client is None:
            return
        try:
            state = client.get_state()
        except Exception:  # noqa: BLE001
            state = None
        if state is None or state.faulted or state.mode not in (menlo.Mode.MOVE, menlo.Mode.STAND):
            # The operator or firmware took it (DAMP, fault): never re-arm the robot from here.
            logger.warning(
                "Asimov reports %s: leaving it to the operator.", getattr(state, "mode", "no state")
            )
            return
        try:
            glide = list(target)  # legs stay where they are held
            for name in self.config.controlled_joints:
                glide[JOINT_INDEX[name]] = home[JOINT_INDEX[name]]
            client.set_joints(glide, duration=self.config.release_s, wait=False, **self._gains)
            time.sleep(self.config.release_s + 0.3)
        except Exception:  # noqa: BLE001
            logger.exception("Asimov: could not glide back before the handover")
        for attempt in range(2):
            try:
                client.stand(timeout=5.0)
            except Exception:  # noqa: BLE001 - balance() is still owed
                logger.exception("Asimov stand() failed")
            try:
                client.balance(timeout=5.0)
            except Exception:  # noqa: BLE001
                logger.exception("Asimov balance() failed: use Asimov Manager")
                return
            time.sleep(_CONFIRM_BALANCE_S)
            try:
                mode = client.get_state().mode
            except Exception:  # noqa: BLE001
                return
            if mode == menlo.Mode.MOVE:
                logger.info("Asimov handed back to its balance controller (%s).", reason)
                return
            logger.warning("Asimov reports %s after balance(), not MOVE (attempt %d).", mode, attempt + 1)

    def stop(self) -> None:
        """Velocity: end our hold, keeping onboard balancing. Joints: hand back to balancing."""
        if self._joint_mode:
            self._end_joint_control("stop")
            return
        if self._client is not None and self._command_sent:
            # An operator may already have changed mode. Do not re-enter MOVE.
            if self._client.get_state().mode == menlo.Mode.MOVE:
                self._client.balance(wait=False, timeout=0.0)
            self._command_sent = False

    def disconnect(self) -> None:
        """Never DAMP a biped. Velocity: the SDK sends zero before closing, without STAND.
        Joints: hand back with stand() -> balance() first (unless the operator took over)."""
        errors: list[Exception] = []
        if self._joint_mode:
            self._end_joint_control("disconnect")
            self._stop_watchdog.set()
        if self._client is not None:
            try:
                self._client.close()
                self._client = None
                self._command_sent = False
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
