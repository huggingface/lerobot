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

"""Single-arm Metal with the gravity-compensated MIT control loop of the YAM follower."""

import logging
import math
import sys
import threading
import time
from collections.abc import Callable
from functools import cached_property
from typing import Any, cast

import numpy as np
from huggingface_hub import hf_hub_download

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model import GravityCompensation
from lerobot.utils import check_if_already_connected, check_if_not_connected, require_package
from lerobot.utils.constants import ROBOT_URDFS_REPO_ID

from ..robot import Robot
from ..yam_follower.mit_control import (
    _GRIPPER_FRICTION_NM,
    JointState,
    MitArmParams,
    MitCommand,
    MitController,
    MitGripperParams,
    _gravity_feedforward,
    _GripperForceLimiter,
    _joint_to_motor,
    clip_to_limits,
)
from ..yam_follower.yam_follower import _from_public, _to_public
from .config_metal_follower import (
    JOINT_LIMITS_DEG,
    METAL_FEATURE_NAMES,
    MOTOR_NAMES,
    MetalFollowerConfig,
    MetalFollowerRobotConfig,
)
from .mit_bus import MetalMotorsBus

logger = logging.getLogger(__name__)

_JOINT_LIMITS = np.radians(JOINT_LIMITS_DEG)
_GRAVITY_MODEL_REMOTE_PATH = "metal/v1/metal_with_gripper.urdf"
# URDF joints in motor order; URDF angles equal motor angles. The gripper's mass is lumped into the last link.
_MODEL_ARM_JOINTS = tuple(f"JOINT{i}" for i in range(1, 7))
_MODEL_BASE_FRAME = "base_link"
# Per-joint caps on the gravity torque, in newton-metres, from the Metal Arm v1 profile.
_MAX_GRAVITY_TORQUE_NM = np.array([1.0, 25.0, 10.0, 3.0, 1.0, 0.5])
# The gripper is closed at motor zero and opens to 116.4 motor degrees.
_GRIPPER_OPEN_RAD = math.radians(116.4)
# Return to the folded rest pose (zero) on disconnect. Torque goes off only once consecutive samples, each a new
# reply from every arm joint, show them near zero and still: the motors have no brakes.
_RETURN_SPEED_RAD_S = math.radians(30.0)
_RETURN_MARGIN_S = 2.0
_RETURN_PERIOD_S = 0.02
_REST_TOLERANCE_RAD = math.radians(3.0)
_REST_SPEED_RAD_S = math.radians(5.0)
_REST_SAMPLES = 5
# Whether Metal keeps holding when it can't verify the rest pose: after a fault, an obstruction or an interrupted
# return (unless the operator types RELEASE). `False` turns torque off instead, and the arm drops.
HOLD_WHEN_UNSAFE = True


def _metal_arm_params(config: MetalFollowerConfig) -> MitArmParams:
    """Build the control settings of a Metal arm's joints from its config."""
    return MitArmParams(
        joint_limits=_JOINT_LIMITS,
        kp=np.asarray(config.kp, dtype=float),
        kd=np.asarray(config.kd, dtype=float),
        max_joint_speed=math.radians(config.max_joint_speed_deg_s),
        max_tracking_error=math.radians(config.max_tracking_error_deg),
        gravity_factors=np.asarray(config.gravity_factors, dtype=float),
        max_gravity_torque=float(_MAX_GRAVITY_TORQUE_NM.max()),
        float_kd=np.asarray(config.float_kd, dtype=float),
        coulomb_friction=np.asarray(config.coulomb_friction, dtype=float),
        control_frequency=config.control_frequency,
        feedback_timeout_s=config.feedback_timeout_s,
        command_timeout_s=config.command_timeout_s,
        idle_mode=config.idle_mode,
    )


def _metal_gripper_params(config: MetalFollowerConfig) -> MitGripperParams:
    """Build the settings of a Metal gripper from its config.

    `MitGripperParams` describes a linear gripper. Metal's is rotary, so the finger stroke is the motor stroke:
    the force limit then reads as motor torque, and the blocked gripper holds
    `force_limit_n + _GRIPPER_FRICTION_NM`, which is `gripper_hold_torque_nm`.
    """
    return MitGripperParams(
        closed=0.0,
        open=_GRIPPER_OPEN_RAD,
        kp=config.gripper_kp,
        kd=config.gripper_kd,
        max_speed=config.max_gripper_speed,
        force_limit_n=config.gripper_hold_torque_nm - _GRIPPER_FRICTION_NM,
        finger_stroke_m=_GRIPPER_OPEN_RAD,
    )


class _FreeGripper(_GripperForceLimiter):
    """Gripper commands without force regulation, for a gripper with no position gain: the leader's."""

    def step(self, position: float, velocity: float, torque: float, command: float, now: float) -> MitCommand:
        params = self.gripper_params
        goal = params.to_raw(float(np.clip(command, 0.0, 1.0)))
        return MitCommand(position=goal, velocity=0.0, kp=0.0, kd=params.kd, torque=0.0)


class _MetalController(MitController):
    """`MitController` with a check of the state torque is enabled from, a free leader gripper, and a hold after
    a fault.

    `MitController` stops sending on a fault and relies on the motors' CAN timeout. Metal's motors have none,
    so they keep their last command: a hold for a tracking arm, but gravity torque without stiffness for a
    floating one. With `HOLD_WHEN_UNSAFE` this sends, once, a stiff hold at the measured pose; otherwise it
    disables torque. A failed start leaves torque to the robot, which turns it off only at rest.
    """

    def __init__(
        self, *args: Any, start_check: Callable[[JointState], None] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.start_check = start_check

    def start(self) -> None:
        """Read the arm, check it, enable torque with a stiff hold at its measured position and start control.

        Raises:
            RuntimeError: If the controller is already running.
            ConnectionError: If a fault is set, or the bus fails; torque may then be on.
            ValueError: If the start check refuses the measured state.
        """
        if self.active:
            raise RuntimeError(f"{self.name} controller is already running")
        if self.fault_event.is_set():
            raise ConnectionError(f"{self.name} controller can't start after a fault; reconnect first")
        self._stop.clear()
        self._set_initial_state(self.read_state())
        self._failure = None
        bus = cast(MetalMotorsBus, self._bus)
        bus.preload(_joint_to_motor(self.state.position, self._require_gripper_params()))
        # The other arm of a pair may have faulted meanwhile.
        if self.fault_event.is_set():
            raise ConnectionError(f"{self.name} controller can't start after a fault; reconnect first")
        bus.enable()
        thread = threading.Thread(target=self._run, name=f"{self.name}-controller", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self, timeout_s: float = 2.0, release: bool = True) -> None:
        """Stop the control thread and, if `release`, disable the motors; otherwise they keep their last command."""
        if release:
            super().stop(timeout_s)
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.name} controller did not stop; support the arm or cut motor power")
            self._thread = None

    def hold_at_measured(self, attempts: int = 3) -> list[str]:
        """Send a stiff hold at the measured pose, with gravity torque, to every joint with fresh feedback, and a
        zero-gain command to the gripper, retrying until every motor confirms it.

        Returns:
            `list[str]`: The motors without a confirmed hold after `attempts` tries, which keep their last command.
        """
        bus = cast(MetalMotorsBus, self._bus)
        params, n, names = self._params, self._params.num_joints, bus.motor_names
        unverified: list[str] = []
        for _ in range(attempts):
            states, _, fresh = bus.read_fresh()
            position = np.where(fresh[:n], states.position[:n], self.state.position[:n])
            try:
                torque = _gravity_feedforward(params, self._gravity(np.r_[position, 0.0]))
            except Exception:
                torque = np.zeros(n)
            commands = {
                names[i]: MitCommand(
                    float(position[i]), 0.0, float(params.kp[i]), float(params.kd[i]), float(torque[i])
                )
                for i in range(n)
                if fresh[i]
            }
            commands[names[n]] = MitCommand(float(states.position[n]), 0.0, 0.0, 0.0, 0.0)
            unconfirmed = bus.send_confirmed(commands, enabled=True)
            # The gripper is sent a limp command; it may also have released itself on its own CAN timeout.
            unverified = [name for i, name in enumerate(names[:n]) if not fresh[i] or name in unconfirmed]
            if not unverified:
                if names[n] in unconfirmed:
                    logger.info("%s: the gripper didn't confirm its limp command", self.name)
                return []
        logger.error(
            "%s: %s got no confirmed hold and keep their last command; support the arm",
            self.name,
            ", ".join(unverified),
        )
        return unverified

    def _set_initial_state(self, state: JointState) -> None:
        if self.fault_event.is_set():
            raise ConnectionError(f"{self.name} controller can't start after a fault; reconnect first")
        if self.start_check is not None:
            self.start_check(state)
        super()._set_initial_state(state)
        gripper_params = self._require_gripper_params()
        if gripper_params.kp == 0:
            self._gripper = _FreeGripper(gripper_params)

    def _run(self) -> None:
        super()._run()
        # Even if stop() is waiting: a fault's hold goes out before the robot decides what to do.
        if not self.fault_event.is_set() or not self._bus.enabled:
            return
        try:
            if HOLD_WHEN_UNSAFE:
                self.hold_at_measured()
                logger.warning(
                    "%s: holding the arm at its measured pose after a fault; support it", self.name
                )
            else:
                self._bus.disable_torque()
        except Exception:
            logger.exception(
                "%s: could not secure the arm after a fault; support the arm or cut motor power", self.name
            )


class MetalFollower(Robot):
    """A Metal arm exposing six joint positions and a gripper opening.

    Args:
        config (`MetalFollowerRobotConfig`):
            Hardware and control settings.
        fault_event (`threading.Event | None`, *optional*):
            Fault signal shared by both arms of a bimanual robot.
    """

    config_class = MetalFollowerRobotConfig
    name = "metal_follower"
    supports_position_hold = True

    def __init__(self, config: MetalFollowerRobotConfig, fault_event: threading.Event | None = None) -> None:
        require_package("python-can", extra="metal", import_name="can")
        if config.gravity_compensation:
            require_package("placo", extra="metal")
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.bus = MetalMotorsBus(
            config.port,
            config.can_interface,
            config.feedback_timeout_s,
            hold_kp=np.asarray(config.kp, dtype=float),
            hold_kd=np.asarray(config.kd, dtype=float),
        )
        self.gravity_model: GravityCompensation | None = None
        self.controller = _MetalController(
            str(self),
            self.bus,
            _metal_arm_params(config),
            self._gravity_torque,
            fault_event or threading.Event(),
            gripper_params=_metal_gripper_params(config),
            start_check=lambda state: self._check_initial_position(state.position),
        )
        self._connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        suffixes = (".pos", ".vel", ".torque") if self.config.use_velocity_and_torque else (".pos",)
        return {f"{name}{suffix}": float for name in MOTOR_NAMES for suffix in suffixes}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for name, cfg in self.config.cameras.items():
            if getattr(cfg, "use_rgb", True):
                features[name] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{name}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(METAL_FEATURE_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect the arm and start control from its initial position.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Unused: the joint zeros are stored in the motors.
        """
        self.connect_bus()
        try:
            for camera in self.cameras.values():
                camera.connect()
            self.controller.fault_event.clear()
            self.prepare_controller()
            self.start_controller()
        except BaseException:
            try:
                self._shutdown(return_to_rest=False)
            except Exception:
                logger.exception(f"Failed to clean up {self} after a connection failure")
            raise
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """Log where the zeros are: Metal keeps them in its motors, so LeRobot never changes them."""
        logger.info(
            f"{self} has no calibration step: the joint zeros are stored in the motors, at the folded rest "
            "pose with the gripper closed, and survive power cycles."
        )

    def configure(self) -> None:
        """Nothing to configure: the motors run in MIT mode."""

    @check_if_already_connected
    def connect_bus(self) -> None:
        """Connect the CAN interface, with torque off, once every motor answers.

        `connect` starts with it. A bimanual robot calls it for both arms before either gets torque.
        """
        try:
            self.bus.connect()
            self.bus.read_states()
            self._connected = True
        except BaseException:
            try:
                self._close(release=True)
            except Exception:
                logger.exception(f"Failed to clean up {self} after a connection failure")
            raise

    @check_if_not_connected
    def prepare_controller(self) -> None:
        """Load the gravity model and check the initial position, before the controller starts.

        A bimanual robot calls it for both arms before starting either controller.

        Raises:
            ValueError: If a joint is not within `initial_tolerance_deg` of `initial_position_deg`.
        """
        self._load_gravity_model()
        self._check_initial_position(self.controller.read_state().position)

    @check_if_not_connected
    def start_controller(self) -> None:
        """Enable torque at the measured position and start the control loop."""
        self.controller.start()
        self.controller.latest()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return the latest motor state and camera frames.

        Returns:
            `RobotObservation`: Joint state and the latest frame from each configured camera.

        Raises:
            RuntimeError: If control is not running.
        """
        if not self.controller.active:
            raise RuntimeError(f"{self} control is not running; reconnect")
        state = self.controller.latest()
        use_degrees = self.config.use_degrees
        values = {"pos": _to_public(state.position, use_degrees)}
        if self.config.use_velocity_and_torque:
            values["vel"] = _to_public(state.velocity, use_degrees)
            values["torque"] = state.torque
        result: dict[str, Any] = {
            f"{name}.{kind}": float(data[i])
            for i, name in enumerate(MOTOR_NAMES)
            for kind, data in values.items()
        }
        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                result[name] = camera.read_latest(max_age_ms=200)
            if isinstance(camera, DepthCamera) and camera.use_depth:
                result[f"{name}_depth"] = camera.read_latest_depth(max_age_ms=200)
        return result

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Clip a joint-position target to the joint limits and submit it to the controller.

        Args:
            action (`RobotAction`):
                Absolute targets for all six joints and the gripper.

        Returns:
            `RobotAction`: The clipped target submitted to the controller.

        Raises:
            RuntimeError: If the controller is not running.
            KeyError: If a joint or gripper target is missing.
            ValueError: If a target is NaN or infinite.
        """
        if not self.controller.active:
            raise RuntimeError("Metal commands require an active controller")
        values = np.asarray([action[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)
        target = clip_to_limits(_from_public(values, self.config.use_degrees), _JOINT_LIMITS)
        self.controller.set_target(target)
        sent = _to_public(target, self.config.use_degrees)
        return {f"{name}.pos": float(sent[i]) for i, name in enumerate(MOTOR_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        """Return to the folded rest pose, turn torque off there and release all hardware resources.

        If the rest pose can't be verified, or the return is interrupted, and `HOLD_WHEN_UNSAFE` is set, the arm
        holds stiffly at its measured pose and keeps holding after CAN closes, unless the operator types RELEASE.
        """
        if self._shutdown(return_to_rest=True):
            logger.info(f"{self} disconnected.")

    def _load_gravity_model(self) -> None:
        if not self.config.gravity_compensation or self.gravity_model is not None:
            return
        model_path = self.config.urdf_path or hf_hub_download(
            ROBOT_URDFS_REPO_ID, _GRAVITY_MODEL_REMOTE_PATH, repo_type="dataset"
        )
        self.gravity_model = GravityCompensation(model_path, _MODEL_ARM_JOINTS, base_frame=_MODEL_BASE_FRAME)

    def _gravity_torque(self, positions: np.ndarray) -> np.ndarray:
        """Model gravity torques, capped per joint so that, scaled by `gravity_factors`, they stay in the caps."""
        if self.gravity_model is None:
            return np.zeros(6)
        cap = _MAX_GRAVITY_TORQUE_NM / np.maximum(self.config.gravity_factors, 1e-6)
        return np.clip(self.gravity_model.torques(positions[:6]), -cap, cap)

    def _check_initial_position(self, position: np.ndarray) -> None:
        if self.config.initial_position_deg is not None and np.any(
            np.abs(np.rad2deg(position[:6]) - self.config.initial_position_deg)
            > self.config.initial_tolerance_deg
        ):
            raise ValueError(
                f"{self} is not within initial_tolerance_deg of initial_position_deg; "
                "fold it to its rest pose before connecting"
            )

    def _shutdown(self, return_to_rest: bool) -> bool:
        """Turn torque off only at the verified rest pose; otherwise hold at the measured pose (see `disconnect`).

        Returns:
            `bool`: Whether torque is off.
        """
        interrupt: BaseException | None = None
        at_rest = not self.bus.enabled
        if not at_rest:
            try:
                if return_to_rest:
                    self._return_to_rest()
                at_rest = self._verified_at_rest()
            except BaseException as exc:
                interrupt = exc
        released = self._close(release=at_rest or not HOLD_WHEN_UNSAFE)
        if interrupt is not None:
            raise interrupt
        return released

    def _return_to_rest(self) -> None:
        """Move the joints to zero at 30 deg/s; the gripper stays where it is. Stops early if control stops."""
        if not self.controller.active:
            return
        try:
            position = self.controller.latest().position
            goal = np.r_[np.zeros(6), position[6]]
            speed = min(_RETURN_SPEED_RAD_S, math.radians(self.config.max_joint_speed_deg_s))
            deadline = time.monotonic() + np.max(np.abs(goal - position)[:6]) / speed + _RETURN_MARGIN_S
            target, previous = position.copy(), time.monotonic()
            # Always replace the last action's target, even when the cached pose already reads as rest.
            first = True
            while (
                first or np.any(target != goal) or np.max(np.abs(position - goal)[:6]) > _REST_TOLERANCE_RAD
            ):
                first = False
                now = time.monotonic()
                if now > deadline:
                    logger.warning(f"{self} did not reach its rest pose")
                    return
                step = speed * (now - previous)
                previous = now
                # Lands exactly on the goal once it is within one step.
                remaining = goal[:6] - target[:6]
                target[:6] = goal[:6] - (remaining - np.clip(remaining, -step, step))
                self.controller.set_target(target)
                time.sleep(_RETURN_PERIOD_S)
                position = self.controller.latest().position
        except ConnectionError:
            logger.warning(f"{self} control stopped before the return to its rest pose")

    def _verified_at_rest(self) -> bool:
        """Whether consecutive samples, each a new reply from every arm joint, show them near zero and still.

        The gripper isn't checked: its light jaws don't move the arm when its torque goes off.
        """
        _, _, counts = self.bus.sample()
        good = 0
        deadline = time.monotonic() + _RETURN_MARGIN_S
        while time.monotonic() < deadline:
            previous = counts
            states, fresh, counts = self.bus.sample()
            new = all(counts[name] > previous[name] for name in self.bus.motor_names[:6])
            still = np.all(np.abs(states.velocity[:6]) <= _REST_SPEED_RAD_S)
            near = np.all(np.abs(states.position[:6]) <= _REST_TOLERANCE_RAD)
            good = good + 1 if new and fresh[:6].all() and still and near else 0
            if good >= _REST_SAMPLES:
                return True
            time.sleep(_RETURN_PERIOD_S)
        logger.warning(f"{self} is not verified still at its rest pose")
        return False

    def _operator_releases(self) -> bool:
        """Ask, in an interactive terminal, whether to turn torque off where the arm is."""
        if not sys.stdin.isatty():
            return False
        try:
            answer = input(
                f"[{self.id}] The arm is not at its start pose. Support it, then type RELEASE and press Enter to "
                "turn torque off, or press Enter to leave it holding: "
            )
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip() == "RELEASE"

    def _close(self, release: bool) -> bool:
        """Stop control and close CAN; return whether torque is off. Without `release`, first hold stiffly at the
        measured pose, then ask the operator, in a terminal, whether to turn torque off anyway."""
        try:
            self.controller.stop(release=release)
        except Exception:
            if self.controller.active:
                # Leave CAN open while the thread might still be using it; the bus lock serializes the access.
                if release:
                    try:
                        self.bus.disable_torque()
                    except Exception:
                        logger.exception("Could not disable Metal torque; support the arm or cut motor power")
                logger.exception("Metal controller did not stop; CAN remains open for a disconnect retry")
                raise
            # Torque could not be disabled: bus.disconnect() tries again, and raises after closing CAN.
            logger.exception(
                "Could not disable Metal torque; the motors may still hold; support the arm or cut motor power"
            )
        if not release and self.bus.enabled:
            try:
                unheld = self.controller.hold_at_measured()
            except Exception:
                logger.exception(f"{self}: could not command a hold")
                unheld = list(MOTOR_NAMES[:-1])
            if unheld:
                # An unconfirmed joint may still track its last target: keep CAN open for another disconnect.
                raise ConnectionError(
                    f"{self}: {', '.join(unheld)} have no confirmed hold. Support the arm, then call disconnect "
                    "again or cut motor power."
                )
            release = self._operator_releases()
            if not release:
                logger.error(f"{self} is still holding with torque on. Support it, then cut motor power.")
        try:
            self.bus.disconnect(release=release)
        finally:
            try:
                for camera in self.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()
            finally:
                self._connected = False
        return release
