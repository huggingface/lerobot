import importlib
import logging
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_jaka_zu7_follower import JAKAZu7FollowerConfig

logger = logging.getLogger(__name__)

# Keep these keys identical to the six arm joints in this project's SO-ARM101 leader.
JAKA_ZU7_JOINTS = (
    "shoulder_pan",
    "heavy_joint",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
JAKA_SERVO_PERIOD_S = 0.008


def _limit_joint_step(
    position: list[float],
    velocity: list[float],
    target: list[float],
    dt: float,
    max_velocity: float,
    max_acceleration: float,
) -> tuple[list[float], list[float]]:
    next_position = []
    next_velocity = []

    for pos, vel, goal in zip(position, velocity, target, strict=True):
        error = goal - pos
        braking_speed = math.sqrt(2 * max_acceleration * abs(error))
        desired_velocity = math.copysign(min(max_velocity, braking_speed), error) if error else 0.0
        velocity_delta = max(-max_acceleration * dt, min(max_acceleration * dt, desired_velocity - vel))
        new_velocity = vel + velocity_delta
        new_position = pos + new_velocity * dt

        if error and (goal - new_position) * error <= 0:
            new_position = goal
            new_velocity = 0.0

        next_position.append(new_position)
        next_velocity.append(new_velocity)

    return next_position, next_velocity


class JAKAZu7Follower(Robot):
    """LeRobot follower driver for the six-axis JAKA Zu 7."""

    config_class = JAKAZu7FollowerConfig
    name = "jaka_zu7_follower"

    def __init__(self, config: JAKAZu7FollowerConfig):
        super().__init__(config)
        self.config = config
        self._sdk: Any | None = None
        self._logged_in = False
        self._servo_mode_enabled = False
        self._connected = False
        self._command_position_deg: list[float] | None = None
        self._command_velocity_deg_s = [0.0] * len(JAKA_ZU7_JOINTS)
        self._last_command_time: float | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int | None, ...]]:
        features: dict[str, type | tuple[int | None, ...]] = dict.fromkeys(self.action_features, float)
        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[camera_name] = (camera.height, camera.width, 3)
            if isinstance(camera, DepthCamera) and camera.use_depth:
                features[f"{camera_name}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JAKA_ZU7_JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        try:
            try:
                sdk_module = importlib.import_module("jkrc")
            except ImportError as exc:
                raise ImportError(
                    "Could not import the JAKA Python SDK module 'jkrc'. Add the SDK's Linux Python 3 x86_64 "
                    "directory to PYTHONPATH and its directory containing libjakaAPI.so to LD_LIBRARY_PATH. "
                    "The SDK binary must be compatible with this Python installation."
                ) from exc

            self._sdk = sdk_module.RC(self.config.ip_address)
            self._check_sdk_result(self._sdk.login(), "login")
            self._logged_in = True

            if self.config.auto_power_on:
                self._check_sdk_result(self._sdk.power_on(), "power_on")
                time.sleep(self.config.power_on_delay_s)
            if self.config.auto_enable_robot:
                self._check_sdk_result(self._sdk.enable_robot(), "enable_robot")

            self._command_position_deg = self._read_joints_deg()
            self._check_sdk_result(self._sdk.servo_move_enable(True), "servo_move_enable(True)")
            self._servo_mode_enabled = True
            self._last_command_time = time.monotonic()
            self._connected = True

            for camera in self.cameras.values():
                camera.connect()

            logger.info("%s connected to JAKA controller at %s", self, self.config.ip_address)
        except Exception:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Cleanup failed after JAKA connection error")
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joints_deg = self._read_joints_deg()
        observation: RobotObservation = {
            f"{joint}.pos": position for joint, position in zip(JAKA_ZU7_JOINTS, joints_deg, strict=True)
        }

        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[camera_name] = camera.read_latest()
            if isinstance(camera, DepthCamera) and camera.use_depth:
                observation[f"{camera_name}_depth"] = camera.read_latest_depth()

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        allowed = set(self.action_features)
        extra = set(action) - allowed - {"gripper.pos"}
        if extra:
            raise KeyError(f"Unknown JAKA Zu 7 action keys: {sorted(extra)}")

        missing = allowed - action.keys()
        if missing:
            raise KeyError(f"Missing JAKA Zu 7 action keys: {sorted(missing)}")

        target_deg = [float(action[f"{joint}.pos"]) for joint in JAKA_ZU7_JOINTS]
        if not all(math.isfinite(value) for value in target_deg):
            raise ValueError("JAKA joint targets must all be finite numbers")

        if self._command_position_deg is None:
            raise RuntimeError("JAKA follower has no initial joint position")
        now = time.monotonic()
        servo_window_s = self.config.servo_step_num * JAKA_SERVO_PERIOD_S
        elapsed_s = servo_window_s if self._last_command_time is None else now - self._last_command_time
        dt = max(0.001, min(elapsed_s, servo_window_s))

        command_deg, command_velocity_deg_s = _limit_joint_step(
            position=self._command_position_deg,
            velocity=self._command_velocity_deg_s,
            target=target_deg,
            dt=dt,
            max_velocity=self.config.max_joint_velocity_deg_s,
            max_acceleration=self.config.max_joint_acceleration_deg_s2,
        )
        controller_command_deg = [
            sign * value + offset
            for sign, offset, value in zip(
                self.config.joint_signs, self.config.joint_offsets_deg, command_deg, strict=True
            )
        ]
        command_rad = [math.radians(value) for value in controller_command_deg]
        sdk = self._require_sdk()
        self._check_sdk_result(
            sdk.servo_j_extend(command_rad, 0, self.config.servo_step_num),
            "servo_j_extend",
        )
        self._command_position_deg = command_deg
        self._command_velocity_deg_s = command_velocity_deg_s
        self._last_command_time = now

        if "gripper.pos" in action:
            logger.debug("Ignoring SO-ARM101 leader gripper action; this JAKA setup has no gripper mapping")

        return {f"{joint}.pos": value for joint, value in zip(JAKA_ZU7_JOINTS, command_deg, strict=True)}

    def disconnect(self) -> None:
        first_error: Exception | None = None
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                first_error = first_error or exc

        if self._sdk is not None:
            if self._servo_mode_enabled:
                try:
                    self._check_sdk_result(self._sdk.servo_move_enable(False), "servo_move_enable(False)")
                except Exception as exc:
                    first_error = first_error or exc
                self._servo_mode_enabled = False
            if self._logged_in:
                try:
                    self._check_sdk_result(self._sdk.logout(), "logout")
                except Exception as exc:
                    first_error = first_error or exc
                self._logged_in = False

        self._sdk = None
        self._connected = False
        self._command_position_deg = None
        self._last_command_time = None
        self._command_velocity_deg_s = [0.0] * len(JAKA_ZU7_JOINTS)
        if first_error is not None:
            raise RuntimeError(f"Error while disconnecting {self}") from first_error

    def _read_joints_deg(self) -> list[float]:
        result = self._require_sdk().get_joint_position()
        self._check_sdk_result(result, "get_joint_position")
        if len(result) < 2 or len(result[1]) != len(JAKA_ZU7_JOINTS):
            raise RuntimeError("JAKA get_joint_position returned an unexpected joint vector")
        joints_rad = [float(value) for value in result[1]]
        if not all(math.isfinite(value) for value in joints_rad):
            raise RuntimeError("JAKA get_joint_position returned non-finite joint values")
        joints_deg = [math.degrees(value) for value in joints_rad]
        return [
            (value - offset) / sign
            for sign, offset, value in zip(
                self.config.joint_signs, self.config.joint_offsets_deg, joints_deg, strict=True
            )
        ]

    def _require_sdk(self) -> Any:
        if self._sdk is None:
            raise RuntimeError("JAKA SDK is not connected")
        return self._sdk

    @staticmethod
    def _check_sdk_result(result: Any, operation: str) -> None:
        if not isinstance(result, tuple | list) or not result or result[0] != 0:
            code = result[0] if isinstance(result, tuple | list) and result else result
            raise RuntimeError(f"JAKA SDK call {operation} failed with return value {code!r}")
