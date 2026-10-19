import json
import logging
import math
import socket
import time
from functools import cached_property

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_jaka_zu7_follower import JAKAZu7FollowerConfig

logger = logging.getLogger(__name__)

# Keep these keys identical to the six arm joints in this project's SO-ARM101 leader.
# They map 1:1 to JAKA J1..J6 (signs/offsets are applied in send_action/_read_joints_deg).
JAKA_ZU7_JOINTS = (
    "shoulder_pan",
    "heavy_joint",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

_SOCKET_TIMEOUT_S = 5.0
# Software velocity/acceleration limiter dt ceiling: matches one 60 Hz teleop cycle.
# Prevents a large step when send_action has not been called for a while.
_CONTROL_DT_MAX_S = 0.016


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
    """LeRobot follower driver for the six-axis JAKA Zu 7, via the bridge server.

    The JAKA ``jkrc`` SDK only works on Python <= 3.10 while LeRobot requires >= 3.12,
    so this class does NOT import ``jkrc``. Instead it is a socket client to
    ``jaka_bridge_server.py`` (a Python 3.10 process that owns the SDK). Positions are
    expressed in **degrees** at the LeRobot boundary; the bridge server converts
    deg <-> rad at the SDK boundary.
    """

    config_class = JAKAZu7FollowerConfig
    name = "jaka_zu7_follower"

    def __init__(self, config: JAKAZu7FollowerConfig):
        super().__init__(config)
        self.config = config
        self._sock: socket.socket | None = None
        self._recv_file = None
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
            self._sock = socket.create_connection(
                (self.config.bridge_host, self.config.bridge_port), timeout=_SOCKET_TIMEOUT_S
            )
            self._sock.settimeout(_SOCKET_TIMEOUT_S)
            self._recv_file = self._sock.makefile("rb")
            # Read the initial position once: confirms the bridge is live and seeds
            # the software velocity limiter.
            self._command_position_deg = self._read_joints_deg()
            self._last_command_time = time.monotonic()
            self._connected = True

            for camera in self.cameras.values():
                camera.connect()

            logger.info(
                "%s connected to JAKA bridge at %s:%s",
                self, self.config.bridge_host, self.config.bridge_port,
            )
        except Exception:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Cleanup failed after JAKA bridge connection error")
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
        if self._last_command_time is None:
            dt = _CONTROL_DT_MAX_S
        else:
            dt = max(0.001, min(now - self._last_command_time, _CONTROL_DT_MAX_S))

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

        self._request({"cmd": "set", "joints": controller_command_deg})

        self._command_position_deg = command_deg
        self._command_velocity_deg_s = command_velocity_deg_s
        self._last_command_time = now

        if "gripper.pos" in action:
            logger.debug("Ignoring SO-ARM101 leader gripper action; no JAKA gripper mapping")

        return {f"{joint}.pos": value for joint, value in zip(JAKA_ZU7_JOINTS, command_deg, strict=True)}

    def disconnect(self) -> None:
        first_error: Exception | None = None
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                first_error = first_error or exc

        if self._sock is not None and self._recv_file is not None:
            try:
                self._request({"cmd": "exit"})
            except Exception as exc:
                first_error = first_error or exc

        if self._recv_file is not None:
            try:
                self._recv_file.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass

        self._sock = None
        self._recv_file = None
        self._connected = False
        self._command_position_deg = None
        self._last_command_time = None
        self._command_velocity_deg_s = [0.0] * len(JAKA_ZU7_JOINTS)
        if first_error is not None:
            raise RuntimeError(f"Error while disconnecting {self}") from first_error

    def _read_joints_deg(self) -> list[float]:
        resp = self._request({"cmd": "get"})
        joints_deg = resp.get("joints")
        if not isinstance(joints_deg, (list, tuple)) or len(joints_deg) != len(JAKA_ZU7_JOINTS):
            raise RuntimeError("JAKA bridge returned an unexpected joint vector")
        joints_deg = [float(value) for value in joints_deg]
        if not all(math.isfinite(value) for value in joints_deg):
            raise RuntimeError("JAKA bridge returned non-finite joint values")
        return [
            (value - offset) / sign
            for sign, offset, value in zip(
                self.config.joint_signs, self.config.joint_offsets_deg, joints_deg, strict=True
            )
        ]

    def _request(self, req: dict) -> dict:
        sock = self._require_sock()
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        line = self._recv_file.readline()
        if not line:
            raise RuntimeError("JAKA bridge connection closed by server")
        resp = json.loads(line.decode("utf-8"))
        if resp.get("status") == "error":
            raise RuntimeError(f"JAKA bridge error: {resp.get('error')}")
        return resp

    def _require_sock(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("JAKA bridge is not connected")
        return self._sock
