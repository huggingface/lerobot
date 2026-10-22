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
from functools import cached_property
from math import degrees, isfinite, radians
from typing import TYPE_CHECKING

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _menlo_sdk_available, require_package

from ..robot import Robot
from .config_asimov import AsimovConfig

if TYPE_CHECKING or _menlo_sdk_available:
    import menlo.asimov as menlo
else:
    menlo = None

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


class Asimov(Robot):
    """Asimov 1 with onboard balancing and bounded velocity commands.

    Connect only observes. An operator must put the robot in MOVE using Asimov
    Manager before sending actions. Joint positions are observations in radians;
    locomotion actions follow LeRobot's mobile-base convention (m/s, m/s, deg/s).
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
        # Validate camera names before any resources are opened.
        _ = self.observation_features

    @cached_property
    def action_features(self) -> dict[str, type]:
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

    def stop(self) -> None:
        """End our velocity hold, keeping onboard balancing active."""
        if self._client is not None and self._command_sent:
            # An operator may already have changed mode. Do not re-enter MOVE.
            if self._client.get_state().mode == menlo.Mode.MOVE:
                self._client.balance(wait=False, timeout=0.0)
            self._command_sent = False

    def disconnect(self) -> None:
        """The SDK sends zero before closing; never STAND or DAMP a balancing biped."""
        errors: list[Exception] = []
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
