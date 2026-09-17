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

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _tron2_env_available, require_package

from ..robot import Robot
from .config_limx_tron2 import LimxTron2RobotConfig

if TYPE_CHECKING or _tron2_env_available:
    from tron2_env import Tron2Config, create_motion_controller
    from tron2_env.bridge import BridgeConfig, BridgeObservationProvider
    from tron2_env.motion import MotionController
else:
    BridgeConfig = Any
    BridgeObservationProvider = Any
    MotionController = Any


LIMX_TRON2_JOINTS = (
    "left_arm_joint_1.pos",
    "left_arm_joint_2.pos",
    "left_arm_joint_3.pos",
    "left_arm_joint_4.pos",
    "left_arm_joint_5.pos",
    "left_arm_joint_6.pos",
    "left_arm_joint_7.pos",
    "left_gripper.pos",
    "right_arm_joint_1.pos",
    "right_arm_joint_2.pos",
    "right_arm_joint_3.pos",
    "right_arm_joint_4.pos",
    "right_arm_joint_5.pos",
    "right_arm_joint_6.pos",
    "right_arm_joint_7.pos",
    "right_gripper.pos",
    "head_pitch.pos",
    "head_yaw.pos",
)
LIMX_TRON2_ACTIONS = LIMX_TRON2_JOINTS[:16]
BRIDGE_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

_LEFT_ARM = slice(0, 7)
_LEFT_GRIPPER = 7
_RIGHT_ARM = slice(8, 15)
_RIGHT_GRIPPER = 15


class LimxTron2Robot(Robot):
    config_class = LimxTron2RobotConfig
    name = "limx_tron2"

    def __init__(self, config: LimxTron2RobotConfig):
        require_package("tron2_env", extra="tron2", import_name="tron2_env")
        super().__init__(config)
        self.config = config
        self.controller: MotionController | None = None
        self.bridge_provider: BridgeObservationProvider | None = None
        self._bridge_initial_observation: dict[str, Any] | None = None
        self._bridge_ready = False
        self.cameras = (
            {} if config.observation_source == "bridge" else make_cameras_from_configs(config.cameras)
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(LIMX_TRON2_JOINTS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        if self.config.observation_source == "bridge":
            shape = (self.config.bridge_camera_height, self.config.bridge_camera_width, 3)
            return dict.fromkeys(BRIDGE_CAMERAS, shape)
        features = {}
        for name, config in self.config.cameras.items():
            if config.height is None or config.width is None:
                raise ValueError(f"Camera {name!r} must define width and height")
            features[name] = (config.height, config.width, 3)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(LIMX_TRON2_ACTIONS, float)

    @property
    def is_connected(self) -> bool:
        return (
            self.controller is not None
            and self.controller.is_connected()
            and (self.config.observation_source != "bridge" or self._bridge_ready)
            and all(camera.is_connected for camera in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")

        sdk_config = Tron2Config(
            robot_ip=self.config.robot_ip,
            port=self.config.port,
            init_joints=self.config.init_joints,
            init_head=self.config.init_head,
            state_queue_maxlen=self.config.state_queue_maxlen,
            polling_rate=self.config.polling_rate,
            connection_timeout=self.config.connection_timeout,
        )
        self.controller = create_motion_controller(
            sdk_config,
            publish_rate=self.config.publish_rate,
            eta_default=1.0 / self.config.control_frequency,
        )
        try:
            if self.config.observation_source == "bridge":
                bridge_config = BridgeConfig(
                    host=self.config.bridge_host,
                    ws_path=self.config.bridge_ws_path,
                    image_max_fps=self.config.bridge_image_max_fps,
                    align_max_delay_ms=self.config.bridge_align_max_delay_ms,
                    verify_tls=self.config.bridge_verify_tls,
                    save_debug_images=False,
                )
                bridge_provider = BridgeObservationProvider(bridge_config)
                self.bridge_provider = bridge_provider
                bridge_provider.start()
                self._bridge_initial_observation = bridge_provider.get_obs(
                    timeout=self.config.bridge_connection_timeout
                )
                self._bridge_ready = True
            else:
                for camera in self.cameras.values():
                    camera.connect()
            self.configure()
        except Exception:
            self.disconnect()
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected or self.controller is None:
            raise ConnectionError(f"{self} is not connected")

        bridge_images: dict[str, np.ndarray] = {}
        if self.bridge_provider is not None:
            bridge_observation = self._bridge_initial_observation
            self._bridge_initial_observation = None
            if bridge_observation is None:
                bridge_observation = self.bridge_provider.get_obs(timeout=self.config.bridge_timeout)
            state = np.asarray(bridge_observation["state"], dtype=np.float64)
            bridge_images = bridge_observation.get("images", {})
        else:
            state = np.asarray(
                self.controller.get_joint_states(timeout=self.config.state_timeout)["states"],
                dtype=np.float64,
            )
        if state.shape != (len(LIMX_TRON2_JOINTS),):
            raise RuntimeError(
                f"Expected {len(LIMX_TRON2_JOINTS)} TRON2 state values, received shape {state.shape}"
            )

        observation: RobotObservation = dict(zip(LIMX_TRON2_JOINTS, state, strict=True))
        if self.bridge_provider is not None:
            missing_cameras = set(BRIDGE_CAMERAS).difference(bridge_images)
            if missing_cameras:
                raise RuntimeError(f"Bridge observation is missing cameras: {sorted(missing_cameras)}")
            observation.update({name: bridge_images[name] for name in BRIDGE_CAMERAS})
        else:
            for name, camera in self.cameras.items():
                observation[name] = camera.read_latest()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected or self.controller is None:
            raise ConnectionError(f"{self} is not connected")

        missing = set(LIMX_TRON2_ACTIONS).difference(action)
        unexpected = set(action).difference(LIMX_TRON2_ACTIONS)
        if missing or unexpected:
            raise ValueError(
                f"TRON2 action keys do not match action_features; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        target = np.asarray([action[name] for name in LIMX_TRON2_ACTIONS], dtype=np.float64)
        head_position = np.asarray(self.config.init_head, dtype=np.float64)
        if head_position.shape != (2,):
            raise RuntimeError(f"Expected 2 TRON2 head values, received shape {head_position.shape}")
        servo_target = np.concatenate((target[_LEFT_ARM], target[_RIGHT_ARM], head_position))
        grippers = np.clip(
            target[[_LEFT_GRIPPER, _RIGHT_GRIPPER]] * 100.0,
            0.0,
            100.0,
        )
        self.controller.set_gripper(float(grippers[0]), float(grippers[1]))
        self.controller.command_joints(servo_target)
        return dict(zip(LIMX_TRON2_ACTIONS, target, strict=True))

    def disconnect(self) -> None:
        self._bridge_ready = False
        self._bridge_initial_observation = None
        if self.bridge_provider is not None:
            self.bridge_provider.stop()
            self.bridge_provider = None
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self.controller is not None:
            self.controller.disconnect()
            self.controller = None
