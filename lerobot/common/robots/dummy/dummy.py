# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


import logging

import numpy as np

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .configuration_dummy import DummyConfig


class Dummy(Robot):
    config_class = DummyConfig
    name = "dummy"

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.is_connected = False

    @property
    def state_feature(self) -> dict:
        logging.warning("Dummy has nothing to send.")

    @property
    def action_feature(self) -> dict:
        logging.warning("Dummy has nothing to send.")

    @property
    def camera_features(self) -> dict[str, dict]:
        cam_ft = {
            "cam": {
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            },
        }
        return cam_ft

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Dummy is already connected. Do not run `robot.connect()` twice."
            )

        logging.info("Connecting cameras.")
        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True

    def calibrate(self) -> None:
        logging.warning("Dummy has nothing to calibrate.")
        return

    def get_observation(self) -> dict[str, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError("Dummy is not connected. You need to run `robot.connect()`.")

        obs_dict = {}

        for cam_key, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[f"{OBS_STATE}.{cam_key}"] = frame

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:
        logging.warning("Dummy has nothing to send.")

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Dummy is not connected. You need to run `robot.connect()` before disconnecting."
            )
        for cam in self.cameras.values():
            cam.disconnect()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
