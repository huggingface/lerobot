#!/usr/bin/env python

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

import time

import numpy as np
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot_params import RobotParams

from lerobot.common.errors import DeviceAlreadyConnectedError

from ..teleoperator import Teleoperator
from .configuration_stretch3 import Stretch3GamePadConfig

# from stretch_body.gamepad_controller.GamePadController
GAMEPAD_BUTTONS = [
    "middle_led_ring_button_pressed",
    "left_stick_x",
    "left_stick_y",
    "right_stick_x",
    "right_stick_y",
    "left_stick_button_pressed",
    "right_stick_button_pressed",
    "bottom_button_pressed",
    "top_button_pressed",
    "left_button_pressed",
    "right_button_pressed",
    "left_shoulder_button_pressed",
    "right_shoulder_button_pressed",
    "select_button_pressed",
    "start_button_pressed",
    "left_trigger_pulled",
    "right_trigger_pulled",
    "bottom_pad_pressed",
    "top_pad_pressed",
    "left_pad_pressed",
    "right_pad_pressed",
]


class Stretch3GamePad(Teleoperator):
    """[Stretch 3](https://hello-robot.com/stretch-3-product), by Hello Robot."""

    config_class = Stretch3GamePadConfig
    name = "stretch3"

    def __init__(self, config: Stretch3GamePadConfig):
        raise NotImplementedError
        super().__init__(config)

        self.config = config
        self.robot_type = self.config.type

        self.api = GamePadTeleop(robot_instance=False)

        self.is_connected = False
        self.logs = {}

        # TODO(aliberts): test this
        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(GAMEPAD_BUTTONS),),
            "names": {"buttons": GAMEPAD_BUTTONS},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        self.api.startup()
        self.api._update_state()  # Check controller can be read & written
        self.api._update_modes()
        self.is_connected = True

    def calibrate(self) -> None:
        pass

    def get_action(self) -> np.ndarray:
        # Read Stretch state
        before_read_t = time.perf_counter()
        action = self.api.gamepad_controller.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        action = np.asarray(list(action.values()))

        return action

    def send_feedback(self, feedback: np.ndarray) -> None:
        pass

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        self.api.stop()
        self.is_connected = False
