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

"""Keyboard input mapped to planar base actions before device composition."""

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor.keyboard_base_processor import (
    KeyboardToBaseActionProcessorStep,
    make_keyboard_base_processor,
)

from .configuration_keyboard import KeyboardBaseTeleopConfig
from .teleop_keyboard import KeyboardTeleop


class KeyboardBaseTeleop(KeyboardTeleop):
    """Produce x.vel, y.vel (m/s), and theta.vel (deg/s) without consulting a robot."""

    config_class = KeyboardBaseTeleopConfig
    name = "keyboard_base"

    def __init__(self, config: KeyboardBaseTeleopConfig) -> None:
        super().__init__(config)
        self.action_processor = make_keyboard_base_processor(
            KeyboardToBaseActionProcessorStep(
                key_bindings=dict(config.key_bindings),
                linear_speeds=config.linear_speeds,
                angular_speeds=config.angular_speeds,
            )
        )
        self._observation: RobotObservation = {}

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(("x.vel", "y.vel", "theta.vel"), float)

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self.current_pressed.clear()
        while not self.event_queue.empty():
            self.event_queue.get_nowait()
        self.action_processor.reset()
        self._observation = {}

    def on_observation(self, observation: RobotObservation) -> None:
        self._observation = observation

    def get_action(self, observation: RobotObservation | None = None) -> RobotAction:
        return self.action_processor(
            (
                super().get_action(observation=observation),
                observation if observation is not None else self._observation,
            )
        )
