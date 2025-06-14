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

from typing import Any
from lerobot.common.teleoperators.teleoperator import Teleoperator
from .configuration_spes_teleop import SpesTeleopConfig
from .spes_utils import Teleop as PhoneTeleop
from enum import IntEnum


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}

class SpesTeleop(Teleoperator):
    config_class = SpesTeleopConfig
    name = "spes_teleop"

    def __init__(self, config: SpesTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._calibrated = True
        self._pose: dict = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }

        self._server = PhoneTeleop(host=config.host, port=int(config.port))
        self._server.subscribe(self._on_pose_update)
        self._gripper_state = GripperAction.STAY.value

    def _on_pose_update(self, pose: dict, msg: dict):
        self._pose = pose
        print(f"Pose: {pose}")
    
    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "roll": 3,
                    "pitch": 4,
                    "yaw": 5,
                    "gripper": 6,
                },
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "roll": 3,
                    "pitch": 4,
                    "yaw": 5,
                },
            }

    @property
    def feedback_features(self) -> dict:
        return {}  # No feedback expected from phone teleop

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True
        # self._server.run()
        # Start server in background
        import threading
        threading.Thread(target=self._server.run, daemon=True).start()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        action = self._pose.copy()
        if self.config.use_gripper:
            action["gripper"] = self._gripper_state
        return action


    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
        self._server.stop()
