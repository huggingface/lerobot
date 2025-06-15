#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import time

import logging
import os
import sys
import time
from typing import Any

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_pose_estimation import PoseEstimationConfig

logger = logging.getLogger(__name__)


class PoseEstimation(Teleoperator):
    """
    Pose Estimation Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = PoseEstimationConfig
    name = "pose_estimation"

    def __init__(self, config: PoseEstimationConfig):
        super().__init__(config)
        self.config = config
        self.pose = None

    @property
    def action_features(self) -> dict[str, type]:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, \
                          "delta_pitch": 4, "delta_yaw": 5, 
                          # "delta_gripper": 6
                          }
            }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        # call the pose estimation model to get the action
        self.pose = ...
        if self.pose is None:
            raise DeviceNotConnectedError("Pose estimation model is not initialized.")
        
        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "delta_roll": delta_roll,
            "delta_pitch": delta_pitch,
            "delta_yaw": delta_yaw,
            # "delta_gripper": delta_gripper
        }

        return action_dict