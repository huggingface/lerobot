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

import logging
from typing import Any

from ..remote_teleoperator import RemoteTeleoperator
from .config_so100_leader import SO100RemoteLeaderConfig

logger = logging.getLogger(__name__)


class SO100RemoteLeader(RemoteTeleoperator):
    """
    Remote SO-100 Leader Arm via WebRTC.

    This teleoperator enables receiving SO-100 leader arm actions via WebRTC data channels,
    allowing operators to control SO-100 follower robots over the internet with low latency.
    """

    config_class = SO100RemoteLeaderConfig
    name = "so100_remote_leader"

    def __init__(self, config: SO100RemoteLeaderConfig):
        super().__init__(config)
        self.config = config

    @property
    def action_features(self) -> dict[str, type]:
        """
        Return the same action features as the SO100Leader to maintain compatibility.
        """
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """
        Return empty feedback features as SO100Leader doesn't implement feedback.
        """
        return {}

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the LiveKit server.

        Args:
            calibrate (bool): Ignored for remote teleoperator
        """
        super().connect(calibrate=calibrate)

    @property
    def is_calibrated(self) -> bool:
        """
        Remote teleoperators don't need calibration - always return True.
        """
        return True

    def calibrate(self) -> None:
        """
        No-op for remote teleoperator.
        """
        pass

    def configure(self) -> None:
        """
        No-op for remote teleoperator.
        """
        pass

    def setup_motors(self) -> None:
        """
        No-op for remote teleoperator.
        """
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Retrieve the current action from the remote teleoperator.

        Returns:
            dict[str, Any]: A flat dictionary representing the teleoperator's current actions.
        """
        return super().get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        No-op for SO100 remote leader as feedback is not implemented.
        """
        pass

    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server.
        """
        super().disconnect()
