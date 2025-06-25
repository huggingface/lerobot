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

from lerobot.common.teleoperators.utils import make_teleoperator_from_config

from ..teleoperator import Teleoperator
from .config_multi_arm_leader import MultiArmLeaderConfig

logger = logging.getLogger(__name__)


class MultiArmLeader(Teleoperator):
    """
    Multiple Arms Leader.

    For example, how to run the teleoperate script with multi-arm leader and follower
    being composed of two SO101 arms:
    ```bash
    export arm1="{type: so101_follower, port: /dev/ttyACM0}"
    export arm2="{type: so101_follower, port: /dev/ttyACM1}"
    export teleop1="{type: so101_leader, port: /dev/ttyACM2}"
    export teleop2="{type: so101_leader, port: /dev/ttyACM3}"

    python -m lerobot.teleoperate \
        --robot.type=multi_arm_follower \
        --robot.arms=["$arm1","$arm2"] \
        --robot.id=two-so101-follower \
        --teleop.type=multi_arm_leader \
        --teleop.arms=["$teleop1","$teleop2"] \
        --teleop.id=two-so101-leader
    ```

    """

    config_class = MultiArmLeaderConfig
    name = "multi_arm_leader"

    def __init__(self, config: MultiArmLeaderConfig):
        super().__init__(config)
        self.config = config

        self.arms = [make_teleoperator_from_config(arm_config) for arm_config in config.arms]

    def _encode_arm_index(self, key: str, index: int) -> str:
        return f"arm{index}__{key}"

    def _decode_arm_index(self, key: str) -> int:
        arm_id, *remaining = key.split("__")
        assert arm_id.startswith("arm"), (arm_id, key)
        return int(arm_id[len("arm") :]), "__".join(remaining)

    @property
    def action_features(self) -> dict[str, type]:
        # Get quickly all action_features
        # assuming minimal latency due the loop
        all_actions = [arm.action_features for arm in self.arms]
        # Post-process the results:
        all_actions = [
            {self._encode_arm_index(key, i): value for key, value in action.items()}
            for i, action in enumerate(all_actions)
        ]
        return {k: v for action_fts in all_actions for k, v in action_fts.items()}

    @property
    def feedback_features(self) -> dict[str, type]:
        # Get quickly all action_features
        # assuming minimal latency due the loop
        all_feedback_fts = [arm.feedback_features for arm in self.arms]
        # Post-process the results:
        all_feedback_fts = [
            {self._encode_arm_index(key, i): value for key, value in feedback_ft.items()}
            for i, feedback_ft in enumerate(all_feedback_fts)
        ]
        return {k: v for feedback_fts in all_feedback_fts for k, v in feedback_fts.items()}

    @property
    def is_connected(self) -> bool:
        return all(arm.is_connected for arm in self.arms)

    def connect(self, calibrate: bool = True) -> None:
        for arm in self.arms:
            arm.connect(calibrate=calibrate)

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return all(arm.is_calibrated for arm in self.arms)

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        for arm in self.arms:
            arm.calibrate()

    def configure(self) -> None:
        for arm in self.arms:
            arm.configure()

    def setup_motors(self) -> None:
        for arm in self.arms:
            arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        all_actions = [arm.get_action() for arm in self.arms]
        all_actions = [
            {self._encode_arm_index(key, i): value for key, value in actions.items()}
            for i, actions in enumerate(all_actions)
        ]
        return {k: v for actions in all_actions for k, v in actions.items()}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        for arm in self.arms:
            arm.disconnect()
        logger.info(f"{self} disconnected.")
