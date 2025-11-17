# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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

from functools import cached_property
import logging
import threading

from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.config_bi_sourccey_leader import BiSourcceyLeaderConfig
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.config_sourccey_leader import SourcceyLeaderConfig
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.sourccey_leader import SourcceyLeader
from lerobot.teleoperators.teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class BiSourcceyLeader(Teleoperator):
    """
    [Bimanual Sourccey Leader Arms] designed by Vulcan
    """

    config_class = BiSourcceyLeaderConfig
    name = "bi_sourccey_leader"

    def __init__(self, config: BiSourcceyLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SourcceyLeaderConfig(
            id="sourccey_left",
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            orientation="left",
        )

        right_arm_config = SourcceyLeaderConfig(
            id="sourccey_right",
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            orientation="right",
        )

        self.left_arm = SourcceyLeader(left_arm_config)
        self.right_arm = SourcceyLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def auto_calibrate(self, full_reset: bool = False) -> None:
        """
        Auto-calibrate both arms simultaneously using threading.
        """
        # Create threads for each arm
        left_thread = threading.Thread(
            target=self.left_arm.auto_calibrate,
            kwargs={"reversed": False}
        )
        right_thread = threading.Thread(
            target=self.right_arm.auto_calibrate,
            kwargs={"reversed": True}
        )

        # Start both threads
        left_thread.start()
        right_thread.start()

        # Wait for both threads to complete
        left_thread.join()
        right_thread.join()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    ##################################################################################
    # Motor Configuration Functions
    ##################################################################################
    def set_baud_rate(self, baud_rate: int) -> None:
        self.left_arm.bus.set_baudrate(baud_rate)
        self.right_arm.bus.set_baudrate(baud_rate)
