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

"""NexArm leader (master) teleoperator — LeRobot Teleoperator subclass.

The leader connects to the master ESP32 via USB serial. Torque is disabled so
the operator can freely drag the arm; positions are streamed to the follower
through the normalised LeRobot pipeline. Calibration uses the same
``set_half_turn_homings`` + ``record_ranges_of_motion`` flow as the follower,
which removes the need for any hard-coded mirror or remap of joint values.
"""

from __future__ import annotations

import logging
import time

from lerobot.motors import MotorCalibration
from lerobot.motors.nexarm import NexArmMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_nexarm_leader import NexArmLeaderConfig

logger = logging.getLogger(__name__)


class NexArmLeader(Teleoperator):
    """NexArm 6-DOF leader teleoperator.

    The leader runs the master ESP32 in LeRobot mode. Each tick we read
    Present_Position (normalised to DEGREES / RANGE_0_100 by the bus) and pass
    it straight through as the follower's goal. The follower's calibration
    handles per-joint sign / offset / range differences, so no mirror or
    bespoke remap is needed.
    """

    config_class = NexArmLeaderConfig
    name = "nexarm_leader"

    def __init__(self, config: NexArmLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = NexArmMotorsBus(
            port=config.port,
            calibration=self.calibration,
            baudrate=config.baudrate,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect(handshake=False)
        self.bus.enter_lerobot_mode()
        self.bus.handshake()

        if calibrate and not self.is_calibrated:
            logger.info("No calibration found, running calibration")
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to keep the existing calibration for {self.id}, or type 'c' "
                "and press ENTER to run a new one: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Keeping existing calibration")
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.reset_calibration()
        self.bus.disable_torque()

        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        self.bus.record_ranges_of_motion()

        # shoulder_lift (servo 2) on the leader is mounted mirrored relative
        # to the follower — see Nex_Arm/EspNow_Ctrl.cpp mapMasterTeachToFollower
        # which does target = 4096 - master_pos. The LeRobot bridge forwards
        # CMD 97 raw with no remap, so we compensate here by flipping the
        # leader's normalised output on that joint via drive_mode=1.
        lift = "shoulder_lift"
        cal = self.bus.calibration[lift]
        self.bus.calibration[lift] = MotorCalibration(
            id=cal.id,
            drive_mode=1,
            homing_offset=cal.homing_offset,
            range_min=cal.range_min,
            range_max=cal.range_max,
        )

        self.calibration = dict(self.bus.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        # Leader must stay torque-off so the operator can drag it.
        self.bus.disable_torque()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        positions = self.bus.sync_read("Present_Position")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return {f"{m}.pos": v for m, v in positions.items()}

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Leader has no force feedback channel.
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.bus.exit_lerobot_mode()
        except Exception:  # nosec B110
            pass
        self.bus.disconnect(disable_torque=False)
        logger.info(f"{self} disconnected.")
