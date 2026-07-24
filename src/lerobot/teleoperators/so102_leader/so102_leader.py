#!/usr/bin/env python

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

import json
import logging
import math
import time
from typing import Any

from lerobot.action_mapping import ActionMappingProfile
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.so102 import (
    ACTION_KEYS,
    GRIPPER_ACTION_CLOSED,
    GRIPPER_ACTION_OPEN,
    JOINT_NAMES,
    native_to_action_position,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_so102_leader import SO102LeaderConfig

logger = logging.getLogger(__name__)


class SO102Leader(Teleoperator):
    """Seven-actuator Feetech leader derived from the SO-101 leader.

    SO-101 대비 변경점:
    - ``wrist_yaw``를 추가한 7개 모터를 읽는다.
    - gripper 0..100을 공통 Action 0..-270°로 변환한다.
    - 표준 midpoint/min/max calibration은 SO-102 Follower용으로 유지한다.
    - B601 0°에 대응하는 Leader pose는 별도 sidecar 파일에 저장한다.
    - B601 Follower가 선택된 경우에만 min/zero/max Action 변환을 적용한다.
    """

    config_class = SO102LeaderConfig
    name = "so102_leader"

    def __init__(self, config: SO102LeaderConfig):
        super().__init__(config)
        self.config = config
        self.b601_action_conversion: ActionMappingProfile | None = None

        # SO-101에는 없는 B601 전용 영점이다. 표준 motor calibration JSON 형식을
        # 변경하지 않기 위해 같은 calibration 폴더의 sidecar 파일로 분리한다.
        self.b601_zero_pose_fpath = self.calibration_dir / f"{self.id}.b601_zero_pose.json"
        self.b601_zero_positions: dict[str, float] = {}
        if self.b601_zero_pose_fpath.is_file():
            payload = json.loads(self.b601_zero_pose_fpath.read_text())
            if set(payload) != set(JOINT_NAMES):
                raise ValueError(
                    f"SO-102 B601 zero-pose joints in {self.b601_zero_pose_fpath} "
                    f"must be exactly {sorted(JOINT_NAMES)}"
                )
            self.b601_zero_positions = {joint: float(payload[joint]) for joint in JOINT_NAMES}
            if any(not math.isfinite(value) for value in self.b601_zero_positions.values()):
                raise ValueError(f"SO-102 B601 zero-pose values in {self.b601_zero_pose_fpath} must be finite")
        self.bus = FeetechMotorsBus(
            port=config.port,
            # SO-101의 고정 6모터 사전 대신 공통 JOINT_NAMES의 7개 모터를 만든다.
            motors={
                motor: Motor(
                    config.motor_ids[motor],
                    "sts3215",
                    MotorNormMode.RANGE_0_100 if motor == "gripper" else MotorNormMode.DEGREES,
                )
                for motor in JOINT_NAMES
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        # This leader is intentionally read-only during teleoperation.
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no "
                "calibration file found"
            )
            self.calibrate()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # SO-101 표준 calibration에는 없는 첫 단계:
        # B601의 motor zero(모든 관절 0°)에 대응하는 Leader 자세를 먼저 기록한다.
        # 이후 homing offset을 바꾸더라도 복원할 수 있도록 Actual_Position 좌표로 보관한다.
        input(
            f"Move {self} to the zero pose that should match B601 0° on every joint, "
            "close the gripper, and press ENTER...."
        )
        zero_present_positions = self.bus.sync_read("Present_Position", normalize=False)
        current_homing_offsets = {
            motor: self.bus.read("Homing_Offset", motor, normalize=False) for motor in self.bus.motors
        }
        zero_actual_positions = {
            motor: zero_present_positions[motor] + current_homing_offsets[motor]
            for motor in self.bus.motors
        }

        # 이 시점부터는 SO-101과 동일한 midpoint + min/max calibration이다.
        # 이 표준 결과는 SO-102 Leader → SO-102 Follower 텔레에 그대로 사용된다.
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their entire ranges of "
            "motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {
            motor: MotorCalibration(
                id=definition.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
            for motor, definition in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)

        # 처음 기록한 B601 zero pose를 최종 homing/min/max가 적용된 degree 좌표로
        # 다시 계산한다. 따라서 두 영점은 서로 덮어쓰지 않고 독립적으로 유지된다.
        zero_positions: dict[str, float] = {}
        for motor, definition in self.bus.motors.items():
            calibration = self.calibration[motor]
            zero_present = zero_actual_positions[motor] - homing_offsets[motor]
            if motor == "gripper":
                bounded_zero = min(calibration.range_max, max(calibration.range_min, zero_present))
                native_zero = (
                    (bounded_zero - calibration.range_min)
                    / (calibration.range_max - calibration.range_min)
                    * 100
                )
                zero_positions[motor] = native_to_action_position(motor, native_zero)
            else:
                midpoint = (calibration.range_min + calibration.range_max) / 2
                max_resolution = self.bus.model_resolution_table[definition.model] - 1
                zero_positions[motor] = (zero_present - midpoint) * 360 / max_resolution

        self.b601_zero_positions = zero_positions
        self._save_calibration()
        temporary_zero_path = self.b601_zero_pose_fpath.with_suffix(
            f"{self.b601_zero_pose_fpath.suffix}.tmp"
        )
        temporary_zero_path.write_text(
            json.dumps(self.b601_zero_positions, indent=2, sort_keys=True) + "\n"
        )
        temporary_zero_path.replace(self.b601_zero_pose_fpath)
        print(f"Calibration saved to {self.calibration_fpath}")
        print(f"B601 zero pose saved to {self.b601_zero_pose_fpath}")

    def configure(self) -> None:
        # The leader must remain back-drivable. No torque is enabled anywhere in this class.
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        native_positions = self.bus.sync_read("Present_Position")
        action = {}
        for motor in JOINT_NAMES:
            action_position = native_to_action_position(motor, float(native_positions[motor]))
            action[f"{motor}.pos"] = action_position

        # 기본값(None)이면 SO-102 Follower용 calibrated degree를 그대로 반환한다.
        # B601 Follower 조합일 때만 lerobot-teleoperate가 이 mapping을 주입한다.
        if self.b601_action_conversion is not None:
            action = self.b601_action_conversion.map_action(action)
        logger.debug(f"{self} read action: {(time.perf_counter() - start) * 1e3:.1f}ms")
        return action

    def set_b601_action_conversion(self, profile: ActionMappingProfile) -> None:
        if set(profile.joints) != set(JOINT_NAMES):
            raise ValueError(f"B601 Action conversion joints must be exactly {sorted(JOINT_NAMES)}")
        if (
            profile.joints["gripper"].map(GRIPPER_ACTION_CLOSED) != GRIPPER_ACTION_CLOSED
            or profile.joints["gripper"].map(GRIPPER_ACTION_OPEN) != GRIPPER_ACTION_OPEN
        ):
            raise ValueError(
                "B601 gripper mapping must preserve 0° closed and -270° open; "
                "the Feetech 0..100 conversion has already been applied"
            )
        self.b601_action_conversion = profile

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Deliberately ignore feedback to guarantee that teleoperation never commands the leader.
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(disable_torque=True)
        logger.info(f"{self} disconnected.")
