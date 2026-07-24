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

"""Shared definitions for the custom seven-axis SO-102 implementation.

SO-101 대비 변경점:
1. ``wrist_yaw``를 추가하여 arm 6축 + gripper, 총 7개 모터를 사용한다.
2. gripper Action을 Feetech 0..100 대신 B601과 공통인 0..-270°로 노출한다.
3. SO-102 표준 calibration과 별도로 저장한 B601 zero pose를 사용해
   SO-102 Leader의 min/zero/max를 B601 관절 범위로 변환할 수 있다.
"""

from lerobot.action_mapping import ActionMappingProfile, JointActionMapping

# SO-101은 wrist_yaw가 없는 5 arm joints + gripper 구조다.
# SO-102는 ID 5에 wrist_yaw를 추가하고 wrist_roll/gripper를 ID 6/7로 이동했다.
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = JOINT_NAMES[:-1]
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)

DEFAULT_MOTOR_IDS = {name: index for index, name in enumerate(JOINT_NAMES, start=1)}

# SO-101 계열의 Feetech gripper 내부 단위는 0..100이다.
# SO-102의 외부 Action contract는 B601과 바로 호환되도록 0..-270°를 사용한다.
GRIPPER_NATIVE_CLOSED = 0.0
GRIPPER_NATIVE_OPEN = 100.0
GRIPPER_ACTION_CLOSED = 0.0
GRIPPER_ACTION_OPEN = -270.0

# Feetech Leader와 B601의 관절 증가 방향 차이. 사용자가 CLI에서 입력하지 않고
# ``robot.type=rebot_b601_follower`` 조합일 때만 내부적으로 사용한다.
B601_JOINT_DIRECTIONS = {
    "shoulder_pan": -1,
    "shoulder_lift": -1,
    "elbow_flex": 1,
    "wrist_flex": 1,
    "wrist_yaw": 1,
    "wrist_roll": -1,
    "gripper": 1,
}
B601_ZERO_ENDPOINT_TOLERANCE = 1.0


def validate_motor_ids(motor_ids: dict[str, int]) -> None:
    expected = set(JOINT_NAMES)
    if set(motor_ids) != expected:
        raise ValueError(f"motor_ids keys must be exactly {sorted(expected)}")

    if len(set(motor_ids.values())) != len(JOINT_NAMES):
        raise ValueError("motor_ids must be unique")
    if any(motor_id < 1 or motor_id > 253 for motor_id in motor_ids.values()):
        raise ValueError("motor_ids must be in the Feetech range 1..253")


def native_gripper_to_action(native_position: float) -> float:
    """Convert Feetech's normalized 0..100 gripper value to reBot-compatible degrees."""
    native_span = GRIPPER_NATIVE_OPEN - GRIPPER_NATIVE_CLOSED
    action_span = GRIPPER_ACTION_OPEN - GRIPPER_ACTION_CLOSED
    return GRIPPER_ACTION_CLOSED + (native_position - GRIPPER_NATIVE_CLOSED) * action_span / native_span


def action_gripper_to_native(action_position: float) -> float:
    """Convert reBot-compatible gripper degrees to Feetech's normalized 0..100 value."""
    action_span = GRIPPER_ACTION_OPEN - GRIPPER_ACTION_CLOSED
    native_span = GRIPPER_NATIVE_OPEN - GRIPPER_NATIVE_CLOSED
    return GRIPPER_NATIVE_CLOSED + (action_position - GRIPPER_ACTION_CLOSED) * native_span / action_span


def native_to_action_position(motor: str, native_position: float) -> float:
    return native_gripper_to_action(native_position) if motor == "gripper" else native_position


def action_to_native_position(motor: str, action_position: float) -> float:
    return action_gripper_to_native(action_position) if motor == "gripper" else action_position


def build_b601_action_conversion(
    calibration,
    motors,
    model_resolution_table: dict[str, int],
    joint_limits: dict[str, tuple[float, float]],
    zero_positions: dict[str, float],
) -> ActionMappingProfile:
    """Build an in-memory SO-102 min/zero/max to B601 joint-range conversion.

    SO-101 Leader/Follower는 같은 Feetech 좌표계를 사용하므로 calibrated degree를
    직접 전달한다. B601은 motor zero와 관절 범위가 다르므로 별도로 저장한
    ``zero_positions``를 0° anchor로 사용하고 양쪽 calibration endpoint를
    B601의 joint limits에 대응시킨다.
    """
    if set(calibration) != set(JOINT_NAMES):
        raise ValueError(f"SO-102 calibration joints must be exactly {sorted(JOINT_NAMES)}")
    if set(joint_limits) != set(JOINT_NAMES):
        raise ValueError(f"B601 joint limits must be exactly {sorted(JOINT_NAMES)}")
    if set(zero_positions) != set(JOINT_NAMES):
        raise ValueError(f"SO-102 zero-pose joints must be exactly {sorted(JOINT_NAMES)}")

    mappings: dict[str, JointActionMapping] = {}
    for joint in JOINT_NAMES:
        target_min, target_max = joint_limits[joint]
        source_zero = float(zero_positions[joint])
        
        #wrist roll은 SO102 표준 calibration 범위 전체를 유지하되, B601 텔레오퍼레이션에서는 zero 기준 ±90°만 사용한다.
        """ 
        if joint == "gripper":
            source_low, source_high = GRIPPER_ACTION_OPEN, GRIPPER_ACTION_CLOSED
        else:
            motor = motors[joint]
            max_resolution = model_resolution_table[motor.model] - 1
            source_half_span = (
                calibration[joint].range_max - calibration[joint].range_min
            ) * 180 / max_resolution
            if source_half_span <= 0:
                raise ValueError(f"Invalid SO-102 calibration range for '{joint}'")
            source_low, source_high = -source_half_span, source_half_span
        """
        
        if joint == "gripper":
            source_low, source_high = GRIPPER_ACTION_OPEN, GRIPPER_ACTION_CLOSED

        elif joint == "wrist_roll":
            # SO102 표준 calibration은 전체 회전 범위를 유지하되,
            # B601 텔레오퍼레이션에서는 zero 기준 ±90°만 사용한다.
            source_low = -90.0
            source_high = 90.0

        else:
            motor = motors[joint]
            max_resolution = model_resolution_table[motor.model] - 1
            source_half_span = (
                calibration[joint].range_max - calibration[joint].range_min
            ) * 180 / max_resolution

            if source_half_span <= 0:
                raise ValueError(f"Invalid SO-102 calibration range for '{joint}'")

            source_low = -source_half_span
            source_high = source_half_span
        if not source_low <= source_zero <= source_high:
            raise ValueError(
                f"SO-102 zero pose for '{joint}' ({source_zero:.3f}°) is outside "
                f"its calibrated range {source_low:.3f}°..{source_high:.3f}°"
            )

        if B601_JOINT_DIRECTIONS[joint] == 1:
            source_for_target_min = source_low
            source_for_target_max = source_high
        else:
            source_for_target_min = source_high
            source_for_target_max = source_low

        # B601 uses 0° as its zero pose. A configured limit within 1° of zero
        # (for example shoulder_lift/elbow_flex max=1°) is the same physical
        # endpoint for mapping purposes, so source_zero anchors that side.
        if abs(target_min) <= B601_ZERO_ENDPOINT_TOLERANCE:
            source_for_target_min = None
        if abs(target_max) <= B601_ZERO_ENDPOINT_TOLERANCE:
            source_for_target_max = None

        mappings[joint] = JointActionMapping(
            source_zero=source_zero,
            source_for_target_min=source_for_target_min,
            source_for_target_max=source_for_target_max,
            target_min=float(target_min),
            target_zero=0.0,
            target_max=float(target_max),
        )

    return ActionMappingProfile(joints=mappings)
