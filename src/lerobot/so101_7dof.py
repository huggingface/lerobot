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

GRIPPER_NATIVE_CLOSED = 0.0
GRIPPER_NATIVE_OPEN = 100.0
GRIPPER_ACTION_CLOSED = 0.0
GRIPPER_ACTION_OPEN = -270.0


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
