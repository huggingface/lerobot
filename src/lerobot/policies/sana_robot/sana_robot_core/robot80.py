# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/data/format/robot80.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import numpy as np

ROBOT_DIM = 80

ROBOT_FIELDS = {
    "left_joints": slice(0, 7),
    "left_eef_position": slice(7, 10),
    "left_eef_rotation": slice(10, 16),
    "left_gripper": slice(16, 17),
    "left_hand": slice(17, 29),
    "right_joints": slice(29, 36),
    "right_eef_position": slice(36, 39),
    "right_eef_rotation": slice(39, 45),
    "right_gripper": slice(45, 46),
    "right_hand": slice(46, 58),
    "camera_position": slice(58, 61),
    "camera_rotation": slice(61, 67),
}

ROTATION_FIELDS = ("left_eef_rotation", "right_eef_rotation", "camera_rotation")
GRIPPER_FIELDS = ("left_gripper", "right_gripper")
PAIRED_FIELDS = (
    ("left_joints", "right_joints"),
    ("left_eef_position", "right_eef_position"),
    ("left_eef_rotation", "right_eef_rotation"),
    ("left_gripper", "right_gripper"),
    ("left_hand", "right_hand"),
)

ACTION_MODE_EEF = "robot_base_eef"
ACTION_MODE_JOINT = "joint_only"


def field_slots(name: str, width: int | None = None) -> list[int]:
    """Return the slots of a named field, or its first ``width`` slots for a robot that fills only part of it."""
    slots = ROBOT_FIELDS[name]
    stop = slots.stop if width is None else slots.start + int(width)
    return list(range(slots.start, stop))


def slot_mask(slots) -> np.ndarray:
    """Return the [80] boolean mask that is true on ``slots``."""
    mask = np.zeros(ROBOT_DIM, dtype=bool)
    mask[np.asarray(list(slots), dtype=np.int64)] = True
    return mask


def pack_robot_rows(fields: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the [T, 80] float32 rows and the [80] slot mask of named field values [T, k], each written to the first k slots of its field."""
    length = len(next(iter(fields.values())))
    rows = np.zeros((length, ROBOT_DIM), dtype=np.float32)
    mask = np.zeros(ROBOT_DIM, dtype=bool)
    for name, values in fields.items():
        values = np.asarray(values, dtype=np.float32).reshape(length, -1)
        slots = field_slots(name, values.shape[1])
        rows[:, slots] = values
        mask[slots] = True
    return rows, mask


def rotation_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Return rotation matrices [..., 3, 3] as 6D vectors [..., 6]: the first column, then the second."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.swapaxes(rotation[..., :, :2], -1, -2).reshape(*rotation.shape[:-2], 6)


def rot6d_to_rotation(rot6d: np.ndarray) -> np.ndarray:
    """Return the rotation matrices [..., 3, 3] of 6D vectors [..., 6] by Gram-Schmidt on the two columns."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    first = rot6d[..., :3] / np.linalg.norm(rot6d[..., :3], axis=-1, keepdims=True)
    second = rot6d[..., 3:] - (first * rot6d[..., 3:]).sum(axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def gripper_closedness(value: np.ndarray, open_value: float, closed_value: float) -> np.ndarray:
    """Return a gripper reading as closedness in [0, 1]: 0 at ``open_value`` and 1 at ``closed_value``."""
    value = np.asarray(value, dtype=np.float64)
    return np.clip((value - open_value) / (closed_value - open_value), 0.0, 1.0)
