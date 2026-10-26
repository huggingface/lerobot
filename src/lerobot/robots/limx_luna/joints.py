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

"""Joint layout for the LimX Dynamics Luna humanoid robot (``HU_L04``).

The vendor low-level SDK behind [`limx_luna`][lerobot.robots.limx_luna] addresses the actuated
joints positionally: ``RobotState.q`` and ``RobotCmd.q`` are plain lists whose order is fixed by
the robot build rather than carried by the payload.  The SDK's own ``getMotorNames()`` is not
usable as a name source -- on the humanoid build it reports nothing until a state message has
been received, which requires a connected robot.

The table below therefore uses the vendor's own joint names in the hardware order of the Luna
controller configuration, as published in ``luna-beyondmimic``
(``tools/offline_onnx_exporter/deploy_config.py``, ``REFERENCE_JOINT_NAMES``, and the per-joint
torque budget in ``param.yaml``, both in ``config.yaml`` hardware order).  The URDF in
``luna-description`` (``HU_L04_01.urdf``) lists the joints in the same order.  Deployments that
have to match an existing dataset or policy checkpoint should override ``joint_names`` on the
robot config rather than edit this table.

The order is legs, waist, head, then arms.  Luna's arms have five joints each (no wrist pitch or
roll), so the vector is 27 wide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

#: Number of actuated joints in a Luna state or command vector.
LUNA_DIM = 27

_LEFT_LEG_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)

_RIGHT_LEG_JOINTS: tuple[str, ...] = (
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

_WAIST_JOINTS: tuple[str, ...] = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

_HEAD_JOINTS: tuple[str, ...] = ("head_yaw_joint", "head_pitch_joint")

_ARM_JOINT_SUFFIXES: tuple[str, ...] = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_yaw",
)

#: Default joint names, in SDK vector order.
#:
#: The SDK indexes joints positionally and does not ship a trustworthy name table, so a
#: deployment that must match an existing dataset should override ``joint_names`` on the config.
DEFAULT_JOINT_NAMES: tuple[str, ...] = (
    *_LEFT_LEG_JOINTS,
    *_RIGHT_LEG_JOINTS,
    *_WAIST_JOINTS,
    *_HEAD_JOINTS,
    *(f"left_{suffix}_joint" for suffix in _ARM_JOINT_SUFFIXES),
    *(f"right_{suffix}_joint" for suffix in _ARM_JOINT_SUFFIXES),
)

#: Suffix carried by every per-joint scalar in a LeRobot observation/action dict.
POS_SUFFIX = ".pos"


def pos_key(name: str) -> str:
    """LeRobot feature key for a single scalar joint value.

    Args:
        name (`str`):
            Joint name without the suffix.

    Returns:
        `str`: The joint name with `.pos` appended, e.g. `"left_elbow_joint.pos"`.
    """
    return f"{name}{POS_SUFFIX}"


def resolve_joint_names(names: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a user-supplied joint name table, or fall back to the default.

    Args:
        names (`Sequence[str] | None`):
            The table to validate.  `None` selects `DEFAULT_JOINT_NAMES`.

    Returns:
        `tuple[str, ...]`: The name table, guaranteed to hold `LUNA_DIM` unique entries.

    Raises:
        ValueError: If the table has the wrong length or repeats a name.
    """
    if names is None:
        return DEFAULT_JOINT_NAMES
    resolved = tuple(names)
    if len(resolved) != LUNA_DIM:
        raise ValueError(
            f"joint_names must have {LUNA_DIM} entries (12 leg, 3 waist, 2 head, 10 arm "
            f"joints), got {len(resolved)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("joint_names contains duplicates")
    return resolved


def resolve_joint_vector(values: Sequence[float] | None, field: str) -> tuple[float, ...]:
    """Validate a per-joint numeric table against the vector width.

    Args:
        values (`Sequence[float] | None`):
            The table to validate.  `None` selects the neutral value (all zeros).
        field (`str`):
            Field name, used in the error message.

    Returns:
        `tuple[float, ...]`: The `LUNA_DIM`-long table.

    Raises:
        ValueError: If the table has the wrong length or holds a non-finite value.
    """
    if values is None:
        return (0.0,) * LUNA_DIM
    resolved = tuple(float(value) for value in values)
    if len(resolved) != LUNA_DIM:
        raise ValueError(f"{field} must have {LUNA_DIM} entries, got {len(resolved)}")
    if not np.all(np.isfinite(resolved)):
        # These tables seed `start_pose`, `kp` and `kd`, which reach the SDK without passing
        # through `joint_vector` -- the action path already rejects non-finite targets, and a
        # NaN that only ever arrives through the configuration would otherwise be published
        # for as long as the robot is connected.
        raise ValueError(f"{field} contains non-finite values")
    return resolved


def joint_vector(action: Mapping[str, float], joint_names: Sequence[str]) -> np.ndarray:
    """Build the 27-value joint vector from a LeRobot action dictionary.

    Keys are the ``"<joint>.pos"`` form produced by [`pos_key`].

    Args:
        action (`Mapping[str, float]`):
            The action dictionary to read.
        joint_names (`Sequence[str]`):
            Joint names in SDK vector order.

    Returns:
        `np.ndarray`: The `LUNA_DIM`-long joint target.

    Raises:
        KeyError: If the action is missing one of the expected joint keys.
        ValueError: If a joint value is not finite.
    """
    keys = [pos_key(name) for name in joint_names]
    missing = [key for key in keys if key not in action]
    if missing:
        raise KeyError(f"action is missing joint keys: {missing}")
    vector = np.asarray([action[key] for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("joint target contains non-finite values")
    return vector
