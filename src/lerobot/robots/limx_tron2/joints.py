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

"""Joint layout helpers for the TRON2 arms.

The TRON2 runtime describes its arms as a single 16-value ServoJ target
(``[left_arm(7), right_arm(7), head(2)]``) plus a pair of gripper openings, and
reports an 18-value state (``[left_arm(7), left_gripper(1), right_arm(7),
right_gripper(1), head(2)]``).  LeRobot, by contrast, wants a flat ``{name:
value}`` dictionary on both directions.

This module owns the translation between the two representations, so the
mapping lives in exactly one place.  The slice boundaries mirror
``tron2_env.joints.JointIndex``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ARM_DIM = 7
HEAD_DIM = 2

#: Width of a ServoJ/action vector: both arms plus the head.
SERVOJ_DIM = ARM_DIM * 2 + HEAD_DIM  # 16

#: Width of the state vector reported by the runtime (adds both grippers).
STATE_DIM = ARM_DIM * 2 + 2 + HEAD_DIM  # 18

# Slices into the 18-value state vector.
LEFT_ARM_SLICE = slice(0, ARM_DIM)
LEFT_GRIPPER_INDEX = ARM_DIM
RIGHT_ARM_SLICE = slice(ARM_DIM + 1, ARM_DIM + 1 + ARM_DIM)
RIGHT_GRIPPER_INDEX = ARM_DIM + 1 + ARM_DIM
HEAD_SLICE = slice(RIGHT_GRIPPER_INDEX + 1, RIGHT_GRIPPER_INDEX + 1 + HEAD_DIM)

_ARM_JOINT_SUFFIXES: tuple[str, ...] = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

#: Placeholder joint names, in ServoJ order.
#:
#: The names for the arms are descriptive defaults; the runtime itself indexes
#: joints positionally and does not ship a name table.  Deployments that need to
#: match an existing dataset or policy should override ``joint_names`` in the
#: robot config rather than edit this tuple.
DEFAULT_JOINT_NAMES: tuple[str, ...] = (
    *(f"left_arm_{suffix}" for suffix in _ARM_JOINT_SUFFIXES),
    *(f"right_arm_{suffix}" for suffix in _ARM_JOINT_SUFFIXES),
    "head_pitch",
    "head_yaw",
)

DEFAULT_GRIPPER_NAMES: tuple[str, ...] = ("left_gripper", "right_gripper")

#: Suffix carried by every per-joint scalar in a LeRobot observation/action dict.
POS_SUFFIX = ".pos"


def pos_key(name: str) -> str:
    """LeRobot feature key for a single scalar joint value.

    Args:
        name (`str`):
            Joint name without the suffix.

    Returns:
        `str`: The joint name with `.pos` appended, e.g. `"left_arm_elbow.pos"`.
    """
    return f"{name}{POS_SUFFIX}"


def resolve_joint_names(names: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a user-supplied joint name table, or fall back to the default.

    Args:
        names (`Sequence[str] | None`):
            The table to validate.  `None` selects `DEFAULT_JOINT_NAMES`.

    Returns:
        `tuple[str, ...]`: The name table, guaranteed to hold `SERVOJ_DIM` unique entries.

    Raises:
        ValueError: If the table has the wrong length or repeats a name.
    """
    if names is None:
        return DEFAULT_JOINT_NAMES
    resolved = tuple(names)
    if len(resolved) != SERVOJ_DIM:
        raise ValueError(
            f"joint_names must have {SERVOJ_DIM} entries "
            f"(two arms of {ARM_DIM} plus {HEAD_DIM} head joints), got {len(resolved)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("joint_names contains duplicates")
    return resolved


def resolve_gripper_names(names: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a user-supplied gripper name pair, or fall back to the default.

    Args:
        names (`Sequence[str] | None`):
            The pair to validate.  `None` selects `DEFAULT_GRIPPER_NAMES`.

    Returns:
        `tuple[str, ...]`: The two gripper names.

    Raises:
        ValueError: If the pair does not hold exactly two names.
    """
    if names is None:
        return DEFAULT_GRIPPER_NAMES
    resolved = tuple(names)
    if len(resolved) != 2:
        raise ValueError(f"gripper_names must have 2 entries, got {len(resolved)}")
    return resolved


def _as_state_vector(state: Sequence[float] | np.ndarray) -> np.ndarray:
    """Coerce a state sample into a finite `STATE_DIM`-long float vector.

    Args:
        state (`Sequence[float] | np.ndarray`):
            The raw state sample.

    Returns:
        `np.ndarray`: The validated vector.

    Raises:
        ValueError: If the vector is not `STATE_DIM` long or holds a non-finite value.
    """
    vector = np.asarray(state, dtype=np.float64).reshape(-1)
    if vector.size != STATE_DIM:
        raise ValueError(f"expected a {STATE_DIM}-value state vector, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("state vector contains non-finite values")
    return vector


def split_state(
    state: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray]:
    """Split an 18-value state into ``(left_arm, left_gripper, right_arm, right_gripper, head)``.

    Args:
        state (`Sequence[float] | np.ndarray`):
            The state vector reported by the runtime.

    Returns:
        `tuple[np.ndarray, float, np.ndarray, float, np.ndarray]`: The two seven-joint arms, the two
        gripper openings, and the two head joints.

    Raises:
        ValueError: If the vector is not `STATE_DIM` long or holds a non-finite value.
    """
    vector = _as_state_vector(state)
    return (
        vector[LEFT_ARM_SLICE].copy(),
        float(vector[LEFT_GRIPPER_INDEX]),
        vector[RIGHT_ARM_SLICE].copy(),
        float(vector[RIGHT_GRIPPER_INDEX]),
        vector[HEAD_SLICE].copy(),
    )


def state_to_servoj(state: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert an 18-value state into the 16-value ServoJ layout.

    Used to seed an action from the measured pose -- exactly what
    ``tron2_env.motion.arm_strategy.state_to_servoj`` does.

    Args:
        state (`Sequence[float] | np.ndarray`):
            The state vector reported by the runtime.

    Returns:
        `np.ndarray`: Both arms and the head, with the gripper slots dropped.

    Raises:
        ValueError: If the vector is not `STATE_DIM` long or holds a non-finite value.
    """
    left_arm, _, right_arm, _, head = split_state(state)
    return np.concatenate((left_arm, right_arm, head))


def servoj_to_state(
    servoj: Sequence[float] | np.ndarray,
    left_gripper: float = 0.0,
    right_gripper: float = 0.0,
) -> np.ndarray:
    """Inverse of [`state_to_servoj`], filling the gripper slots explicitly.

    Args:
        servoj (`Sequence[float] | np.ndarray`):
            Both arms and the head, in ServoJ order.
        left_gripper (`float`, *optional*, defaults to 0.0):
            Opening to write into the left gripper slot.
        right_gripper (`float`, *optional*, defaults to 0.0):
            Opening to write into the right gripper slot.

    Returns:
        `np.ndarray`: The `STATE_DIM`-long state vector.

    Raises:
        ValueError: If the ServoJ vector is not `SERVOJ_DIM` long.
    """
    vector = np.asarray(servoj, dtype=np.float64).reshape(-1)
    if vector.size != SERVOJ_DIM:
        raise ValueError(f"expected a {SERVOJ_DIM}-value ServoJ vector, got {vector.size}")
    state = np.zeros(STATE_DIM, dtype=np.float64)
    state[LEFT_ARM_SLICE] = vector[:ARM_DIM]
    state[LEFT_GRIPPER_INDEX] = left_gripper
    state[RIGHT_ARM_SLICE] = vector[ARM_DIM : ARM_DIM * 2]
    state[RIGHT_GRIPPER_INDEX] = right_gripper
    state[HEAD_SLICE] = vector[ARM_DIM * 2 :]
    return state


def servoj_vector(
    action: Mapping[str, float],
    joint_names: Sequence[str],
) -> np.ndarray:
    """Build the 16-value ServoJ vector from a LeRobot action dictionary.

    Keys are the ``"<joint>.pos"`` form produced by [`pos_key`].

    Args:
        action (`Mapping[str, float]`):
            The action dictionary to read.
        joint_names (`Sequence[str]`):
            Joint names in ServoJ order.

    Returns:
        `np.ndarray`: The `SERVOJ_DIM`-long ServoJ target.

    Raises:
        KeyError: If the action is missing one of the expected joint keys.
    """
    keys = [pos_key(name) for name in joint_names]
    missing = [key for key in keys if key not in action]
    if missing:
        raise KeyError(f"action is missing joint keys: {missing}")
    return np.asarray([action[key] for key in keys], dtype=np.float64)


def gripper_openings(
    action: Mapping[str, float],
    gripper_names: Sequence[str],
) -> tuple[float, float] | None:
    """Extract the paired gripper openings, or ``None`` if the action omits them.

    Both keys must be present or absent together: a half-specified gripper pair would silently command the
    other side to its previous value.

    Args:
        action (`Mapping[str, float]`):
            The action dictionary to read.
        gripper_names (`Sequence[str]`):
            The left and right gripper names.

    Returns:
        `tuple[float, float] | None`: The two openings, or `None` when the action carries neither.

    Raises:
        KeyError: If the action carries exactly one of the two keys.
    """
    keys = [pos_key(name) for name in gripper_names]
    present = [key in action for key in keys]
    if not any(present):
        return None
    if not all(present):
        missing = [k for k, ok in zip(keys, present, strict=True) if not ok]
        raise KeyError(f"action specifies only one side of the gripper pair; missing {missing}")
    return float(action[keys[0]]), float(action[keys[1]])
