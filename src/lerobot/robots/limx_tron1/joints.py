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

"""Joint layout for the LimX TRON1 legged robots.

TRON1 ships as several legged builds, all driven by the same low-level SDK through
``Robot(RobotType.PointFoot)``.  The build is selected by the ``ROBOT_TYPE`` environment variable
(``PF_*``, ``SF_*`` or ``WF_*``); the SDK reads that variable when it is first imported and picks
the joint table for the build.  The three build families differ in the fourth per-leg joint and
therefore in vector width:

- ``PF_*`` (point foot) -- 6 actuated joints, three per leg, no ankle;
- ``SF_*`` (sole foot) -- 8 actuated joints, an ``ankle`` as the fourth per-leg joint;
- ``WF_*`` (wheel foot) -- 8 actuated joints, a ``wheel`` as the fourth per-leg joint.

The joint names below are the vendor's own, taken verbatim from the ``joint_names`` list in the
build's deployment parameters (``tron1-rl-deploy-python``, ``controllers/model/<build>/params.yaml``).
The SDK's ``getMotorNames()`` is not usable as a name source: on these builds it returns
concatenated names (``"ankle_L_Jointabad_R_Joint"``).  The SDK addresses joints positionally and
routes commands by index -- the vendor's own controller leaves ``RobotCmd.motor_names`` empty -- so
the name table is for LeRobot's feature keys only.

The order is left leg first, then right leg.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

#: Build families recognised by the driver, keyed by the ``ROBOT_TYPE`` prefix.
#:
#: A TRON1 build is selected by the ``ROBOT_TYPE`` environment variable the SDK reads at import
#: time.  Only the prefix matters for the joint layout: ``PF_*`` is the 6-joint point foot,
#: ``SF_*`` the 8-joint sole foot, ``WF_*`` the 8-joint wheel foot.
POINTFOOT = "pointfoot"
SOLEFOOT = "solefoot"
WHEELFOOT = "wheelfoot"

_VARIANTS: tuple[str, ...] = (POINTFOOT, SOLEFOOT, WHEELFOOT)

#: Per-variant number of actuated joints in a state or command vector.
VARIANT_DIMS: dict[str, int] = {
    POINTFOOT: 6,
    SOLEFOOT: 8,
    WHEELFOOT: 8,
}

#: Default joint names, in SDK vector order (left leg first, then right leg), per variant.
#:
#: These are the vendor's own names from the build's ``params.yaml``.  The SDK indexes joints
#: positionally and does not ship a trustworthy name table, so a deployment that must match an
#: existing dataset should override ``joint_names`` on the config.
VARIANT_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    POINTFOOT: (
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
    ),
    SOLEFOOT: (
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "ankle_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
        "ankle_R_Joint",
    ),
    WHEELFOOT: (
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "wheel_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
        "wheel_R_Joint",
    ),
}

#: Default position gains, one per joint in `VARIANT_JOINT_NAMES` order.
#:
#: Taken from the vendor's deployment parameters for the TRON1A build of each family
#: (``tron1-rl-deploy-python``, ``controllers/model/{PF,SF,WF}_TRON1A/params.yaml``).  The vendor
#: reuses one ``stiffness`` value for every joint, so each vector is uniform.
VARIANT_KP: dict[str, tuple[float, ...]] = {
    POINTFOOT: (42.0,) * 6,
    SOLEFOOT: (45.0,) * 8,
    WHEELFOOT: (42.0,) * 8,
}

#: Default derivative gains, matching `VARIANT_KP` joint for joint.
#:
#: The vendor uses one ``damping`` value for every joint except the fourth per-leg joint on the
#: sole- and wheel-foot builds, which carries its own lower damping (``ankle_joint_damping`` /
#: ``wheel_joint_damping`` in the same ``params.yaml``).
VARIANT_KD: dict[str, tuple[float, ...]] = {
    POINTFOOT: (3.5,) * 6,
    # abad, hip, knee, ankle (L), then the right leg: ankle damping is 1.5.
    SOLEFOOT: (3.0, 3.0, 3.0, 1.5, 3.0, 3.0, 3.0, 1.5),
    # abad, hip, knee, wheel (L), then the right leg: wheel damping is 0.8.
    WHEELFOOT: (2.5, 2.5, 2.5, 0.8, 2.5, 2.5, 2.5, 0.8),
}

#: Suffix carried by every per-joint scalar in a LeRobot observation/action dict.
POS_SUFFIX = ".pos"


def variant_from_robot_type(robot_type: str | None) -> str:
    """Map a ``ROBOT_TYPE`` value onto a build family.

    Args:
        robot_type (`str | None`):
            The SDK ``ROBOT_TYPE`` value, e.g. ``"PF_TRON1A"``, ``"SF_TRON1A"`` or
            ``"WF_TRON1A"``.  `None` selects the point-foot family, the TRON1 default.

    Returns:
        `str`: One of [`POINTFOOT`], [`SOLEFOOT`] or [`WHEELFOOT`].

    Raises:
        ValueError: If the value carries an unknown prefix.
    """
    if robot_type is None:
        return POINTFOOT
    prefix = robot_type.strip().upper()
    if prefix.startswith("SF"):
        return SOLEFOOT
    if prefix.startswith("WF"):
        return WHEELFOOT
    if prefix.startswith("PF"):
        return POINTFOOT
    raise ValueError(f"unknown TRON1 build {robot_type!r}: expected a PF_/SF_/WF_ prefixed ROBOT_TYPE")


def pos_key(name: str) -> str:
    """LeRobot feature key for a single scalar joint value.

    Args:
        name (`str`):
            Joint name without the suffix.

    Returns:
        `str`: The joint name with `.pos` appended, e.g. `"hip_L_Joint.pos"`.
    """
    return f"{name}{POS_SUFFIX}"


def resolve_joint_names(names: Sequence[str] | None, variant: str) -> tuple[str, ...]:
    """Validate a user-supplied joint name table, or fall back to the variant default.

    Args:
        names (`Sequence[str] | None`):
            The table to validate.  `None` selects the default table for `variant`.
        variant (`str`):
            One of [`POINTFOOT`], [`SOLEFOOT`] or [`WHEELFOOT`]; determines the expected width.

    Returns:
        `tuple[str, ...]`: The name table, guaranteed to hold the variant's width and to contain
        unique entries.

    Raises:
        ValueError: If the table has the wrong length or repeats a name.
    """
    dim = VARIANT_DIMS[variant]
    if names is None:
        return VARIANT_JOINT_NAMES[variant]
    resolved = tuple(names)
    if len(resolved) != dim:
        raise ValueError(f"joint_names must have {dim} entries for the {variant} build, got {len(resolved)}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("joint_names contains duplicates")
    return resolved


def resolve_joint_vector(values: Sequence[float] | None, dim: int, field: str) -> tuple[float, ...]:
    """Validate a per-joint numeric table against a vector width.

    Args:
        values (`Sequence[float] | None`):
            The table to validate.  `None` selects the neutral value (all zeros).
        dim (`int`):
            Expected number of entries.
        field (`str`):
            Field name, used in the error message.

    Returns:
        `tuple[float, ...]`: The `dim`-long table.

    Raises:
        ValueError: If the table has the wrong length.
    """
    if values is None:
        return (0.0,) * dim
    resolved = tuple(float(value) for value in values)
    if len(resolved) != dim:
        raise ValueError(f"{field} must have {dim} entries, got {len(resolved)}")
    return resolved


def joint_vector(action: Mapping[str, float], joint_names: Sequence[str]) -> np.ndarray:
    """Build the joint vector from a LeRobot action dictionary.

    Keys are the ``"<joint>.pos"`` form produced by [`pos_key`].

    Args:
        action (`Mapping[str, float]`):
            The action dictionary to read.
        joint_names (`Sequence[str]`):
            Joint names in SDK vector order.

    Returns:
        `np.ndarray`: The joint target, as long as `joint_names`.

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
