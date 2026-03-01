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

"""Translational manipulability metrics for singularity detection.

Computes σ_min(Jv) and cond(Jv) where Jv is the 3×n_arm translational
Jacobian at the end-effector frame. Used to detect near-singular
configurations during teleop and inference.

Frame convention: Jv is extracted from the full 6×nv frame Jacobian
returned by placo.RobotWrapper.frame_jacobian(), which uses
LOCAL_WORLD_ALIGNED reference (origin at frame, axes aligned with world).

Translation-only to avoid unit-mixing with angular velocity components.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ManipulabilityResult:
    """Result of a manipulability computation.

    Attributes:
        sigma_min: Smallest singular value of Jv. Approaches 0 at singularities.
            Units: meters/radian.
        sigma_max: Largest singular value of Jv.
        condition_number: σ_max / σ_min. Approaches ∞ at singularities.
        singular_values: All singular values of Jv, shape (min(3, n_arm),).
    """

    sigma_min: float
    sigma_max: float
    condition_number: float
    singular_values: np.ndarray


def extract_translational_jacobian(
    j_full: np.ndarray,
    arm_joint_indices: list[int] | None = None,
) -> np.ndarray:
    """Extract the translational Jacobian Jv from a full 6×nv frame Jacobian.

    Args:
        j_full: Full frame Jacobian, shape (6, nv). Top 3 rows = linear velocity,
            bottom 3 rows = angular velocity.
        arm_joint_indices: Column indices for arm joints. If None, uses all columns.

    Returns:
        Translational Jacobian, shape (3, n_arm).
    """
    if arm_joint_indices is not None:
        return j_full[:3, arm_joint_indices]
    return j_full[:3, :]


def compute_manipulability(jv: np.ndarray) -> ManipulabilityResult:
    """Compute manipulability metrics from a translational Jacobian.

    Args:
        jv: Translational Jacobian, shape (3, n_arm).

    Returns:
        ManipulabilityResult with σ_min, σ_max, condition number, and all singular values.
    """
    sigmas = np.linalg.svd(jv, compute_uv=False)
    sigma_min = float(sigmas[-1])
    sigma_max = float(sigmas[0])
    cond = sigma_max / sigma_min if sigma_min > 1e-15 else float("inf")
    return ManipulabilityResult(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=cond,
        singular_values=sigmas,
    )
