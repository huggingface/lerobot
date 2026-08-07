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

"""Pinocchio gravity + Coriolis + viscous-friction feedforward for the Metal arm.

Used by `MetalLeader` to make the arm feel weightless in a human's hand: the computed torque is
streamed as the MIT feedforward term with `kp=0`, so the motors hold the arm's own weight while
the operator supplies only the force needed to move it.

The per-joint coefficients below are the vendor's (`kdl_solver.cpp`); they scale the analytic
model to the physical arm, which carries cabling and gearbox mass the URDF does not describe.
"""

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _pinocchio_available, require_package

if TYPE_CHECKING or _pinocchio_available:
    import pinocchio as pin

# The arm's 6 revolute joints. The gripper is not part of the dynamics model: this URDF variant
# lumps its mass into Link6 (see `urdf.py`), and the leader leaves the gripper backdrivable.
NUM_ARM_JOINTS = 6

# Per-joint scaling of each analytic term, from the vendor solver.
GRAVITY_COEFFICIENTS = (1.2, 1.15, 1.1, 1.15, 1.0, 1.0)
CORIOLIS_COEFFICIENTS = (1.1, 1.15, 1.0, 1.0, 1.1, 1.1)
VISCOUS_COEFFICIENTS = (0.15, 0.3025, 0.52, 0.42, 0.015, 0.015)

# The vendor clamps the velocity fed to the viscous term on the two joints that carry the most
# inertia, so a fast swing cannot demand a friction torque larger than the joint can hold.
VISCOUS_VELOCITY_CLAMP_RAD_S = {2: 0.8, 3: 1.2}

GRAVITY_Z_M_S2 = -9.81


class MetalGravityModel:
    """Rigid-body dynamics of the Metal arm, evaluated per control tick.

    Pinocchio is imported at construction rather than at module import: `lerobot[metal]` installs
    it, but the CLI must stay importable on a machine that only has the follower's dependencies.
    """

    def __init__(self, urdf_path: str):
        require_package("pin", "metal", import_name="pinocchio")

        self.model = pin.buildModelFromUrdf(urdf_path)
        if self.model.nq != NUM_ARM_JOINTS:
            raise ValueError(
                f"Metal URDF at {urdf_path} has nq={self.model.nq}, expected exactly "
                f"{NUM_ARM_JOINTS} revolute joints. The `metal_description` variant of this file "
                "models the gripper jaws as prismatic joints (nq=8) and cannot be used here; use "
                "the `metal_sdk/example/urdf` variant."
            )
        self.data = self.model.createData()
        self.model.gravity.linear = np.array([0.0, 0.0, GRAVITY_Z_M_S2])

    def _gravity_torque(self, q_rad: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, q_rad)

    def _coriolis_torque(self, q_rad: np.ndarray, dq_rad: np.ndarray) -> np.ndarray:
        coriolis = pin.computeCoriolisMatrix(self.model, self.data, q_rad, dq_rad)
        return coriolis @ dq_rad

    def _viscous_friction_torque(self, dq_rad: np.ndarray) -> np.ndarray:
        clamped = dq_rad.copy()
        for index, limit in VISCOUS_VELOCITY_CLAMP_RAD_S.items():
            clamped[index] = np.clip(clamped[index], -limit, limit)
        return clamped * VISCOUS_COEFFICIENTS

    def feedforward_torque(self, q_rad: list[float], dq_rad: list[float]) -> list[float]:
        """Total feedforward torque (N·m) for the 6 arm joints at the given state.

        With `dq_rad` all zero this is the pure gravity term — what holds the arm up at rest.
        """
        q = np.asarray(q_rad[:NUM_ARM_JOINTS], dtype=float)
        dq = np.asarray(dq_rad[:NUM_ARM_JOINTS], dtype=float)
        torque = (
            GRAVITY_COEFFICIENTS * self._gravity_torque(q)
            + CORIOLIS_COEFFICIENTS * self._coriolis_torque(q, dq)
            + self._viscous_friction_torque(dq)
        )
        return torque.tolist()

    def blended_feedforward_torque(
        self, q_rad: list[float], dq_rad: list[float], scales: list[float]
    ) -> list[float]:
        """Gravity torque plus a per-joint fraction of the velocity-dependent terms.

        `scales[i] == 0` gives joint `i` gravity compensation only; `1.0` gives it the vendor's
        full Coriolis + friction cancellation. Values above 1 make the joint feel lighter still
        but move it toward positive feedback — past some joint-specific threshold it runs away,
        which is why these are tuned per joint on hardware rather than shared.
        """
        gravity_only = self.feedforward_torque(q_rad, [0.0] * NUM_ARM_JOINTS)
        if not any(scale != 0.0 for scale in scales):
            return gravity_only

        full = self.feedforward_torque(q_rad, dq_rad)
        return [
            gravity + scale * (total - gravity)
            for gravity, total, scale in zip(gravity_only, full, scales, strict=True)
        ]
