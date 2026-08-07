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

"""Friction feedforward for the Metal leader's gripper.

The gripper is not in the arm's dynamics model, so it gets its own compensation: without it the
leader's jaws are noticeably stiffer to squeeze than the follower's are to drive, and the operator
feels a dead band before the follower responds. Ported from the vendor's
`GripperTorqueCompensation` (`kdl_solver.cpp`).
"""

import math

# Applied at rest so the jaws break free of stiction on the first squeeze rather than needing a
# jolt of extra hand force.
STOP_TORQUE_NM = 0.06

# Coulomb friction, applied in the direction of travel once moving.
STATIC_FRICTION_NM = 0.03

VISCOUS_COEFFICIENT = 0.01
VISCOUS_VELOCITY_CLAMP_RAD_S = 3.0

# Below this the gripper counts as stationary. Motor velocity noise would otherwise flip the sign
# of the Coulomb term every tick and make the jaws buzz at rest.
AT_REST_VELOCITY_RAD_S = 1e-5


def gripper_friction_torque(velocity_rad_s: float) -> float:
    """Feedforward torque (N·m) that cancels the gripper's own friction.

    Scale or disable it with `MetalLeaderConfig.gripper_friction_scale`.
    """
    if abs(velocity_rad_s) < AT_REST_VELOCITY_RAD_S:
        return STOP_TORQUE_NM

    coulomb = math.copysign(STATIC_FRICTION_NM, velocity_rad_s)
    clamped = max(-VISCOUS_VELOCITY_CLAMP_RAD_S, min(VISCOUS_VELOCITY_CLAMP_RAD_S, velocity_rad_s))
    return clamped * VISCOUS_COEFFICIENT + coulomb
