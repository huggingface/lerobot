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

"""The proprioception G1 controllers read from a lowstate."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.robots.unitree_g1.g1_utils import NUM_MOTORS, get_gravity_orientation


@dataclass(frozen=True)
class G1Proprio:
    q: np.ndarray
    """(29,) joint positions, in DDS slot order (``G1_29_JointIndex``)."""
    dq: np.ndarray
    """(29,) joint velocities, in DDS slot order."""
    quat: np.ndarray
    """(4,) base orientation ``(w, x, y, z)``, as the IMU reports it."""
    ang_vel: np.ndarray
    """(3,) base angular velocity from the gyroscope."""
    gravity: np.ndarray
    """(3,) gravity direction in the base frame."""


def read_proprio(lowstate: Any) -> G1Proprio:
    """Read the joint and IMU state of every DDS slot from a lowstate."""
    quaternion = lowstate.imu_state.quaternion
    return G1Proprio(
        q=np.array([lowstate.motor_state[i].q for i in range(NUM_MOTORS)], np.float32),
        dq=np.array([lowstate.motor_state[i].dq for i in range(NUM_MOTORS)], np.float32),
        quat=np.array(quaternion, np.float32),
        ang_vel=np.array(lowstate.imu_state.gyroscope, np.float32),
        gravity=get_gravity_orientation(quaternion),
    )
