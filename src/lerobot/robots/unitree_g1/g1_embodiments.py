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

"""G1 embodiment definitions without importing IK, simulators, or hardware drivers."""

from dataclasses import dataclass
from enum import IntEnum

from .g1_utils import NUM_MOTORS, G1_23_JointArmIndex, G1_23_JointIndex, G1_29_JointArmIndex, G1_29_JointIndex

# Identical physical joints use identical gains across G1 embodiments. Embodiments
# project this table into the sparse, 29-slot DDS layout below.
_DEFAULT_GAINS_BY_JOINT: dict[str, tuple[float, float]] = {
    "kLeftHipPitch": (150.0, 2.0),
    "kLeftHipRoll": (150.0, 2.0),
    "kLeftHipYaw": (150.0, 2.0),
    "kLeftKnee": (300.0, 4.0),
    "kLeftAnklePitch": (40.0, 2.0),
    "kLeftAnkleRoll": (40.0, 2.0),
    "kRightHipPitch": (150.0, 2.0),
    "kRightHipRoll": (150.0, 2.0),
    "kRightHipYaw": (150.0, 2.0),
    "kRightKnee": (300.0, 4.0),
    "kRightAnklePitch": (40.0, 2.0),
    "kRightAnkleRoll": (40.0, 2.0),
    "kWaistYaw": (250.0, 5.0),
    "kWaistRoll": (250.0, 5.0),
    "kWaistPitch": (250.0, 5.0),
    "kLeftShoulderPitch": (50.0, 3.0),
    "kLeftShoulderRoll": (50.0, 3.0),
    "kLeftShoulderYaw": (80.0, 3.0),
    "kLeftElbow": (80.0, 3.0),
    "kLeftWristRoll": (40.0, 1.5),
    "kLeftWristPitch": (40.0, 1.5),
    "kLeftWristYaw": (40.0, 1.5),
    "kRightShoulderPitch": (50.0, 3.0),
    "kRightShoulderRoll": (50.0, 3.0),
    "kRightShoulderYaw": (80.0, 3.0),
    "kRightElbow": (80.0, 3.0),
    "kRightWristRoll": (40.0, 1.5),
    "kRightWristPitch": (40.0, 1.5),
    "kRightWristYaw": (40.0, 1.5),
}


@dataclass(frozen=True)
class G1Embodiment:
    name: str
    joint_index: type[IntEnum]
    arm_index: type[IntEnum]
    unsupported_controllers: frozenset[str] = frozenset()
    supports_simulation: bool = False
    supports_gravity_compensation: bool = False

    def default_gains(self) -> tuple[list[float], list[float]]:
        kp, kd = [0.0] * NUM_MOTORS, [0.0] * NUM_MOTORS
        for joint in self.joint_index:
            kp[joint.value], kd[joint.value] = _DEFAULT_GAINS_BY_JOINT[joint.name]
        return kp, kd


_EMBODIMENTS = {
    "g1_29": G1Embodiment(
        name="g1_29",
        joint_index=G1_29_JointIndex,
        arm_index=G1_29_JointArmIndex,
        supports_simulation=True,
        supports_gravity_compensation=True,
    ),
    "g1_23": G1Embodiment(
        name="g1_23",
        joint_index=G1_23_JointIndex,
        arm_index=G1_23_JointArmIndex,
        unsupported_controllers=frozenset({"SonicWholeBodyController"}),
    ),
}


def get_g1_embodiment(name: str) -> G1Embodiment:
    try:
        return _EMBODIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown G1 embodiment: {name!r}. Available: {list(_EMBODIMENTS)}") from exc
