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

"""Embodiment metadata without importing IK, simulators, or hardware drivers."""

from dataclasses import dataclass
from enum import IntEnum

from .g1_utils import G1_23_JointArmIndex, G1_23_JointIndex, G1_29_JointArmIndex, G1_29_JointIndex


@dataclass(frozen=True)
class G1Embodiment:
    name: str
    joint_index: type[IntEnum]
    arm_index: type[IntEnum]
    model_repository: str
    model_urdf: str
    simulation_env: str | None = None
    supports_hardware: bool = False
    supports_controller: bool = False
    supports_gravity_compensation: bool = False


_EMBODIMENTS = {
    "g1_29": G1Embodiment(
        name="g1_29",
        joint_index=G1_29_JointIndex,
        arm_index=G1_29_JointArmIndex,
        model_repository="https://huggingface.co/lerobot/unitree-g1-mujoco",
        model_urdf="assets/g1_body29_hand14.urdf",
        simulation_env="lerobot/unitree-g1-mujoco",
        supports_hardware=True,
        supports_controller=True,
        supports_gravity_compensation=True,
    ),
    "g1_23": G1Embodiment(
        name="g1_23",
        joint_index=G1_23_JointIndex,
        arm_index=G1_23_JointArmIndex,
        model_repository="https://github.com/unitreerobotics/unitree_lerobot",
        model_urdf="unitree_lerobot/eval_robot/assets/g1/g1_body23.urdf",
    ),
}


def get_g1_embodiment(name: str) -> G1Embodiment:
    try:
        return _EMBODIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown G1 embodiment: {name!r}. Available: {list(_EMBODIMENTS)}") from exc
