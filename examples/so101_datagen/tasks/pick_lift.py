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

"""Pick an object up and hold it in the air."""

import mujoco
import numpy as np

from ..motion import Planner
from ..scene import SceneInfo, SceneParams
from . import Task, TaskInstance, find, is_lifted, phrase, register

TEMPLATES = [
    "pick up the {obj}",
    "grab the {obj}",
    "lift the {obj}",
    "take the {obj}",
    "pick the {obj} up",
    "grasp the {obj} and lift it",
    "get the {obj}",
    "raise the {obj} off the table",
    "can you pick up the {obj}?",
    "lift up the {obj}",
]


@register
class PickLift(Task):
    name = "pick_lift"

    def sample(self, scene: SceneParams, rng: np.random.Generator) -> TaskInstance | None:
        obj = scene.objects[rng.integers(len(scene.objects))]
        instruction, phrasings = phrase(TEMPLATES, rng, 10, obj=obj.label)
        return TaskInstance(
            task=self.name,
            instruction=instruction,
            roles={"object": obj.name},
            phrasings=phrasings,
            memories=[f"I picked up the {obj.label}."],
        )

    def script(self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator):
        obj = find(scene, instance.roles["object"])
        planner.subtask(f"pick up the {obj.label}")
        grasp = planner.pick(obj)
        planner.subtask(f"hold the {obj.label} up")
        # Bring it up and a little towards the robot, then hold it there.
        pos, _ = planner.arm.tcp_pose(planner.q)
        up = pos + np.array([*(-pos[:2] * rng.uniform(0.0, 0.3)), rng.uniform(0.02, 0.08)])
        planner.reach(up, grasp.tilt)
        planner.dwell(rng.uniform(0.5, 1.2))

    def success(
        self, instance: TaskInstance, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo
    ) -> bool:
        return is_lifted(instance.roles["object"], model, data, info)
