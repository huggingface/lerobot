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

"""Put an object in a container: pick it up, drop it inside, optionally go back to rest."""

import mujoco
import numpy as np

from ..motion import Planner
from ..scene import SceneInfo, SceneParams
from ..so101 import REST
from . import Task, TaskInstance, find, footprint, phrase, register, touches_gripper

TEMPLATES = [
    "put the {obj} in the {target}",
    "place the {obj} into the {target}",
    "move the {obj} into the {target}",
    "drop the {obj} in the {target}",
    "pick up the {obj} and put it in the {target}",
    "grab the {obj} and place it in the {target}",
    "store the {obj} in the {target}",
    "transfer the {obj} to the {target}",
    "the {obj} goes in the {target}",
    "can you put the {obj} in the {target}?",
    "put the {obj} inside the {target}",
    "take the {obj} and drop it into the {target}",
]


@register
class PlaceIn(Task):
    name = "place_in"
    containers = 1
    min_objects = 1

    def sample(self, scene: SceneParams, rng: np.random.Generator) -> TaskInstance | None:
        container = scene.containers[0]
        # It must fit, with room to spare for an imprecise drop.
        fits = [o for o in scene.objects if footprint(o) < container.inner_half - 0.012]
        if not fits:
            return None
        obj = fits[rng.integers(len(fits))]
        names = {"obj": obj.label, "target": container.label}
        instruction, phrasings = phrase(TEMPLATES, rng, 10, **names)
        return TaskInstance(
            task=self.name,
            instruction=instruction,
            roles={"object": obj.name, "target": container.name},
            phrasings=phrasings,
            memories=[f"I picked up the {obj.label}.", f"The {obj.label} is in the {container.label}."],
        )

    def script(self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator):
        obj = find(scene, instance.roles["object"])
        container = find(scene, instance.roles["target"])
        planner.subtask(f"pick up the {obj.label}")
        grasp = planner.pick(obj)
        planner.subtask(f"put the {obj.label} in the {container.label}")
        # Somewhere inside, with the object's footprint clear of the walls.
        room = max(container.inner_half - footprint(obj) - 0.008, 0.0)
        offset = rng.uniform(-1, 1, 2) * room * 0.7
        clearance = rng.uniform(0.008, 0.025)
        release = np.array(
            [
                container.pos[0] + offset[0],
                container.pos[1] + offset[1],
                container.thickness + grasp.height + clearance,
            ]
        )
        planner.place(release, grasp.tilt)
        if rng.random() < planner.cfg.return_rest_prob:
            planner.subtask("go back to the rest position")
            planner.transit(REST + rng.uniform(-3, 3, 6) * [1, 1, 1, 1, 1, 0])

    def success(
        self, instance: TaskInstance, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo
    ) -> bool:
        container = find(info.params, instance.roles["target"])
        body = info.object_bodies[instance.roles["object"]]
        center = info.container_bodies[instance.roles["target"]]
        rel = data.xmat[center].reshape(3, 3).T @ (data.xpos[body] - data.xpos[center])
        inside = bool(
            np.all(np.abs(rel[:2]) < container.inner_half) and rel[2] < container.wall_height + 0.03
        )
        return inside and not touches_gripper(instance.roles["object"], model, data, info)
