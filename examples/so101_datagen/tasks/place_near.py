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

"""Place an object next to another one on the table."""

import mujoco
import numpy as np

from ..motion import Planner, PlanningError
from ..scene import SceneInfo, SceneParams
from ..so101 import REST
from . import Task, TaskInstance, find, footprint, phrase, register, resting_on_table

TEMPLATES = [
    "place the {obj} next to the {target}",
    "put the {obj} next to the {target}",
    "move the {obj} next to the {target}",
    "put the {obj} beside the {target}",
    "place the {obj} near the {target}",
    "set the {obj} down next to the {target}",
    "pick up the {obj} and put it next to the {target}",
    "bring the {obj} close to the {target}",
    "move the {obj} so it is next to the {target}",
    "can you put the {obj} beside the {target}?",
    "put the {obj} by the {target}",
]
# Gap between the two footprints when placed, and how far off counts as "next to".
GAP = (0.01, 0.035)
NEAR = 0.05


@register
class PlaceNear(Task):
    name = "place_near"
    min_objects = 2

    def sample(self, scene: SceneParams, rng: np.random.Generator) -> TaskInstance | None:
        if len(scene.objects) < 2:
            return None
        obj, target = (scene.objects[i] for i in rng.choice(len(scene.objects), 2, replace=False))
        instruction, phrasings = phrase(TEMPLATES, rng, 10, obj=obj.label, target=target.label)
        return TaskInstance(
            task=self.name,
            instruction=instruction,
            roles={"object": obj.name, "target": target.name},
            phrasings=phrasings,
            memories=[f"I picked up the {obj.label}.", f"The {obj.label} is next to the {target.label}."],
        )

    def script(self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator):
        obj = find(scene, instance.roles["object"])
        target = find(scene, instance.roles["target"])
        spot = self._free_spot(scene, obj, target, planner.workspace, rng)
        planner.subtask(f"pick up the {obj.label}")
        grasp = planner.pick(obj)
        planner.subtask(f"place the {obj.label} next to the {target.label}")
        release = np.array([spot[0], spot[1], grasp.height + rng.uniform(0.004, 0.015)])
        planner.place(release, grasp.tilt)
        if rng.random() < planner.cfg.return_rest_prob:
            planner.subtask("go back to the rest position")
            planner.transit(REST + rng.uniform(-3, 3, 6) * [1, 1, 1, 1, 1, 0])

    @staticmethod
    def _free_spot(scene: SceneParams, obj, target, radius, rng: np.random.Generator) -> np.ndarray:
        """A spot next to ``target`` that is reachable and clear of everything else."""
        others = [o for o in scene.objects if o.name not in (obj.name, target.name)]
        for _ in range(40):
            angle = rng.uniform(-np.pi, np.pi)
            distance = footprint(obj) + footprint(target) + rng.uniform(*GAP)
            spot = np.asarray(target.pos) + distance * np.array([np.cos(angle), np.sin(angle)])
            if not radius[0] <= np.linalg.norm(spot) <= radius[1]:
                continue
            clear = all(
                np.linalg.norm(spot - np.asarray(o.pos)) > footprint(obj) + footprint(o) + 0.01
                for o in others
            )
            if clear:
                return spot
        raise PlanningError("No free spot next to the target")

    def success(
        self, instance: TaskInstance, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo
    ) -> bool:
        obj = find(info.params, instance.roles["object"])
        target = find(info.params, instance.roles["target"])
        a = data.xpos[info.object_bodies[obj.name]][:2]
        b = data.xpos[info.object_bodies[target.name]][:2]
        near = np.linalg.norm(a - b) < footprint(obj) + footprint(target) + GAP[1] + NEAR
        return bool(near and resting_on_table(obj.name, model, data, info))
