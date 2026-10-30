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

"""Tasks: what to do in a scene, how to script it, and how to tell it worked.

A task is a ``Task`` subclass registered with ``@register``. It declares what the scene needs
(containers, objects), picks the objects it acts on, writes its instruction, scripts the motion
subtask by subtask with a ``Planner``, and checks success on the simulated state. Adding a task is one
new module in this package, imported below.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..motion import Planner
from ..scene import ObjectParams, SceneInfo, SceneParams


@dataclass
class TaskInstance:
    """One sampled task: its roles in the scene and its language."""

    task: str
    instruction: str
    # Scene item names by role, e.g. {"object": "object_0", "target": "container_0"}.
    roles: dict[str, str]
    # Other phrasings of the instruction, for ``task_aug`` rows.
    phrasings: list[str] = field(default_factory=list)
    # What the robot has done at the start of each subtask after the first, for ``memory`` rows.
    memories: list[str] = field(default_factory=list)


class Task(ABC):
    name: str
    # What the scene must contain.
    containers: int = 0
    min_objects: int = 1

    @abstractmethod
    def sample(self, scene: SceneParams, rng: np.random.Generator) -> TaskInstance | None:
        """Choose the objects to act on and phrase the instruction; None when the scene doesn't fit."""

    @abstractmethod
    def script(self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator):
        """Append the task's motion to ``planner``, opening each subtask with ``planner.subtask``."""

    @abstractmethod
    def success(
        self, instance: TaskInstance, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo
    ) -> bool:
        """Whether the final simulated state completes the task."""

    def check(
        self, mark: str, instance: TaskInstance, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo
    ) -> bool:
        """Mid-episode checks at the planner's marks: the object must be up once lifted."""
        if mark == "lifted":
            return is_lifted(instance.roles["object"], model, data, info)
        return True


TASKS: dict[str, Task] = {}


def register(cls: type[Task]) -> type[Task]:
    if cls.name in TASKS:
        raise ValueError(f"Task {cls.name!r} is already registered")
    TASKS[cls.name] = cls()
    return cls


# Shared helpers.


def find(scene: SceneParams, name: str):
    return next(item for item in [*scene.objects, *scene.containers] if item.name == name)


def footprint(obj: ObjectParams) -> float:
    """Radius of the object's footprint on the table."""
    if obj.shape == "cylinder":
        return obj.half_size[0]
    return float(np.hypot(obj.half_size[0], obj.half_size[1]))


def obstacle_top(
    scene: SceneParams, start: np.ndarray, end: np.ndarray, exclude: set[str], corridor: float = 0.05
) -> float:
    """Height of the tallest object or container whose footprint comes within ``corridor`` of the
    straight path from ``start`` to ``end`` on the table; zero when the way is clear."""
    start, end = np.asarray(start[:2], dtype=float), np.asarray(end[:2], dtype=float)
    path = end - start
    top = 0.0
    for item in [*scene.objects, *scene.containers]:
        if item.name in exclude:
            continue
        if isinstance(item, ObjectParams):
            height, radius = 2 * item.half_size[-1], footprint(item)
        else:
            height, radius = item.wall_height, np.sqrt(2) * (item.inner_half + item.thickness)
        s = np.clip(np.dot(np.asarray(item.pos[:2]) - start, path) / max(np.dot(path, path), 1e-9), 0, 1)
        if np.linalg.norm(start + s * path - np.asarray(item.pos[:2])) < radius + corridor:
            top = max(top, height)
    return top


def sentence(text: str) -> str:
    return text[0].upper() + text[1:] + ("" if text.endswith(("?", ".")) else ".")


def phrase(templates: list[str], rng: np.random.Generator, n: int, **names: str) -> tuple[str, list[str]]:
    """The instruction from one template, and up to ``n`` other templates as paraphrases."""
    order = rng.permutation(len(templates))
    texts = [sentence(templates[i].format(**names)) for i in order]
    return texts[0], texts[1 : n + 1]


def touches_gripper(name: str, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo) -> bool:
    geoms = set(info.geoms[name])
    jaws = {model.body("gripper").id, model.body("moving_jaw_so101_v1").id}
    for contact in data.contact[: data.ncon]:
        pair = (contact.geom1, contact.geom2)
        for a, b in (pair, pair[::-1]):
            if a in geoms and model.geom_bodyid[b] in jaws:
                return True
    return False


def is_lifted(name: str, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo) -> bool:
    body = info.object_bodies[name]
    tcp = data.site_xpos[model.site("gripperframe").id]
    obj = find(info.params, name)
    raised = data.xpos[body][2] > obj.half_size[-1] + 0.015
    held = np.linalg.norm(data.xpos[body] - tcp) < 0.06
    return bool(raised and held)


def resting_on_table(name: str, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo) -> bool:
    obj = find(info.params, name)
    body = info.object_bodies[name]
    return bool(
        abs(data.xpos[body][2] - obj.half_size[-1]) < 0.01 and not touches_gripper(name, model, data, info)
    )


from . import pick_lift, place_in, place_near  # noqa: E402, F401
