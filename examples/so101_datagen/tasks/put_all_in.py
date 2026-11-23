# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Put every graspable table object into one named container."""

from dataclasses import replace
from itertools import permutations

import numpy as np

from ..config import GenerationConfig
from ..motion import Planner
from ..scene import SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register
from .collect_objects import CollectObjects
from .sort_objects import SortObjects

TEMPLATES = [
    "put all the objects in the {target}",
    "collect everything on the table in the {target}",
    "move all the objects into the {target}",
    "put every object on the table inside the {target}",
]


@register
class PutAllIn(CollectObjects):
    name = "put_all_in"
    precise_container_layout = True

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        cfg = super().scene_config(cfg)
        return replace(cfg, objects=replace(cfg.objects, count=(2, 3)))

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        cfg = cfg or GenerationConfig()
        if len(scene.containers) != 1 or not 2 <= len(scene.objects) <= 3:
            return None
        container = scene.containers[0]
        c, s = np.cos(container.yaw), np.sin(container.yaw)
        rotation = np.array([[c, -s], [s, c]])
        radii = np.array([footprint(obj) for obj in scene.objects])
        room = container.inner_half_xy - radii[:, None] - 0.008
        if np.any(room <= 0):
            return None
        # Search the actual usable floor, preserving source object sizes. Central candidates and
        # randomized packings cover differently sized objects without forcing them against the rim.
        candidates = []
        for _ in range(200):
            placed = {}
            for i in np.argsort(-radii):
                locations = rng.uniform(-room[i], room[i], size=(60, 2))
                locations = np.vstack((locations, np.zeros((1, 2))))
                for local in locations:
                    point = np.asarray(container.pos) + rotation @ local
                    if not cfg.layout.radius[0] <= np.linalg.norm(point) <= cfg.layout.radius[1]:
                        continue
                    if all(
                        np.linalg.norm(local - other) >= radii[i] + radii[j] + 0.012
                        for j, other in placed.items()
                    ):
                        placed[i] = local
                        break
                else:
                    break
            if len(placed) == len(scene.objects):
                points = np.array([placed[i] for i in range(len(scene.objects))])
                # Prefer central packings with more room around the outside walls.
                candidates.append((float(np.sum(points**2)), points))
                if len(candidates) >= 8:
                    break
        if not candidates:
            return None
        points = min(candidates, key=lambda option: option[0])[1]
        transfers = [
            Transfer(
                f"transfer_{i}",
                obj.name,
                container.name,
                (np.asarray(container.pos) + rotation @ point).tolist(),
            )
            for i, (obj, point) in enumerate(zip(scene.objects, points, strict=True))
        ]
        instruction, phrasings = phrase(TEMPLATES, rng, 10, target=container.label)
        roles = {"object" if i == 0 else f"object_{i}": obj.name for i, obj in enumerate(scene.objects)}
        roles["target"] = container.name
        return TaskInstance(self.name, instruction, roles, phrasings, transfers=transfers)

    def route(
        self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> list[Transfer]:
        tcp, _ = planner.arm.tcp_pose(planner.q)

        def distance(route: tuple[Transfer, ...]) -> float:
            current = tcp[:2]
            total = 0.0
            for transfer in route:
                source = np.array(find(scene, transfer.object_name).pos)
                destination = np.array(transfer.position)
                total += float(np.linalg.norm(source - current) + np.linalg.norm(destination - source))
                current = destination
            return total

        return list(min(permutations(instance.transfers), key=distance))

    def prepare_transfer(
        self, transfer: Transfer, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> None:
        # Sorting uses the same observed-floor reservation contract, independent of its color rules.
        SortObjects.prepare_transfer(self, transfer, scene, planner, rng)
