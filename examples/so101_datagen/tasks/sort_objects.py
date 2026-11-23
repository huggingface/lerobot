# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Sort three uniquely named primitives from two color groups into two containers."""

from dataclasses import replace
from itertools import combinations, permutations

import numpy as np

from ..config import GenerationConfig
from ..motion import Planner, PlanningError
from ..randomization import PALETTE, _confusable
from ..scene import Look, SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register
from .collect_objects import CollectObjects

TEMPLATES = [
    "sort the objects: put the {color_a} objects in the {bin_a} and the {color_b} object in the {bin_b}",
    "put the {a} and the {b} in the {bin_a}, and the {c} in the {bin_b}",
    "sort by color: {color_a} goes in the {bin_a}, {color_b} goes in the {bin_b}",
    "collect the {color_a} objects in the {bin_a} and move the {color_b} object into the {bin_b}",
]


@register
class SortObjects(CollectObjects):
    name = "sort_objects"
    containers = 2
    min_objects = 3
    precise_container_layout = True

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        # Color groups require known, uniform colors; scanned objects retain their native materials
        # in other tasks. Three objects keep every member of each requested group explicit.
        return replace(
            cfg,
            objects=replace(cfg.objects, count=(3, 3), mesh_prob=0.0),
            container=replace(cfg.container, inner_size=cfg.container.sorting_inner_size),
        )

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        cfg = cfg or GenerationConfig()
        if (
            len(scene.objects) != 3
            or len(scene.containers) != 2
            or len({c.label for c in scene.containers}) != 2
            or len({c.color for c in scene.containers}) != 2
        ):
            return None
        # Give the repeated color two distinct shape nouns, avoiding ambiguous grasp annotations.
        options = []
        for pair in combinations(scene.objects, 2):
            if pair[0].label.split(" ", 1)[1] == pair[1].label.split(" ", 1)[1]:
                continue
            single = next(obj for obj in scene.objects if obj not in pair)
            for first, second in (scene.containers, scene.containers[::-1]):
                pair_instance = super().sample(
                    replace(scene, objects=list(pair), containers=[first]), rng, cfg
                )
                if pair_instance is None:
                    continue
                radius = footprint(single)
                if np.any(second.inner_half_xy <= radius + 0.008):
                    continue
                transfers = pair_instance.transfers + [
                    Transfer("transfer_2", single.name, second.name, list(second.pos))
                ]
                if any(
                    not cfg.layout.radius[0] <= np.linalg.norm(t.position) <= cfg.layout.radius[1]
                    for t in transfers
                ):
                    continue
                options.append((pair, single, first, second, transfers))
        if not options:
            return None
        pair, single, first, second, transfers = options[rng.integers(len(options))]
        available = [
            color
            for color in PALETTE
            if color
            not in ({c.color for c in scene.containers} | _confusable({c.color for c in scene.containers}))
        ]
        pairs = [(a, b) for a in available for b in available if b not in _confusable({a}) and a != b]
        if not pairs:
            return None
        color_a, color_b = pairs[rng.integers(len(pairs))]
        for obj, color in ((pair[0], color_a), (pair[1], color_a), (single, color_b)):
            obj.label = f"{color} {obj.label.split(' ', 1)[1]}"
            obj.color = color
            rgba = np.clip(
                np.array(PALETTE[color])
                + rng.uniform(-cfg.objects.color_jitter, cfg.objects.color_jitter, 3),
                0.05,
                1,
            )
            obj.look = Look(rgba=[*rgba.tolist(), 1.0])
        instruction, phrasings = phrase(
            TEMPLATES,
            rng,
            10,
            a=pair[0].label,
            b=pair[1].label,
            c=single.label,
            color_a=color_a,
            color_b=color_b,
            bin_a=first.label,
            bin_b=second.label,
        )
        return TaskInstance(
            self.name,
            instruction,
            {
                "object": pair[0].name,
                "object_1": pair[1].name,
                "object_2": single.name,
                "target": first.name,
                "target_1": second.name,
            },
            phrasings,
            transfers=transfers,
        )

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
        observed = scene if planner.rollout is None else planner.rollout.scene_now()
        obj = find(observed, transfer.object_name)
        container = find(observed, transfer.target_name)
        radius = footprint(obj)
        room = container.inner_half_xy - radius - 0.008
        c, s = np.cos(container.yaw), np.sin(container.yaw)
        rotation = np.array([[c, -s], [s, c]])

        def free(point: np.ndarray) -> bool:
            local = rotation.T @ (point - container.pos)
            return bool(
                np.all(np.abs(local) <= room)
                and planner.workspace[0] <= np.linalg.norm(point) <= planner.workspace[1]
                and all(
                    np.linalg.norm(point - other.pos) > radius + footprint(other) + 0.012
                    for other in observed.objects
                    if other.name != obj.name
                )
            )

        destination = np.array(transfer.position)
        if free(destination):
            return
        # Previously deposited objects may roll. Refresh the remaining free floor from observed poses.
        if np.all(room > 0):
            for _ in range(100):
                point = np.asarray(container.pos) + rotation @ rng.uniform(-room, room)
                if free(point):
                    transfer.position = point.tolist()
                    return
        raise PlanningError("No free sorting destination after observing earlier transfers")
