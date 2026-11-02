# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Ordered transfers with either a stationary reference or a moved reference inside the container."""

from itertools import permutations

import mujoco
import numpy as np

from ..config import GenerationConfig
from ..motion import Planner, PlanningError
from ..scene import SceneInfo, SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register, resting_on_table
from .collect_objects import CollectObjects, inside_container
from .place_near import GAP, MIN_START_GAP, NEAR, PlaceNear

TEMPLATES = [
    "first {first}, then {second}",
    "{first}, and after that {second}",
    "start by moving the {a} {dest_a}, then move the {b} {dest_b}",
    "move the {a} {dest_a} before moving the {b} {dest_b}",
]


@register
class OrderedTransfers(CollectObjects):
    name = "ordered_transfers"
    min_objects = 3

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        # A dependent variant puts both objects in the bin, so reserve collection-sized capacity.
        return super().scene_config(cfg)

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        if rng.random() < 0.5:
            return self.sample_inside_reference(scene, rng)
        container = scene.containers[0]
        options = [
            (inside, beside, anchor)
            for inside, beside, anchor in permutations(scene.objects, 3)
            if footprint(inside) < container.inner_half - 0.012
            and np.linalg.norm(np.subtract(beside.pos, anchor.pos)) - footprint(beside) - footprint(anchor)
            >= MIN_START_GAP
        ]
        if not options:
            return None
        inside, beside, anchor = options[rng.integers(len(options))]
        transfers = [
            Transfer("", inside.name, container.name, list(container.pos)),
            Transfer("", beside.name, anchor.name, [], "place_near"),
        ]
        if rng.random() < 0.5:
            transfers.reverse()
        for i, transfer in enumerate(transfers):
            transfer.id = f"transfer_{i}"
        a, b = [find(scene, t.object_name).label for t in transfers]
        destinations = [
            ("in" if t.kind == "place_in" else "next to") + " the " + find(scene, t.target_name).label
            for t in transfers
        ]
        instruction, phrasings = phrase(
            TEMPLATES,
            rng,
            10,
            a=a,
            b=b,
            dest_a=destinations[0],
            dest_b=destinations[1],
            first=f"put the {a} {destinations[0]}",
            second=f"put the {b} {destinations[1]}",
        )
        return TaskInstance(
            self.name,
            instruction,
            {
                "object": transfers[0].object_name,
                "object_1": transfers[1].object_name,
                "target": container.name,
                "anchor": anchor.name,
            },
            phrasings,
            transfers=transfers,
        )

    def sample_inside_reference(self, scene: SceneParams, rng: np.random.Generator) -> TaskInstance | None:
        instance = CollectObjects.sample(self, scene, rng)
        if instance is None:
            return None
        first, second = instance.transfers
        second.container_name = first.target_name
        second.target_name, second.kind = first.object_name, "place_near"
        a, b = find(scene, first.object_name), find(scene, second.object_name)
        container = find(scene, first.target_name)
        instruction, phrasings = phrase(
            TEMPLATES,
            rng,
            10,
            a=a.label,
            b=b.label,
            dest_a=f"in the {container.label}",
            dest_b=f"next to the {a.label}",
            first=f"put the {a.label} in the {container.label}",
            second=f"put the {b.label} next to the {a.label}",
        )
        instance.instruction, instance.phrasings = instruction, phrasings
        instance.roles["anchor"] = first.object_name
        return instance

    def route(
        self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> list[Transfer]:
        # Reserve both destinations before any execution. The bin is also excluded by _free_spot.
        for transfer in instance.transfers:
            if transfer.kind == "place_near" and transfer.container_name is None:
                transfer.position = PlaceNear._free_spot(
                    scene,
                    find(scene, transfer.object_name),
                    find(scene, transfer.target_name),
                    planner.workspace,
                    rng,
                ).tolist()
        return list(instance.transfers)

    def prepare_transfer(
        self, transfer: Transfer, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> None:
        if transfer.container_name is None:
            return
        observed = scene if planner.rollout is None else planner.rollout.scene_now()
        obj, reference = find(observed, transfer.object_name), find(observed, transfer.target_name)
        container = find(observed, transfer.container_name)
        c, s = np.cos(container.yaw), np.sin(container.yaw)
        rotation = np.array([[c, s], [-s, c]])
        room = container.inner_half_xy - footprint(obj) - 0.008
        candidates = [np.asarray(transfer.position)]
        for _ in range(100):
            angle = rng.uniform(-np.pi, np.pi)
            distance = footprint(obj) + footprint(reference) + rng.uniform(*GAP)
            candidates.append(np.asarray(reference.pos) + distance * np.array([np.cos(angle), np.sin(angle)]))
        for point in candidates:
            separation = np.linalg.norm(point - reference.pos)
            inside = np.all(np.abs(rotation @ (point - container.pos)) <= room)
            near = (
                footprint(obj) + footprint(reference) + GAP[0]
                <= separation
                <= footprint(obj) + footprint(reference) + GAP[1] + 0.015
            )
            reachable = planner.workspace[0] <= np.linalg.norm(point) <= planner.workspace[1]
            if inside and near and reachable:
                transfer.position = point.tolist()
                return
        raise PlanningError("No free reachable spot beside the observed reference inside its container")

    def completed_transfer_valid(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        transfer = next(t for t in instance.transfers if t.id == transfer_id)
        if transfer.kind == "place_in":
            return inside_container(
                transfer.object_name, transfer.target_name, model, data, info, during_motion=True
            )
        return self.transfer_success(transfer_id, instance, model, data, info)

    def transfer_success(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        transfer = next(t for t in instance.transfers if t.id == transfer_id)
        if transfer.kind == "place_in":
            return inside_container(transfer.object_name, transfer.target_name, model, data, info)
        obj, target = find(info.params, transfer.object_name), find(info.params, transfer.target_name)
        a, b = [data.xpos[info.object_bodies[n]][:2] for n in (obj.name, target.name)]
        separation = np.linalg.norm(a - b)
        near = separation < footprint(obj) + footprint(target) + GAP[1] + NEAR
        if transfer.container_name is not None:
            return bool(near and inside_container(obj.name, transfer.container_name, model, data, info))
        anchor_unchanged = np.linalg.norm(b - target.pos) <= 0.015
        return bool(near and anchor_unchanged and resting_on_table(obj.name, model, data, info))
