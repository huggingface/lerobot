# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Clear two objects from a shallow tray onto separate free table positions."""

from dataclasses import replace
from itertools import combinations

import mujoco
import numpy as np

from ..config import GenerationConfig
from ..motion import Planner, PlanningError
from ..randomization import _corners, _in_view
from ..scene import ContainerParams, SceneInfo, SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register, resting_on_table
from .collect_objects import CollectObjects

TEMPLATES = [
    "take the {a} and the {b} out of the {tray} and put them on the table",
    "clear the {tray} by moving the {a} and the {b} onto the table",
    "empty the {tray}: move the {a} and the {b} to separate spots on the table",
    "move both the {a} and the {b} from the {tray} onto the table",
]


def distance_from_container(point: np.ndarray, container: ContainerParams) -> float:
    c, s = np.cos(container.yaw), np.sin(container.yaw)
    local = np.array([[c, s], [-s, c]]) @ (np.asarray(point) - container.pos)
    return float(np.linalg.norm(np.maximum(np.abs(local) - container.outer_half_xy, 0)))


@register
class ClearContainer(CollectObjects):
    name = "clear_container"
    precise_container_layout = True

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        cfg = super().scene_config(cfg)
        return replace(cfg, container=replace(cfg.container, wall_height=cfg.container.clearing_wall_height))

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        cfg = cfg or GenerationConfig()
        container = scene.containers[0]
        c, s = np.cos(container.yaw), np.sin(container.yaw)
        rotation = np.array([[c, -s], [s, c]])
        # Keep every unrelated object clearly outside the tray walls from the start.
        if any(
            distance_from_container(np.array(obj.pos), container) <= footprint(obj) + 0.012
            for obj in scene.objects
        ):
            return None
        options = []
        for objects in combinations(scene.objects, 2):
            radii = np.array([footprint(obj) for obj in objects])
            room = container.inner_half_xy - radii[:, None] - 0.012
            if np.any(room <= 0):
                continue
            # Pack along several directions, including tangential ones near the reachable boundary.
            for angle in np.arange(16) * np.pi / 8:
                direction = np.array([np.cos(angle), np.sin(angle)])
                capacity = np.min(room / np.maximum(np.abs(direction), 1e-9), axis=1)
                separation = radii.sum() + 0.03
                if separation > capacity.sum():
                    continue
                distances = separation * capacity / capacity.sum()
                offsets = np.array([direction * distances[0], -direction * distances[1]])
                points = np.asarray(container.pos) + offsets @ rotation.T
                if all(
                    cfg.layout.radius[0] <= np.linalg.norm(point) <= cfg.layout.radius[1] for point in points
                ):
                    options.append((objects, points))
        if not options:
            return None
        objects, points = options[rng.integers(len(options))]
        destinations = [list(obj.pos) for obj in objects]
        for obj, point in zip(objects, points, strict=True):
            obj.pos = point.tolist()
            obj.support_height = container.thickness
        instruction, phrasings = phrase(
            TEMPLATES, rng, 10, a=objects[0].label, b=objects[1].label, tray=container.label
        )
        return TaskInstance(
            self.name,
            instruction,
            {"object": objects[0].name, "object_1": objects[1].name, "target": container.name},
            phrasings,
            transfers=[
                Transfer(f"transfer_{i}", obj.name, "table", destination, "place_on_table")
                for i, (obj, destination) in enumerate(zip(objects, destinations, strict=True))
            ],
        )

    def prepare_transfer(
        self, transfer: Transfer, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> None:
        observed = scene if planner.rollout is None else planner.rollout.scene_now()
        obj = find(observed, transfer.object_name)
        aspect = (
            4 / 3
            if planner.rollout is None
            else planner.rollout.episode.config.width / planner.rollout.episode.config.height
        )

        def free(point: np.ndarray) -> bool:
            table = observed.background
            return bool(
                planner.workspace[0] <= np.linalg.norm(point) <= planner.workspace[1]
                and np.all(
                    np.abs(point - table.table_center) + footprint(obj) < np.array(table.table_half) - 0.01
                )
                and all(
                    distance_from_container(point, c) > footprint(obj) + 0.012 for c in observed.containers
                )
                and all(
                    np.linalg.norm(point - other.pos) > footprint(obj) + footprint(other) + 0.012
                    for other in observed.objects
                    if other.name != obj.name
                )
                and _in_view(
                    observed.front_camera,
                    _corners(replace(obj, pos=point.tolist(), support_height=0)),
                    aspect,
                )
            )

        destination = np.array(transfer.position)
        if free(destination):
            return
        # A previously placed object may roll into a reserved spot. Keep recovery local to that goal.
        for _ in range(100):
            angle, distance = rng.uniform(-np.pi, np.pi), rng.uniform(0.015, 0.08)
            candidate = destination + distance * np.array([np.cos(angle), np.sin(angle)])
            if free(candidate):
                transfer.position = candidate.tolist()
                return
        raise PlanningError("No free table destination after observing earlier placements")

    def completed_transfer_valid(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
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
        obj = find(info.params, transfer.object_name)
        container = find(info.params, instance.roles["target"])
        point = data.xpos[info.object_bodies[obj.name], :2]
        table = info.params.background
        fits_table = np.all(
            np.abs(point - table.table_center) + footprint(obj) < np.array(table.table_half) - 0.01
        )
        outside = distance_from_container(point, container) > footprint(obj) + 0.008
        table_contact = any(
            info.table_geom in (contact.geom1, contact.geom2)
            and any(g in info.geoms[obj.name] for g in (contact.geom1, contact.geom2))
            for contact in data.contact[: data.ncon]
        )
        return bool(
            fits_table and outside and table_contact and resting_on_table(obj.name, model, data, info)
        )

    def success(
        self,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        if not super().success(instance, model, data, info):
            return False
        container = find(info.params, instance.roles["target"])
        c, s = np.cos(container.yaw), np.sin(container.yaw)
        rotation = np.array([[c, s], [-s, c]])
        for obj in info.params.objects:
            pos = data.xpos[info.object_bodies[obj.name]]
            local = rotation @ (pos[:2] - container.pos)
            overlap = np.linalg.norm(np.maximum(np.abs(local) - container.inner_half_xy, 0)) < footprint(obj)
            if overlap and pos[2] - obj.half_size[-1] < container.wall_height:
                return False
        # Bounding circles reserve generous landing space, but can overlap for two separate elongated
        # meshes. Judge the physical result by table support and actual inter-object contacts.
        first, second = [set(info.geoms[t.object_name]) for t in instance.transfers]
        return not any(
            (contact.geom1 in first and contact.geom2 in second)
            or (contact.geom2 in first and contact.geom1 in second)
            for contact in data.contact[: data.ncon]
        )
