# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Collect two named objects into reserved, non-overlapping positions in one container."""

from dataclasses import replace
from itertools import combinations

import mujoco
import numpy as np

from ..config import GenerationConfig
from ..errors import EpisodeError
from ..motion import Grasp, Objective, Planner, TransferResult
from ..randomization import uniform
from ..scene import ContainerParams, ObjectParams, SceneInfo, SceneParams
from ..so101 import REST
from . import Task, TaskInstance, Transfer, find, footprint, obstacle_top, phrase, register, touches_gripper

TEMPLATES = [
    "put the {a} and the {b} in the {target}",
    "collect the {a} and the {b} in the {target}",
    "move both the {a} and the {b} into the {target}",
    "place the {a} and the {b} inside the {target}",
]


def inside_container(
    name: str,
    target: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: SceneInfo,
    *,
    during_motion: bool = False,
) -> bool:
    obj, container = find(info.params, name), find(info.params, target)
    body, center = info.object_bodies[name], info.container_bodies[target]
    rotation = data.xmat[center].reshape(3, 3).T
    relative = rotation @ (data.xpos[body] - data.xpos[center])
    axes = rotation @ data.xmat[body].reshape(3, 3)
    if obj.shape == "cylinder":
        extent = obj.half_size[0] * np.linalg.norm(axes[:, :2], axis=1) + obj.half_size[1] * np.abs(
            axes[:, 2]
        )
    else:
        extent = np.abs(axes) @ np.array(obj.half_size)
    # During later transfers an object can roll or briefly touch the gripper/rim. Its centre must
    # stay in the bin; require its entire footprint and release again at the final success check.
    footprint_extent = 0.0 if during_motion else extent[:2]
    max_z = (
        container.wall_height + extent[2] + 0.015
        if during_motion
        else container.thickness + obj.half_size[-1] + 0.012
    )
    return bool(
        np.all(np.abs(relative[:2]) + footprint_extent < container.inner_half_xy - 0.001)
        and relative[2] < max_z
        and relative[2] + extent[2] > container.thickness - 0.005
        and (during_motion or not touches_gripper(name, model, data, info))
    )


@register
class CollectObjects(Task):
    name = "collect_objects"
    containers = 1
    min_objects = 2
    long_horizon = True

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        return replace(cfg, container=replace(cfg.container, inner_size=cfg.container.collection_inner_size))

    def completed_transfer_valid(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        transfer = next(t for t in instance.transfers if t.id == transfer_id)
        # The goal is containment, not a fixed landing point or zero contact during later motions.
        return inside_container(
            transfer.object_name, transfer.target_name, model, data, info, during_motion=True
        )

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        container = scene.containers[0]
        options = []
        for objects in combinations(scene.objects, 2):
            radii = np.array([footprint(o) for o in objects])
            room = container.inner_half - radii - 0.008
            if np.any(room <= 0):
                continue
            if container.style != "procedural":
                half = container.inner_half_xy
                rooms = half[0] - radii - 0.008
                separation = radii.sum() + 0.012
                if radii.max() + 0.008 < half[1] and separation <= rooms.sum():
                    distances = separation * rooms / rooms.sum()
                    offsets = np.array([[distances[0], 0], [-distances[1], 0]])
                    c, s = np.cos(container.yaw), np.sin(container.yaw)
                    rotation = np.array([[c, -s], [s, c]])
                    options.append((objects, np.asarray(container.pos) + offsets @ rotation.T))
                continue
            # Pack along a diagonal, keeping the pair central instead of aiming at the walls.
            for diagonal in ([1, 1], [1, -1]):
                reach = (radii.sum() + 0.012) / np.sqrt(2)
                if reach > room.sum():
                    continue
                distances = reach * room / room.sum()
                offsets = np.array([distances[0] * np.array(diagonal), -distances[1] * np.array(diagonal)])
                angle = container.yaw
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                points = np.asarray(container.pos) + offsets @ rotation.T
                options.append((objects, points))
        if not options:
            return None
        objects, points = options[rng.integers(len(options))]
        transfers = [
            Transfer(f"transfer_{i}", obj.name, container.name, point.tolist())
            for i, (obj, point) in enumerate(zip(objects, points, strict=True))
        ]
        instruction, phrasings = phrase(
            TEMPLATES, rng, 10, a=objects[0].label, b=objects[1].label, target=container.label
        )
        return TaskInstance(
            self.name,
            instruction,
            {"object": objects[0].name, "object_1": objects[1].name, "target": container.name},
            phrasings,
            transfers=transfers,
        )

    def script(self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator):
        route = self.route(instance, scene, planner, rng)
        instance.transfers[:] = route
        instance.objectives[:] = [
            objective
            for transfer in route
            for objective in (
                Objective(
                    f"{transfer.id}:pick",
                    f"pick up the {find(scene, transfer.object_name).label}",
                    transfer.object_name,
                    transfer.id,
                ),
                Objective(
                    f"{transfer.id}:place",
                    self.place_label(transfer, scene),
                    transfer.object_name,
                    transfer.id,
                ),
            )
        ]
        return_rest = rng.random() < planner.cfg.return_rest_prob
        if return_rest:
            instance.objectives.append(Objective("rest", "go back to the rest position"))
        planner.objectives = list(instance.objectives)
        for transfer in route:
            self.prepare_transfer(transfer, scene, planner, rng)
            obj = find(scene, transfer.object_name)
            container = (
                None
                if transfer.kind == "place_on_table"
                else find(scene, transfer.container_name or transfer.target_name)
            )
            planner.begin_transfer(transfer.id, obj.name, f"{transfer.id}:pick")

            def place(
                grasp: Grasp,
                transfer: Transfer = transfer,
                obj: ObjectParams = obj,
                container: ContainerParams | ObjectParams | None = container,
            ) -> None:
                planner.subtask(
                    self.place_label(transfer, scene),
                    memory=f"I picked up the {obj.label}.",
                    objective_id=f"{transfer.id}:place",
                )
                is_bin = transfer.kind == "place_in" or transfer.container_name is not None
                floor = container.thickness if is_bin else 0.0
                release = np.array([*transfer.position, floor + grasp.height + uniform(rng, (0.008, 0.015))])
                if grasp.style == "tangential":
                    rim = container.wall_height + 0.015 if is_bin else 0.025
                    release[2] = max(release[2], rim + grasp.height)
                # Refresh placed-object geometry before planning the next carry arc.
                observed = scene
                if planner.rollout is not None:
                    observed = planner.rollout.scene_now()
                top = obstacle_top(observed, find(observed, obj.name).pos, release, exclude={obj.name})
                clear = top + grasp.height
                planner.place(
                    release,
                    grasp.tilt,
                    hover_z=clear + uniform(rng, planner.cfg.carry_margin),
                    min_hover_z=clear + planner.cfg.min_carry_margin,
                )

            self.pick_and_place(obj, planner, place, preflight=True, objective_id=f"{transfer.id}:pick")
            if planner.rollout is not None:
                planner.dwell(0.2)
                planner.flush()
                if not planner.rollout.transfer_satisfied(transfer.id):
                    raise EpisodeError("transfer_success")
                planner.completed_transfers.append(TransferResult(transfer.id, obj.name, len(planner.frames)))
        if return_rest:
            planner.subtask(
                "go back to the rest position", memory="I placed both objects.", objective_id="rest"
            )
            planner.transit(REST + rng.uniform(-3, 3, 6) * [1, 1, 1, 1, 1, 0])

    def route(
        self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> list[Transfer]:
        tcp, _ = planner.arm.tcp_pose(planner.q)
        return sorted(
            instance.transfers,
            key=lambda t: np.linalg.norm(np.array(find(scene, t.object_name).pos) - tcp[:2]),
        )

    def prepare_transfer(
        self, transfer: Transfer, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> None:
        """Update a reserved destination against observed physics before beginning its transfer."""

    @staticmethod
    def place_label(transfer: Transfer, scene: SceneParams) -> str:
        if transfer.kind == "place_on_table":
            return f"put the {find(scene, transfer.object_name).label} on the table"
        relation = "in" if transfer.kind == "place_in" else "next to"
        return f"put the {find(scene, transfer.object_name).label} {relation} the {find(scene, transfer.target_name).label}"

    def transfer_success(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        transfer = next(t for t in instance.transfers if t.id == transfer_id)
        return inside_container(transfer.object_name, transfer.target_name, model, data, info)

    def success(
        self,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        return all(self.transfer_success(t.id, instance, model, data, info) for t in instance.transfers)
