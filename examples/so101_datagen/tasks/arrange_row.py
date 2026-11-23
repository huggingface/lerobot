# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Arrange three named objects left-to-right as viewed by the static front camera."""

from dataclasses import replace
from itertools import combinations, permutations

import mujoco
import numpy as np

from ..config import GenerationConfig
from ..motion import Planner, PlanningError
from ..randomization import _corners, _in_view
from ..scene import ObjectParams, SceneInfo, SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register, resting_on_table
from .collect_objects import CollectObjects

POSITION_TOLERANCE = 0.02
ROW_GAP = 0.025
TEMPLATES = [
    "arrange the {a}, the {b} and the {c} in a row, in that order from left to right",
    "make a row with the {a} on the left, the {b} in the middle and the {c} on the right",
    "make a left-to-right row ordered as the {a}, the {b} and the {c}",
    "line up the {a}, the {b} and the {c} from left to right",
]


def front_right(scene: SceneParams) -> np.ndarray:
    """The fixed front camera's image-right direction projected onto the tabletop."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(scene.front_camera.quat, dtype=float))
    right = rotation.reshape(3, 3)[:2, 0]
    return right / np.linalg.norm(right)


def front_horizontal(scene: SceneParams, points: np.ndarray) -> np.ndarray:
    """Perspective image-x coordinates, used to validate actual left-to-right order."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(scene.front_camera.quat, dtype=float))
    local = (points - scene.front_camera.pos) @ rotation.reshape(3, 3)
    return local[:, 0] / -local[:, 2]


@register
class ArrangeRow(CollectObjects):
    name = "arrange_row"
    containers = 0
    min_objects = 3

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        return cfg

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        cfg = cfg or GenerationConfig()
        right = front_right(scene)
        groups = list(combinations(scene.objects, 3))
        if not groups:
            return None
        for _ in range(240):
            group = groups[rng.integers(len(groups))]
            objects = [group[i] for i in rng.permutation(3)]
            radii = np.array([footprint(obj) for obj in objects])
            distance = rng.uniform(*cfg.layout.radius)
            angle = np.deg2rad(rng.uniform(*cfg.layout.angle))
            center = distance * np.array([np.cos(angle), np.sin(angle)])
            offsets = np.array([-(radii[0] + radii[1] + ROW_GAP), 0, radii[1] + radii[2] + ROW_GAP])
            points = center + offsets[:, None] * right
            if not all(
                self._free_destination(point, obj, scene, cfg.layout.radius, cfg.width / cfg.height)
                and np.linalg.norm(point - obj.pos) > POSITION_TOLERANCE + 0.02
                for point, obj in zip(points, objects, strict=True)
            ):
                continue
            positions = np.array(
                [[*point, obj.half_size[-1]] for obj, point in zip(objects, points, strict=True)]
            )
            if np.any(np.diff(front_horizontal(scene, positions)) <= 0):
                continue
            instruction, phrasings = phrase(
                TEMPLATES, rng, 10, a=objects[0].label, b=objects[1].label, c=objects[2].label
            )
            return TaskInstance(
                self.name,
                instruction,
                {"object": objects[0].name, "object_1": objects[1].name, "object_2": objects[2].name},
                phrasings,
                transfers=[
                    Transfer(f"transfer_{i}", obj.name, "table", point.tolist(), "place_on_table")
                    for i, (obj, point) in enumerate(zip(objects, points, strict=True))
                ],
            )
        return None

    @staticmethod
    def _free_destination(
        point: np.ndarray,
        obj: ObjectParams,
        scene: SceneParams,
        workspace: tuple[float, float],
        aspect: float,
    ) -> bool:
        table = scene.background
        return bool(
            workspace[0] <= np.linalg.norm(point) <= workspace[1]
            and np.all(
                np.abs(point - table.table_center) + footprint(obj) < np.array(table.table_half) - 0.01
            )
            and all(
                np.linalg.norm(point - other.pos) > footprint(obj) + footprint(other) + 0.012
                for other in scene.objects
                if other.name != obj.name
            )
            and _in_view(
                scene.front_camera, _corners(replace(obj, pos=point.tolist(), support_height=0)), aspect
            )
        )

    def route(
        self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> list[Transfer]:
        tcp, _ = planner.arm.tcp_pose(planner.q)

        def distance(route: tuple[Transfer, ...]) -> float:
            current, total = tcp[:2], 0.0
            for transfer in route:
                source = np.array(find(scene, transfer.object_name).pos)
                point = np.array(transfer.position)
                total += float(np.linalg.norm(source - current) + np.linalg.norm(point - source))
                current = point
            return total

        return list(min(permutations(instance.transfers), key=distance))

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
        # The row's centers are explicit goals: do not silently move them during recovery.
        if not self._free_destination(np.array(transfer.position), obj, observed, planner.workspace, aspect):
            raise PlanningError("Reserved row destination is occupied or unreachable")

    @staticmethod
    def place_label(transfer: Transfer, scene: SceneParams) -> str:
        slot = {"transfer_0": "left", "transfer_1": "middle", "transfer_2": "right"}[transfer.id]
        return f"put the {find(scene, transfer.object_name).label} in the {slot} position of the row"

    def transfer_success(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        transfer = next(t for t in instance.transfers if t.id == transfer_id)
        name = transfer.object_name
        body = info.object_bodies[name]
        at_goal = np.linalg.norm(data.xpos[body, :2] - transfer.position) <= POSITION_TOLERANCE
        supported = any(
            info.table_geom in (contact.geom1, contact.geom2)
            and any(geom in info.geoms[name] for geom in (contact.geom1, contact.geom2))
            for contact in data.contact[: data.ncon]
        )
        return bool(at_goal and supported and resting_on_table(name, model, data, info))

    def completed_transfer_valid(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        return self.transfer_success(transfer_id, instance, model, data, info)

    def success(
        self, instance: TaskInstance, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo
    ) -> bool:
        if not super().success(instance, model, data, info):
            return False
        ordered = sorted(instance.transfers, key=lambda transfer: transfer.id)
        objects = [find(info.params, transfer.object_name) for transfer in ordered]
        positions = np.array([data.xpos[info.object_bodies[obj.name]] for obj in objects])
        if np.any(np.diff(front_horizontal(info.params, positions)) <= 0):
            return False
        # Conservative bounding footprints also reject interleaved or touching elongated objects.
        return all(
            np.linalg.norm(positions[j, :2] - positions[i, :2])
            > footprint(objects[i]) + footprint(objects[j])
            for i, j in combinations(range(3), 2)
        )
