# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Exchange two objects' initial positions using one reserved free table buffer."""

from dataclasses import replace
from itertools import combinations

import mujoco
import numpy as np

from ..config import GenerationConfig
from ..motion import Planner, PlanningError
from ..randomization import _corners, _in_view
from ..scene import ObjectParams, SceneInfo, SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register, resting_on_table
from .collect_objects import CollectObjects

TEMPLATES = [
    "swap the positions of the {a} and the {b}",
    "exchange the {a} and the {b}, using a free spot on the table while you move them",
    "put the {a} where the {b} started, and the {b} where the {a} started",
    "switch the places of the {a} and the {b}",
]
POSITION_TOLERANCE = 0.02
DESTINATION_GAP = 0.012


def table_goal(transfer: Transfer, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo) -> bool:
    """A released object must occupy its reserved center and physically rest on the table."""
    body = info.object_bodies[transfer.object_name]
    geoms = set(info.geoms[transfer.object_name])
    table_contact = any(
        (contact.geom1 in geoms and contact.geom2 == info.table_geom)
        or (contact.geom2 in geoms and contact.geom1 == info.table_geom)
        for contact in data.contact[: data.ncon]
    )
    return bool(
        np.linalg.norm(data.xpos[body, :2] - transfer.position) <= POSITION_TOLERANCE
        and table_contact
        and resting_on_table(transfer.object_name, model, data, info)
    )


def free_table_goal(
    point: np.ndarray,
    obj: ObjectParams,
    scene: SceneParams,
    workspace: tuple[float, float],
    aspect: float,
    *,
    exclude: set[str],
) -> bool:
    radius = footprint(obj)
    table = scene.background
    return bool(
        workspace[0] <= np.linalg.norm(point) <= workspace[1]
        and np.all(np.abs(point - table.table_center) + radius < np.array(table.table_half) - 0.01)
        and all(
            np.linalg.norm(point - other.pos) > radius + footprint(other) + DESTINATION_GAP
            for other in scene.objects
            if other.name not in exclude
        )
        and _in_view(scene.front_camera, _corners(replace(obj, pos=point.tolist(), support_height=0)), aspect)
    )


@register
class SwapObjects(CollectObjects):
    name = "swap_objects"
    containers = 0
    min_objects = 2

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        return cfg

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        cfg = cfg or GenerationConfig()
        options = []
        aspect = cfg.width / cfg.height
        for pair in combinations(scene.objects, 2):
            names = {obj.name for obj in pair}
            # Both objects must fit at the other's source position, including distractor clearance.
            if not all(
                free_table_goal(np.array(other.pos), obj, scene, cfg.layout.radius, aspect, exclude=names)
                for obj, other in (pair, pair[::-1])
            ):
                continue
            if np.linalg.norm(np.array(pair[0].pos) - pair[1].pos) <= (
                footprint(pair[0]) + footprint(pair[1]) + 2 * POSITION_TOLERANCE
            ):
                continue
            for first, second in (pair, pair[::-1]):
                # A margin around the fixed source positions keeps the buffer out of the swap goals
                # even if the smaller object leaves room that its larger partner later occupies.
                radius = max(footprint(first), footprint(second))
                for _ in range(120):
                    distance = float(rng.uniform(*cfg.layout.radius))
                    angle = np.deg2rad(rng.uniform(*cfg.layout.angle))
                    point = distance * np.array([np.cos(angle), np.sin(angle)])
                    if not free_table_goal(
                        point, first, scene, cfg.layout.radius, aspect, exclude={first.name}
                    ):
                        continue
                    if any(
                        np.linalg.norm(point - other.pos) <= footprint(first) + radius + 0.025
                        for other in pair
                    ):
                        continue
                    options.append((first, second, point))
                    break
        if not options:
            return None
        first, second, buffer = options[rng.integers(len(options))]
        instruction, phrasings = phrase(TEMPLATES, rng, 10, a=first.label, b=second.label)
        return TaskInstance(
            self.name,
            instruction,
            {"object": first.name, "object_1": second.name},
            phrasings,
            transfers=[
                Transfer("transfer_0", first.name, "table", buffer.tolist(), "place_on_table"),
                Transfer("transfer_1", second.name, "table", list(first.pos), "place_on_table"),
                Transfer("transfer_2", first.name, "table", list(second.pos), "place_on_table"),
            ],
        )

    def route(
        self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> list[Transfer]:
        # The buffer dependency fixes the three intentional transfers and their completion order.
        return list(instance.transfers)

    def prepare_transfer(
        self, transfer: Transfer, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> None:
        observed = scene if planner.rollout is None else planner.rollout.scene_now()
        # Check before changing active_transfer_id: an accidental loss of the temporary goal
        # must not be mistaken for its intentional retirement by the second pick of A.
        if (
            transfer.id == "transfer_2"
            and planner.rollout is not None
            and not planner.rollout.transfer_satisfied("transfer_0")
        ):
            raise PlanningError("Swap buffer goal was lost before its intentional reuse")
        obj = find(observed, transfer.object_name)
        aspect = (
            4 / 3
            if planner.rollout is None
            else planner.rollout.episode.config.width / planner.rollout.episode.config.height
        )
        if not free_table_goal(
            np.array(transfer.position), obj, observed, planner.workspace, aspect, exclude={obj.name}
        ):
            # Original swap destinations are fixed; adjusting a blocked goal would change the task.
            raise PlanningError("Reserved swap destination is no longer free")

    @staticmethod
    def place_label(transfer: Transfer, scene: SceneParams) -> str:
        obj = find(scene, transfer.object_name)
        if transfer.id == "transfer_0":
            return f"put the {obj.label} in the temporary spot on the table"
        other = next(
            other
            for other in scene.objects
            if other.name != obj.name and np.linalg.norm(np.array(other.pos) - transfer.position) < 1e-8
        )
        return f"put the {obj.label} where the {other.label} started"

    def transfer_success(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        transfer = next(t for t in instance.transfers if t.id == transfer_id)
        return table_goal(transfer, model, data, info)

    def completed_transfer_valid(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        if transfer_id == "transfer_0" and instance.active_transfer_id == "transfer_2":
            # A is intentionally lifted out of the buffer. B's completed swap goal stays protected.
            return True
        return self.transfer_success(transfer_id, instance, model, data, info)

    def success(
        self,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        return all(
            self.transfer_success(name, instance, model, data, info) for name in ("transfer_1", "transfer_2")
        )
