# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Clear a physically obstructed grasp corridor before placing its target in a bin."""

from collections.abc import Callable
from dataclasses import replace
from itertools import permutations

import mujoco
import numpy as np

from ..config import GenerationConfig
from ..motion import Grasp, Planner, PlanningError
from ..randomization import _corners, _in_view
from ..render import StateRenderer
from ..scene import ObjectParams, SceneInfo, SceneParams
from . import TaskInstance, Transfer, find, footprint, phrase, register, resting_on_table
from .clear_container import distance_from_container
from .collect_objects import CollectObjects, inside_container

TEMPLATES = [
    "move the {blocker} out of the way, then put the {object} in the {target}",
    "clear the {blocker} away from the {object}, then place the {object} in the {target}",
    "first move the {blocker} aside, then put the {object} into the {target}",
    "move the {blocker} to a free spot so you can put the {object} in the {target}",
]


def projected_half_extent(obj: ObjectParams, direction: np.ndarray, yaw: float | None = None) -> float:
    """Support of a conservative object bound along a horizontal separating axis."""
    if obj.shape == "cylinder":
        return float(obj.half_size[0])
    angle = obj.yaw if yaw is None else yaw
    c, s = np.cos(angle), np.sin(angle)
    local = np.array([[c, s], [-s, c]]) @ direction
    return float(np.abs(local) @ np.asarray(obj.half_size[:2]))


def swept_gripper_clearance(
    planner: Planner,
    blocker_name: str,
    frames: list[np.ndarray],
    *,
    qpos: np.ndarray,
    info: SceneInfo,
    restore_initial: bool,
    initial_pose: list[float] | np.ndarray | None = None,
) -> float:
    """Surface clearance of the actual selected open-gripper path against one blocker.

    Restoring just the blocker's original free-joint pose is a counterfactual collision test, not a
    physics reset. The ongoing episode and all completed transfers remain untouched.
    """
    model = planner.arm.model
    qpos = qpos.copy()
    body = info.object_bodies[blocker_name]
    if restore_initial:
        address = model.jnt_qposadr[model.body_jntadr[body]]
        if initial_pose is None:
            raise ValueError("Counterfactual checks require the measured initial blocker pose")
        qpos[address : address + 7] = initial_pose
    gripper_bodies = {model.body("gripper").id, model.body("moving_jaw_so101_v1").id}
    gripper_geoms = [
        geom
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] in gripper_bodies
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    ]
    blocker_geoms = [
        geom for geom in info.geoms[blocker_name] if model.geom_contype[geom] or model.geom_conaffinity[geom]
    ]
    data = mujoco.MjData(model)
    minimum = np.inf
    # These are real planner samples from the open approach, including both endpoints.
    for index in np.unique(np.linspace(0, len(frames) - 1, min(32, len(frames))).astype(int)):
        data.qpos[:] = qpos
        data.qpos[planner.arm.qadr] = planner.arm.to_ctrl(frames[index])
        mujoco.mj_forward(model, data)
        for a in gripper_geoms:
            for b in blocker_geoms:
                minimum = min(minimum, mujoco.mj_geomDistance(model, data, a, b, 0.2, None))
    return float(minimum)


@register
class UnblockThenPlace(CollectObjects):
    name = "unblock_then_place"
    min_objects = 2
    precise_container_layout = True

    def scene_config(self, cfg: GenerationConfig) -> GenerationConfig:
        # Only the target goes in the bin; preserve single-object container dimensions.
        return replace(cfg, objects=replace(cfg.objects, count=(2, 2)))

    def sample(
        self, scene: SceneParams, rng: np.random.Generator, cfg: GenerationConfig | None = None
    ) -> TaskInstance | None:
        cfg = cfg or GenerationConfig()
        container = scene.containers[0]
        options = []
        for target, blocker in permutations(scene.objects, 2):
            if footprint(target) + 0.01 >= min(container.inner_half_xy):
                continue
            radial = np.asarray(target.pos) / np.linalg.norm(target.pos)
            for angle in (-0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7):
                c, s = np.cos(angle), np.sin(angle)
                direction = np.array([[c, -s], [s, c]]) @ radial
                yaw = blocker.yaw
                if blocker.shape == "box":
                    yaw = float(
                        np.arctan2(direction[1], direction[0])
                        + (np.pi / 2 if blocker.half_size[0] <= blocker.half_size[1] else 0)
                    )
                gap = float(rng.uniform(0.005, 0.010))
                separation = (
                    projected_half_extent(target, direction)
                    + projected_half_extent(blocker, direction, yaw)
                    + gap
                )
                position = np.asarray(target.pos) - separation * direction
                radius = footprint(blocker)
                if not cfg.layout.radius[0] <= np.linalg.norm(position) <= cfg.layout.radius[1]:
                    continue
                if np.linalg.norm(position - blocker.pos) < 0.07:
                    continue
                if any(
                    np.linalg.norm(position - other.pos) <= radius + footprint(other) + 0.008
                    for other in scene.objects
                    if other.name not in (target.name, blocker.name)
                ):
                    continue
                if any(distance_from_container(position, c) <= radius + 0.012 for c in scene.containers):
                    continue
                table = scene.background
                if np.any(
                    np.abs(position - table.table_center) + radius >= np.array(table.table_half) - 0.01
                ):
                    continue
                if not _in_view(
                    scene.front_camera,
                    _corners(replace(blocker, pos=position.tolist())),
                    cfg.width / cfg.height,
                ):
                    continue
                options.append((target, blocker, position, yaw, gap))
        if not options:
            return None
        target, blocker, position, yaw, gap = options[rng.integers(len(options))]
        parking = list(blocker.pos)
        blocker.pos = position.tolist()
        blocker.yaw = yaw
        instruction, phrasings = phrase(
            TEMPLATES, rng, 10, blocker=blocker.label, object=target.label, target=container.label
        )
        return TaskInstance(
            self.name,
            instruction,
            {
                "object": blocker.name,
                "object_1": target.name,
                "blocker": blocker.name,
                "target_object": target.name,
                "target": container.name,
            },
            phrasings,
            transfers=[
                Transfer("clear_blocker", blocker.name, "table", parking, "place_on_table"),
                Transfer("place_target", target.name, container.name, list(container.pos)),
            ],
            evidence={
                "dependency_checked": False,
                "target_tilt_preference_degrees": [35.0, 60.0],
                "target_inclined_tilt_preference_degrees": [45.0, 75.0],
                "target_inclined_tilt_probability": 0.3,
                "initial_surface_gap_lower_bound": gap,
            },
        )

    def route(
        self, instance: TaskInstance, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> list[Transfer]:
        if planner.rollout is None:
            raise PlanningError("Unblocking requires live physics")
        rollout = planner.rollout
        cfg = rollout.episode.config
        body = rollout.episode.info.object_bodies[instance.roles["blocker"]]
        address = rollout.model.jnt_qposadr[rollout.model.body_jntadr[body]]
        instance.evidence["initial_blocker_pose"] = rollout.data.qpos[address : address + 7].tolist()
        renderer = StateRenderer(rollout.model, cfg.height, cfg.width)
        try:
            renderer.set_state(rollout.data.qpos)
            visible = renderer.visible("front", rollout.episode.info, cfg.annotation.min_visible_pixels)
            if not {instance.roles["blocker"], instance.roles["target_object"]}.issubset(visible):
                raise PlanningError("Both the blocker and target must initially be visible")
        finally:
            renderer.close()
        instance.evidence["initial_front_objects_visible"] = True
        return list(instance.transfers)

    def prepare_transfer(
        self, transfer: Transfer, scene: SceneParams, planner: Planner, rng: np.random.Generator
    ) -> None:
        observed = planner.rollout.scene_now()
        obj = find(observed, transfer.object_name)
        if transfer.kind == "place_on_table":
            point = np.asarray(transfer.position)
            if any(
                np.linalg.norm(point - other.pos) <= footprint(obj) + footprint(other) + 0.012
                for other in observed.objects
                if other.name != obj.name
            ):
                raise PlanningError("The blocker's reserved parking spot is occupied")
        else:
            original = find(scene, transfer.object_name)
            displacement = float(np.linalg.norm(np.asarray(obj.pos) - original.pos))
            planner.rollout.episode.instance.evidence["target_displacement_before_pick"] = displacement
            if displacement > 0.01:
                raise PlanningError("Clearing the blocker disturbed the constrained target")
            target = find(observed, transfer.target_name)
            if np.any(
                np.abs(np.asarray(transfer.position) - target.pos) + footprint(obj) + 0.008
                >= target.inner_half_xy
            ):
                raise PlanningError("The target no longer fits its destination")

    def pick_and_place(
        self,
        obj: ObjectParams,
        planner: Planner,
        place: Callable[[Grasp], None],
        *,
        preflight: bool = False,
        objective_id: str | None = None,
    ) -> None:
        instance = planner.rollout.episode.instance
        if obj.name != instance.roles["target_object"]:
            return super().pick_and_place(obj, planner, place, preflight=preflight, objective_id=objective_id)
        rollout = planner.rollout

        def checked_place(grasp: Grasp) -> None:
            frames = planner.frames[planner.marks["looked"] : planner.marks["grasp_approach_end"] + 1]
            if not frames:
                raise PlanningError("The target needs a checked approach")
            counterfactual = swept_gripper_clearance(
                planner,
                instance.roles["blocker"],
                frames,
                qpos=rollout.data.qpos,
                info=rollout.episode.info,
                restore_initial=True,
                initial_pose=instance.evidence["initial_blocker_pose"],
            )
            actual = swept_gripper_clearance(
                planner,
                instance.roles["blocker"],
                frames,
                qpos=rollout.data.qpos,
                info=rollout.episode.info,
                restore_initial=False,
            )
            if counterfactual >= -0.0005:
                raise PlanningError("The initial blocker does not obstruct the selected target approach")
            if actual < planner.cfg.approach_clearance:
                raise PlanningError("The parked blocker still obstructs the target approach")
            proof = {
                "dependency_checked": True,
                "counterfactual_gripper_clearance": counterfactual,
                "cleared_gripper_clearance": actual,
                "checked_approach_samples": min(32, len(frames)),
                "approach_frames": len(frames),
                "checked_target_tilt": float(grasp.tilt),
                "checked_grasp_style": grasp.style,
                "validation_phase": "preflight" if planner.rollout is None else "execution",
            }
            place(grasp)
            instance.evidence.update(proof)
            instance.evidence.setdefault("dependency_checks", []).append(proof)

        original = planner.cfg
        # This dependency concerns an inclined grasp corridor. Prefer it locally; the sampled
        # approach still has to pass reachability, camera, live collision and counterfactual checks.
        planner.cfg = replace(original, tilt=(35.0, 60.0), inclined_tilt=(45.0, 75.0), inclined_tilt_prob=0.3)
        try:
            super().pick_and_place(obj, planner, checked_place, preflight=True, objective_id=objective_id)
        finally:
            planner.cfg = original

    @staticmethod
    def place_label(transfer: Transfer, scene: SceneParams) -> str:
        if transfer.kind == "place_on_table":
            return f"move the {find(scene, transfer.object_name).label} out of the way"
        return CollectObjects.place_label(transfer, scene)

    def completed_transfer_valid(
        self,
        transfer_id: str,
        instance: TaskInstance,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        info: SceneInfo,
    ) -> bool:
        if transfer_id == "place_target":
            return inside_container(
                instance.roles["target_object"],
                instance.roles["target"],
                model,
                data,
                info,
                during_motion=True,
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
            return bool(
                instance.evidence.get("dependency_checked")
                and inside_container(transfer.object_name, transfer.target_name, model, data, info)
            )
        body = info.object_bodies[transfer.object_name]
        at_goal = np.linalg.norm(data.xpos[body, :2] - transfer.position) <= 0.04
        obj = find(info.params, transfer.object_name)
        point = data.xpos[body, :2]
        clears_bins = all(
            distance_from_container(point, c) > footprint(obj) + 0.008 for c in info.params.containers
        )
        table_contact = any(
            info.table_geom in (contact.geom1, contact.geom2)
            and any(g in info.geoms[obj.name] for g in (contact.geom1, contact.geom2))
            for contact in data.contact[: data.ncon]
        )
        return bool(
            at_goal
            and clears_bins
            and table_contact
            and resting_on_table(transfer.object_name, model, data, info)
        )
