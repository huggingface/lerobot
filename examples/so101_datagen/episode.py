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

"""Sample a scene, then interleave motion planning and persistent physics for picks and retries.

Rendering is the expensive part, so it only happens afterwards and only for episodes that succeeded:
``render.py`` replays the recorded states.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from .assets import MeshAsset
from .config import GenerationConfig
from .errors import EpisodeError
from .motion import DOWN, Plan, Planner, PlanningError
from .randomization import _corners, _in_view, sample_scene, uniform
from .scene import ObjectParams, SceneInfo, SceneParams, build_scene
from .so101 import GRIPPER, PARKED, REST, Arm
from .tasks import TASKS, Task, TaskInstance, is_lifted, touches_gripper

# Frames the scene settles for before the episode starts, the arm holding its start pose.
SETTLE_FRAMES = 10
# How far an object the task doesn't act on may be pushed, in meters.
DISTURBANCE = 0.015
# Bounds of random collision-free start poses, in degrees.
# Poses a person might leave the arm in: raised over the table, wrist pointing down-ish, not twisted.
FREE_POSE_LOW = np.array([-45.0, -100.0, 20.0, 20.0, -30.0, 0.0])
FREE_POSE_HIGH = np.array([45.0, -30.0, 95.0, 95.0, 30.0, 40.0])


@dataclass
class Episode:
    """Everything sampled for one episode, and its physics rollout."""

    seed: int
    task: str
    instance: TaskInstance
    scene: SceneParams
    start_mode: str
    start: np.ndarray
    model: mujoco.MjModel
    info: SceneInfo
    arm: Arm
    plan: Plan | None
    config: GenerationConfig
    rng_state: dict
    # Simulated state at the start of every frame, and the joint command sent during it.
    qpos: np.ndarray | None = None
    commands: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)


def choose_task(cfg: GenerationConfig, rng: np.random.Generator) -> Task:
    names = list(cfg.tasks.weights)
    weights = np.asarray([cfg.tasks.weights[name] for name in names], dtype=np.float64)
    unknown = set(names) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown tasks {sorted(unknown)}; registered: {sorted(TASKS)}")
    if (
        not len(weights)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isfinite(weights.sum())
        or not weights.sum() > 0
    ):
        raise ValueError("Task weights must be finite, nonnegative and have a positive sum")
    return TASKS[names[rng.choice(len(names), p=weights / weights.sum())]]


def prepare(
    seed: int, cfg: GenerationConfig, meshes: dict[str, MeshAsset], *, script: bool = False
) -> Episode:
    """Sample the scene and start pose. Optional static scripting supports trajectory inspection."""
    cfg.recovery.validate()
    rng = np.random.default_rng(seed)
    task = choose_task(cfg, rng)
    scene = sample_scene(
        rng,
        task.scene_config(cfg),
        meshes,
        task.containers,
        task.min_objects,
        precise_containers=task.precise_container_layout,
    )
    if scene is None:
        raise EpisodeError("layout")
    instance = task.sample(scene, rng, cfg)
    if instance is None:
        raise EpisodeError("task_sample")
    # Tasks may arrange objects after scene sampling (for example inside a tray). Recheck the actual
    # initial geometry with the configured camera aspect and workspace before compiling physics.
    corners = np.concatenate(
        [
            _corners(item, precise_containers=task.precise_container_layout)
            for item in [*scene.objects, *scene.containers]
        ]
    )
    if not _in_view(scene.front_camera, corners, cfg.width / cfg.height):
        raise EpisodeError("camera")
    if any(
        not cfg.layout.radius[0] <= np.linalg.norm(obj.pos) <= cfg.layout.radius[1]
        for obj in scene.objects
        if obj.name in instance.roles.values()
    ):
        raise EpisodeError("layout")
    model, info = build_scene(scene, meshes, cfg.height, cfg.width)
    substeps = max(1, round(1 / (cfg.fps * model.opt.timestep)))
    model.opt.timestep = 1 / (cfg.fps * substeps)
    arm = Arm(model)
    start_mode, start = sample_start(model, arm, cfg, rng)
    episode = Episode(
        seed,
        task.name,
        instance,
        scene,
        start_mode,
        start,
        model,
        info,
        arm,
        None,
        cfg,
        deepcopy(rng.bit_generator.state),
    )
    if script:
        script_episode(episode)
    return episode


def script_episode(episode: Episode, rollout: "Rollout | None" = None) -> Planner:
    cfg, task = episode.config, TASKS[episode.task]
    if task.long_horizon and rollout is None:
        raise ValueError("Long tasks require simulate() so each transfer uses observed physics")
    scene, instance, arm = episode.scene, episode.instance, episode.arm
    start, start_mode = episode.start, episode.start_mode
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(episode.rng_state)
    planner = Planner(
        arm,
        start,
        cfg.fps,
        cfg.motion,
        rng,
        workspace=cfg.layout.radius,
        object_geoms=episode.info.geoms,
        rollout=rollout,
    )

    if rollout is not None:
        rollout.planner = planner

    def script_and_check_camera() -> None:
        task.script(instance, scene, planner, rng)
        if any(not arm.wrist_camera_ok(q) for q in planner.frames):
            raise PlanningError("Wrist camera is underneath during the motion")

    def script_with_style_fallback() -> None:
        # Retry only unexecuted geometry. A straight entry is a checked fallback for random
        # sideways bends that clip an object or cross an IK branch near the workspace boundary.
        checkpoint = planner._snapshot()
        original_cfg = planner.cfg
        executed = 0 if rollout is None else rollout.cursor
        last_error = None
        for straight in (False, True):
            first_style = None
            for alternate in (False, True):
                planner._restore(checkpoint)
                planner.grasp = None
                planner.cfg = deepcopy(original_cfg)
                if straight:
                    planner.cfg.approach_offset = 0.0
                    planner._straight_entry = True
                if alternate:
                    if first_style is None:
                        break
                    style = "tangential" if first_style == "object_aligned" else "object_aligned"
                    planner.cfg.grasp_styles = {style: 1.0}
                try:
                    script_and_check_camera()
                    return
                except PlanningError as error:
                    if rollout is not None and rollout.cursor != executed:
                        raise
                    last_error = error
                    if planner.grasp is not None:
                        first_style = planner.grasp.style
                finally:
                    planner.cfg = original_cfg
        raise last_error

    try:
        if start_mode == "parked":
            # A parked gripper starts near the table; lift it clear before reaching.
            planner.transit(REST + rng.uniform(-3, 3, 6) * [1, 1, 1, 1, 1, 0])
        if start_mode == "random_free":
            # Try the direct path from a free pose. If planning or the sampled collision check fails,
            # retry from the original pose through the neutral lift-off.
            state = deepcopy(rng.bit_generator.state)
            planner.check_initial_transit = True
            try:
                obj = next(o for o in scene.objects if o.name == instance.roles["object"])
                tcp, _ = arm.tcp_pose(start)
                if (
                    np.linalg.norm(tcp - np.array([*obj.pos[:2], obj.support_height + obj.half_size[-1]]))
                    > 0.09
                ):
                    raise PlanningError("Free start is not close to the object")
                script_with_style_fallback()
            except PlanningError:
                planner.q = start.copy()
                if rollout is not None and rollout.cursor:
                    raise
                planner.attempts.clear()
                planner.memories.clear()
                planner.retry_frames.clear()
                planner._flows.clear()
                planner.frames.clear()
                planner.subtask_ids.clear()
                planner.subtasks.clear()
                planner.objective_ids.clear()
                planner.objectives.clear()
                planner.active_object = None
                planner.pick_objective_id = None
                planner.marks.clear()
                planner._blendable = 0
                planner.grasp = None
                planner.cfg = cfg.motion
                planner.check_initial_transit = False
                rng.bit_generator.state = state
                planner.transit(REST + rng.uniform(-3, 3, 6) * [1, 1, 1, 1, 1, 0])
                script_with_style_fallback()
        else:
            script_with_style_fallback()
    except PlanningError as e:
        raise EpisodeError("planning") from e
    try:
        plan = planner.plan()
    except PlanningError as e:
        raise EpisodeError("planning") from e
    maximum = cfg.max_long_episode_seconds if task.long_horizon else cfg.max_episode_seconds
    if len(plan.commands) > maximum * cfg.fps:
        raise EpisodeError("too_long")
    episode.plan = plan
    return planner


def sample_start(
    model: mujoco.MjModel, arm: Arm, cfg: GenerationConfig, rng: np.random.Generator
) -> tuple[str, np.ndarray]:
    """A start pose: rest or parked (jittered), a random collision-free pose, or hovering as if mid-task."""
    sc = cfg.start_pose
    modes = ["rest", "parked", "random_free", "mid_task"]
    weights = np.asarray([sc.rest, sc.parked, sc.random_free, sc.mid_task], dtype=np.float64)
    mode = modes[rng.choice(len(modes), p=weights / weights.sum())]
    data = mujoco.MjData(model)
    for _ in range(50):
        if mode == "random_free":
            q = rng.uniform(FREE_POSE_LOW, FREE_POSE_HIGH)
        elif mode == "mid_task":
            radius = uniform(rng, cfg.layout.radius)
            angle = np.deg2rad(uniform(rng, cfg.layout.angle))
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.04, 0.18)])
            tilt = rng.uniform(0, np.deg2rad(30))
            heading = rng.uniform(-np.pi, np.pi)
            approach = DOWN * np.cos(tilt) + np.array([np.cos(heading), np.sin(heading), 0.0]) * np.sin(tilt)
            init = REST.copy()
            init[GRIPPER] = rng.uniform(0, 80)
            q, err = arm.ik(pos, approach, None, init)
            if err > 0.005:
                continue
        elif mode == "parked":
            q = PARKED + rng.uniform(-1, 1, 6) * np.asarray(sc.parked_jitter)
            q[GRIPPER] = rng.uniform(0, 5)
        else:
            q = REST + rng.uniform(-1, 1, 6) * sc.rest_jitter
            q[GRIPPER] = rng.uniform(0, 30)
        if _free(model, data, arm, q, min_height=0.01 if mode == "parked" else 0.03) and arm.wrist_camera_ok(
            q
        ):
            return mode, q
    raise EpisodeError("start_pose")


def _free(
    model: mujoco.MjModel, data: mujoco.MjData, arm: Arm, q: np.ndarray, min_height: float = 0.03
) -> bool:
    """Whether the arm at ``q`` keeps its gripper ``min_height`` above the table and touches nothing but itself."""
    data.qpos[arm.qadr] = arm.to_ctrl(q)
    mujoco.mj_forward(model, data)
    if data.site_xpos[arm.tcp][2] < min_height:
        return False
    for contact in data.contact[: data.ncon]:
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if (a in arm.bodies) != (b in arm.bodies):
            return False
    return True


class Rollout:
    """Persistent physics for a planner: execute pending frames once, then observe and replan."""

    def __init__(
        self,
        episode: Episode,
        task: Task,
        fps: int,
        command_filter: Callable[[np.ndarray], np.ndarray] | None = None,
        actuation: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.episode, self.task, self.fps = episode, task, fps
        self.model, self.arm = episode.model, episode.arm
        self.data = mujoco.MjData(self.model)
        self.command_filter, self.actuation = command_filter, actuation
        self.substeps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self.original_qpos0 = self.model.qpos0.copy()
        self.data.qpos[self.arm.qadr] = self.arm.to_ctrl(episode.start)
        self.data.ctrl[self.arm.actuators] = self.arm.to_ctrl(episode.start)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=SETTLE_FRAMES * self.substeps)
        self.initial = {
            name: self.data.xpos[body][:2].copy() for name, body in episode.info.object_bodies.items()
        }
        self.qpos: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []
        self.cursor = 0
        self.attempts_used = 0
        self.total_attempts_used = 0
        self.planner: Planner | None = None
        self.last_plan: Plan | None = None
        self.rng = np.random.default_rng([episode.seed, 2])

    def sync_scene(self) -> None:
        # Scratch IK/collision data must use the objects' observed poses, not their initial poses.
        self.arm.scene_qpos[:] = self.data.qpos

    def scene_now(self) -> SceneParams:
        """Collision planning and carry clearance use all objects' observed positions."""
        return replace(
            self.episode.scene,
            objects=[
                replace(
                    obj,
                    pos=self.data.xpos[self.episode.info.object_bodies[obj.name], :2].tolist(),
                    support_height=max(
                        0.0,
                        float(
                            self.data.xpos[self.episode.info.object_bodies[obj.name], 2] - obj.half_size[-1]
                        ),
                    ),
                )
                for obj in self.episode.scene.objects
            ],
        )

    def transfer_satisfied(self, transfer_id: str) -> bool:
        return self.task.transfer_success(
            transfer_id, self.episode.instance, self.model, self.data, self.episode.info
        )

    def check_completed(self) -> None:
        if self.planner is not None and any(
            not self.task.completed_transfer_valid(
                result.transfer_id, self.episode.instance, self.model, self.data, self.episode.info
            )
            for result in self.planner.completed_transfers
        ):
            raise EpisodeError("completed_goal_disturbed")

    def object_now(self, obj: ObjectParams) -> ObjectParams | None:
        body = self.episode.info.object_bodies[obj.name]
        rot = self.data.xmat[body].reshape(3, 3)
        pos = self.data.xpos[body]
        lo, hi = self.episode.config.layout.radius
        if rot[2, 2] < np.cos(np.deg2rad(30)) or not lo <= np.linalg.norm(pos[:2]) <= hi:
            return None
        return replace(
            obj,
            pos=pos[:2].tolist(),
            yaw=float(np.arctan2(rot[1, 0], rot[0, 0])),
            support_height=max(0.0, float(pos[2] - obj.half_size[-1])) if obj.support_height > 0 else 0.0,
        )

    def lifted(self, name: str) -> bool:
        return is_lifted(name, self.model, self.data, self.episode.info)

    def held(self, name: str) -> bool:
        return touches_gripper(name, self.model, self.data, self.episode.info)

    def run_until(self, frame: int) -> None:
        plan = self.planner.plan() if self.planner is not None else self.episode.plan
        self.last_plan = plan
        if frame < self.cursor or frame > len(plan.commands):
            raise ValueError("Cannot rewind physics or execute commands that have not been planned")
        cfg = self.episode.config
        maximum = cfg.max_long_episode_seconds if self.task.long_horizon else cfg.max_episode_seconds
        if frame > maximum * self.fps:
            raise EpisodeError("too_long")
        attempts = plan.attempts
        for t in range(self.cursor, frame):
            attempt = plan.attempt_at(t)
            target = (
                attempt.object_name
                if attempt is not None and attempt.object_name
                else self.episode.instance.roles["object"]
            )
            geoms = set(self.episode.info.geoms[target])
            body = self.episode.info.object_bodies[target]
            watching = False
            if attempts:
                for attempt in attempts:
                    watching |= attempt.looked <= t < attempt.approach_end
                    if (
                        attempt.start <= t <= attempt.closing
                        and plan.approach_clearance > 0
                        and min(self.arm.jaw_clearances(None, geoms, qpos=self.data.qpos))
                        < plan.approach_clearance
                    ):
                        raise EpisodeError("approach_clearance")
            else:
                a, b = plan.marks.get("looked", 0), plan.marks.get("grasp_approach_end", 0)
                watching = a <= t < b
                if (
                    plan.marks.get("approach_start", a) <= t <= plan.marks.get("closing", b)
                    and plan.approach_clearance > 0
                    and min(self.arm.jaw_clearances(None, geoms, qpos=self.data.qpos))
                    < plan.approach_clearance
                ):
                    raise EpisodeError("approach_clearance")
                for name, index in plan.marks.items():
                    if index == t and not self.task.check(
                        name, self.episode.instance, self.model, self.data, self.episode.info
                    ):
                        raise EpisodeError(f"check_{name}")
            if not self.arm.wrist_camera_ok(
                None, body if watching else None, geoms if watching else None, qpos=self.data.qpos
            ):
                raise EpisodeError("wrist_view")
            self.qpos.append(self.data.qpos.copy())
            command = plan.commands[t].copy()
            if self.command_filter is not None:
                command = self.command_filter(command)
            self.commands.append(command.copy())
            executed = command if self.actuation is None else self.actuation(command)
            self.data.ctrl[self.arm.actuators] = self.arm.to_ctrl(executed)
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
            self.cursor += 1
        self.check_completed()

    def finish(self) -> None:
        if self.episode.instance.transfers:
            expected = [t.id for t in self.episode.instance.transfers]
            actual = [t.transfer_id for t in self.planner.completed_transfers]
            if actual != expected:
                raise EpisodeError("transfer_order")
        mujoco.mj_step(self.model, self.data, nstep=5 * self.substeps)
        self.check_completed()
        if not self.task.success(self.episode.instance, self.model, self.data, self.episode.info):
            raise EpisodeError("success")
        acted_on = set(self.episode.instance.roles.values())
        for name, body in self.episode.info.object_bodies.items():
            if (
                name not in acted_on
                and np.linalg.norm(self.data.xpos[body][:2] - self.initial[name]) > DISTURBANCE
            ):
                raise EpisodeError("disturbed")
        self.episode.qpos = np.asarray(self.qpos)
        self.episode.commands = np.asarray(self.commands)


def simulate(
    episode: Episode,
    task: Task,
    fps: int,
    command_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    actuation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Plan and execute with physics feedback, recording states for later rendering."""
    rollout = Rollout(episode, task, fps, command_filter, actuation)
    try:
        if episode.plan is None:
            planner = script_episode(episode, rollout)
            planner.flush()
            episode.plan = planner.plan()
        else:
            rollout.run_until(len(episode.plan.commands))
        rollout.finish()
    except PlanningError as e:
        raise EpisodeError("planning") from e
    finally:
        if episode.commands is None and rollout.last_plan is not None:
            episode.qpos = np.asarray(rollout.qpos)
            episode.commands = np.asarray(rollout.commands)
            episode.plan = replace(
                rollout.last_plan,
                commands=rollout.last_plan.commands[: rollout.cursor],
                subtask_of_frame=rollout.last_plan.subtask_of_frame[: rollout.cursor],
            )
            if rollout.planner is not None:
                for old, current in zip(episode.plan.attempts, rollout.planner.attempts, strict=False):
                    old.success = current.success
        episode.arm.scene_qpos[:] = rollout.original_qpos0
