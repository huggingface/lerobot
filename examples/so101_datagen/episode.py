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

"""One episode, success first: sample and script it, then simulate physics only.

Rendering is the expensive part, so it only happens afterwards and only for episodes that succeeded:
``render.py`` replays the recorded states.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .assets import MeshAsset
from .config import GenerationConfig
from .motion import DOWN, Plan, Planner, PlanningError
from .randomization import sample_scene, uniform
from .scene import SceneInfo, SceneParams, build_scene
from .so101 import GRIPPER, REST, Arm
from .tasks import TASKS, Task, TaskInstance

# Frames the scene settles for before the episode starts, the arm holding its start pose.
SETTLE_FRAMES = 10
# How far an object the task doesn't act on may be pushed, in meters.
DISTURBANCE = 0.015
# Bounds of random collision-free start poses, in degrees.
FREE_POSE_LOW = np.array([-90.0, -100.0, -40.0, -80.0, -120.0, 0.0])
FREE_POSE_HIGH = np.array([90.0, 20.0, 95.0, 95.0, 120.0, 70.0])


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
    plan: Plan
    # Simulated state at the start of every frame, and the joint command sent during it.
    qpos: np.ndarray | None = None
    commands: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)


class EpisodeError(Exception):
    """The episode can't be used; ``reason`` says at which stage."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def choose_task(cfg: GenerationConfig, rng: np.random.Generator) -> Task:
    names = list(cfg.tasks.weights)
    weights = np.asarray([cfg.tasks.weights[name] for name in names], dtype=np.float64)
    unknown = set(names) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown tasks {sorted(unknown)}; registered: {sorted(TASKS)}")
    return TASKS[names[rng.choice(len(names), p=weights / weights.sum())]]


def prepare(seed: int, cfg: GenerationConfig, meshes: dict[str, MeshAsset]) -> Episode:
    """Sample the task, scene and start pose of ``seed``, build the scene and script the motion."""
    rng = np.random.default_rng(seed)
    task = choose_task(cfg, rng)
    scene = sample_scene(rng, cfg, meshes, task.containers, task.min_objects)
    if scene is None:
        raise EpisodeError("layout")
    instance = task.sample(scene, rng)
    if instance is None:
        raise EpisodeError("task_sample")
    model, info = build_scene(scene, meshes, cfg.height, cfg.width)
    substeps = max(1, round(1 / (cfg.fps * model.opt.timestep)))
    model.opt.timestep = 1 / (cfg.fps * substeps)
    arm = Arm(model)
    start_mode, start = sample_start(model, arm, cfg, rng)
    planner = Planner(arm, start, cfg.fps, cfg.motion, rng, workspace=cfg.layout.radius)
    try:
        task.script(instance, scene, planner, rng)
    except PlanningError as e:
        raise EpisodeError("planning") from e
    plan = planner.plan()
    if len(plan.commands) > cfg.max_episode_seconds * cfg.fps:
        raise EpisodeError("too_long")
    return Episode(seed, task.name, instance, scene, start_mode, start, model, info, arm, plan)


def sample_start(
    model: mujoco.MjModel, arm: Arm, cfg: GenerationConfig, rng: np.random.Generator
) -> tuple[str, np.ndarray]:
    """A start pose: rest (jittered), a random collision-free pose, or hovering as if mid-task."""
    sc = cfg.start_pose
    modes = ["rest", "random_free", "mid_task"]
    weights = np.asarray([sc.rest, sc.random_free, sc.mid_task], dtype=np.float64)
    mode = modes[rng.choice(3, p=weights / weights.sum())]
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
        else:
            q = REST + rng.uniform(-1, 1, 6) * sc.rest_jitter
            q[GRIPPER] = rng.uniform(0, 30)
        if _free(model, data, arm, q):
            return mode, q
    raise EpisodeError("start_pose")


def _free(model: mujoco.MjModel, data: mujoco.MjData, arm: Arm, q: np.ndarray) -> bool:
    """Whether the arm at ``q`` stays above the table and touches nothing but itself."""
    data.qpos[arm.qadr] = arm.to_ctrl(q)
    mujoco.mj_forward(model, data)
    if data.site_xpos[arm.tcp][2] < 0.03:
        return False
    for contact in data.contact[: data.ncon]:
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if (a in arm.bodies) != (b in arm.bodies):
            return False
    return True


def simulate(
    episode: Episode,
    task: Task,
    fps: int,
    command_filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Run the plan with physics only, recording states and commands; raise ``EpisodeError`` on failure.

    ``command_filter`` turns each planned command into the executed one (the teleop-side processors).
    """
    model, arm, plan = episode.model, episode.arm, episode.plan
    data = mujoco.MjData(model)
    substeps = max(1, round(1 / (fps * model.opt.timestep)))
    data.qpos[arm.qadr] = arm.to_ctrl(episode.start)
    data.ctrl[arm.actuators] = arm.to_ctrl(episode.start)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=SETTLE_FRAMES * substeps)

    initial = {name: data.xpos[body][:2].copy() for name, body in episode.info.object_bodies.items()}
    checks = {frame: name for name, frame in plan.marks.items()}
    n = len(plan.commands)
    qpos = np.empty((n, model.nq))
    commands = np.empty((n, len(arm.qadr)))
    for t in range(n):
        if t in checks and not task.check(checks[t], episode.instance, model, data, episode.info):
            raise EpisodeError(f"check_{checks[t]}")
        qpos[t] = data.qpos
        command = plan.commands[t] if command_filter is None else command_filter(plan.commands[t])
        commands[t] = command
        data.ctrl[arm.actuators] = arm.to_ctrl(command)
        mujoco.mj_step(model, data, nstep=substeps)
    # Let the scene come to rest before judging it.
    mujoco.mj_step(model, data, nstep=5 * substeps)
    if not task.success(episode.instance, model, data, episode.info):
        raise EpisodeError("success")
    # A careful operator doesn't knock other objects around.
    acted_on = set(episode.instance.roles.values())
    for name, body in episode.info.object_bodies.items():
        if name not in acted_on and np.linalg.norm(data.xpos[body][:2] - initial[name]) > DISTURBANCE:
            raise EpisodeError("disturbed")
    episode.qpos = qpos
    episode.commands = commands
