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

import json
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
os.environ.setdefault("MUJOCO_GL", "egl")

from examples.so101_datagen.annotations import persistent_rows  # noqa: E402
from examples.so101_datagen.assets import load_meshes  # noqa: E402
from examples.so101_datagen.config import CalibrationConfig, GenerationConfig, MotionConfig  # noqa: E402
from examples.so101_datagen.episode import EpisodeError, Rollout, prepare, simulate  # noqa: E402
from examples.so101_datagen.generate import GenerateConfig, generate  # noqa: E402
from examples.so101_datagen.motion import (  # noqa: E402
    DroppedError,
    GraspAttempt,
    Objective,
    Plan,
    Planner,
    PlanningError,
    rounded_path,
    tangential_grasp_for,
)
from examples.so101_datagen.processors import (  # noqa: E402
    VirtualCalibration,
    make_actuation_pipeline,
    make_command_pipeline,
    start_episode,
    to_dict,
    to_vector,
)
from examples.so101_datagen.scene import Look, ObjectParams  # noqa: E402
from examples.so101_datagen.so101 import GRIPPER, JOINT_LIMITS_DEG, JOINTS, REST, Arm, arm_spec  # noqa: E402
from examples.so101_datagen.tasks import TASKS, TaskInstance, find  # noqa: E402
from examples.so101_datagen.worker import EPISODES_FILE  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.datasets.language_render import render_sample  # noqa: E402
from lerobot.datasets.recipe import load_recipe  # noqa: E402
from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.motors_bus import SerialMotorsBus  # noqa: E402
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT  # noqa: E402

RECIPES = ["subtask", "subtask_mem", "subtask_mem_vqa_speech"]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["pick", "hold"], ["1. pick\n2. hold", "1. hold"]),
        (
            ["pick", "pick", "pick", "hold"],
            ["1. pick\n2. hold", "1. pick\n2. hold", "1. pick\n2. hold", "1. hold"],
        ),
        (
            ["pick", "place", "pick", "place", "rest"],
            [
                "1. pick\n2. place\n3. rest",
                "1. place\n2. rest",
                "1. pick\n2. place\n3. rest",
                "1. place\n2. rest",
                "1. rest",
            ],
        ),
    ],
)
def test_retry_plans_do_not_reveal_future_failures(labels, expected):
    memories = [None, *[f"observed event {i}" for i in range(1, len(labels))]]
    plan = Plan(
        commands=np.zeros((len(labels), 6)),
        subtask_of_frame=np.arange(len(labels)),
        subtasks=labels,
        memories=memories,
    )
    rows = persistent_rows(
        TaskInstance("pick_lift", "Pick the object.", {}), plan, GenerationConfig().annotation
    )
    assert [(frame, row["content"]) for frame, row in rows if row["style"] == "plan"] == list(
        enumerate(expected)
    )
    # Re-entering a task objective updates current progress without removing observed recovery text.
    assert [(frame, row["content"]) for frame, row in rows if row["style"] == "subtask"] == list(
        enumerate(labels)
    )
    assert [(frame, row["content"]) for frame, row in rows if row["style"] == "memory"] == [
        (frame, memory) for frame, memory in enumerate(memories) if memory is not None
    ]


def test_explicit_objectives_preserve_intended_repetition_without_future_retries():
    objectives = [
        Objective("out", "move the block", "object_0", "out"),
        Objective("other", "move the cylinder", "object_1", "other"),
        Objective("back", "move the block", "object_0", "back"),
    ]
    plan = Plan(
        commands=np.zeros((4, 6)),
        subtask_of_frame=np.arange(4),
        subtasks=["move the block", "move the block", "move the cylinder", "move the block"],
        objectives=objectives,
        objective_ids=["out", "out", "other", "back"],
    )
    rows = persistent_rows(
        TaskInstance("swap_objects", "Swap the objects.", {}), plan, GenerationConfig().annotation
    )
    plans = [r["content"] for _, r in rows if r["style"] == "plan"]
    assert plans == [
        "1. move the block\n2. move the cylinder\n3. move the block",
        "1. move the block\n2. move the cylinder\n3. move the block",
        "1. move the cylinder\n2. move the block",
        "1. move the block",
    ]
    # Distinct transfers of the same object must also use distinct grasp points.
    plan.attempts = [
        GraspAttempt(None, 0, 0, 0, 0, 0, object_name="object_0", transfer_id="out"),
        GraspAttempt(None, 2, 2, 2, 2, 2, object_name="object_1", transfer_id="other"),
        GraspAttempt(None, 4, 4, 4, 4, 4, object_name="object_0", transfer_id="back"),
    ]
    assert plan.attempt_at(2) is plan.attempts[1]
    assert plan.attempt_at(3) is plan.attempts[2]


@pytest.mark.parametrize(
    "weights", [{}, {"place_in": -1}, {"place_in": 0}, {"place_in": float("nan")}, {"missing": 1}]
)
def test_invalid_task_weights_are_rejected(weights):
    cfg = GenerationConfig()
    cfg.tasks.weights = weights
    with pytest.raises(ValueError):
        prepare(0, cfg, {})


def test_collect_objects_continues_and_recovers_on_second_transfer(monkeypatch):
    cfg = GenerationConfig()
    cfg.tasks.weights = {"collect_objects": 1}
    cfg.container.collection_inner_size = (0.08, 0.13)
    cfg.recovery.miss_prob = 0
    cfg.recovery.max_attempts = 2
    ep = prepare(9, cfg, load_meshes())
    pick_once = Planner._pick_once
    failed = []
    prefixes = []
    run_until = Rollout.run_until

    def miss_second(planner, obj):
        if planner.transfer_id == ep.instance.transfers[1].id and not failed:
            failed.append(obj.name)
            planner._miss_mode = "high"
            planner._miss_distance = 0.03
        return pick_once(planner, obj)

    def frozen_prefix(rollout, frame):
        if rollout.cursor:
            np.testing.assert_array_equal(rollout.planner.frames[: rollout.cursor], rollout.commands)
        run_until(rollout, frame)
        if rollout.planner.completed_transfers:
            prefixes.append(rollout.cursor)

    monkeypatch.setattr(Planner, "_pick_once", miss_second)
    monkeypatch.setattr(Rollout, "run_until", frozen_prefix)
    simulate(ep, TASKS[ep.task], cfg.fps)
    assert len(ep.plan.completed_transfers) == 2
    assert prefixes
    assert failed
    assert {a.object_name for a in ep.plan.attempts} == {t.object_name for t in ep.instance.transfers}
    assert ep.plan.attempts[0].miss_mode is None
    second = [a for a in ep.plan.attempts if a.transfer_id == ep.instance.transfers[1].id]
    assert [a.success for a in second] == [False, True]
    assert ep.plan.completed_transfers[0].frame < ep.plan.completed_transfers[1].frame
    assert TASKS[ep.task].success(ep.instance, ep.model, _final_data(ep), ep.info)

    # Completing only one transfer is insufficient, and later damage to a completed goal is rejected.
    rollout = Rollout(ep, TASKS[ep.task], cfg.fps)
    rollout.data.qpos[:] = ep.qpos[-1]
    mujoco.mj_forward(ep.model, rollout.data)
    rollout.planner = type("Completed", (), {"completed_transfers": ep.plan.completed_transfers})()
    rollout.check_completed()
    obj = ep.instance.transfers[0].object_name
    joint = ep.model.body_jntadr[ep.info.object_bodies[obj]]
    address = ep.model.jnt_qposadr[joint]
    rollout.data.qpos[address : address + 3] = ep.model.qpos0[address : address + 3]
    mujoco.mj_forward(ep.model, rollout.data)
    assert not TASKS[ep.task].success(ep.instance, ep.model, rollout.data, ep.info)
    with pytest.raises(EpisodeError, match="completed_goal_disturbed"):
        rollout.check_completed()


def _final_data(ep):
    data = mujoco.MjData(ep.model)
    data.qpos[:] = ep.qpos[-1]
    mujoco.mj_forward(ep.model, data)
    return data


def test_collection_accepts_puck_rolling_inside_tray():
    cfg = GenerationConfig()
    cfg.tasks.weights = {"collect_objects": 1}
    cfg.container.collection_inner_size = (0.08, 0.13)
    ep = prepare(60, cfg, load_meshes())
    command = make_command_pipeline(cfg.noise, cfg.fps)
    actuation = make_actuation_pipeline(cfg.noise)
    start_episode(np.random.default_rng([60, 1]), command, actuation)
    simulate(
        ep,
        TASKS[ep.task],
        cfg.fps,
        command_filter=lambda q: to_vector(command((to_dict(q), {}))),
        actuation=lambda q: to_vector(actuation((to_dict(q), {}))),
    )
    task = TASKS[ep.task]
    assert len(ep.plan.completed_transfers) == 2
    assert task.success(ep.instance, ep.model, _final_data(ep), ep.info)
    first = ep.plan.completed_transfers[0]
    data = mujoco.MjData(ep.model)
    transient = False
    for state in ep.qpos[first.frame :]:
        data.qpos[:] = state
        mujoco.mj_forward(ep.model, data)
        if not task.transfer_success(first.transfer_id, ep.instance, ep.model, data, ep.info):
            assert task.completed_transfer_valid(first.transfer_id, ep.instance, ep.model, data, ep.info)
            transient = True
    assert transient


def test_side_grasp_uses_tangential_jaws_and_rejects_short_objects():
    arm = Arm(arm_spec().compile())
    obj = ObjectParams(
        name="object",
        label="cylinder",
        shape="cylinder",
        half_size=[0.015, 0.015, 0.03],
        look=Look(rgba=[1, 1, 1, 1]),
        mass=0.04,
        friction=1.0,
        pos=[0.28, 0.0],
        yaw=0.0,
    )
    grasp = tangential_grasp_for(arm, obj, 65.0, np.random.default_rng(2), 0.0, 0.025, 90)
    assert grasp.style == "tangential"
    assert abs(grasp.roll) == 90
    assert abs(np.dot(grasp.approach, grasp.jaw)) < 1e-6
    assert abs(grasp.jaw[0]) < 0.06
    assert 0.025 <= grasp.height <= 0.05
    np.testing.assert_allclose(np.dot(grasp.pos - grasp.point, grasp.approach), 0.01)
    start = REST.copy()
    start[4] = grasp.roll
    _, error = arm.ik(grasp.pos, grasp.approach, grasp.jaw, start, iters=300)
    assert error < 0.004
    shallower = tangential_grasp_for(arm, obj, 45.0, np.random.default_rng(2), 0.0, 0.025, 90)
    assert shallower.tilt == 45.0
    assert abs(np.dot(shallower.approach, shallower.jaw)) < 1e-6
    vertical = tangential_grasp_for(arm, obj, 0.0, np.random.default_rng(2), 0.0, 0.025, 90)
    assert vertical.style == "tangential"
    np.testing.assert_allclose(vertical.approach, [0, 0, -1], atol=1e-8)
    assert abs(vertical.jaw[0]) < 0.06
    obj.half_size[0] = arm.max_jaw_span
    with pytest.raises(PlanningError, match="too wide"):
        tangential_grasp_for(arm, obj, 45.0, np.random.default_rng(2), 0.0, 0.025)
    obj.half_size[-1] = 0.015
    obj.half_size[0] = 0.015
    with pytest.raises(PlanningError, match="too short"):
        tangential_grasp_for(arm, obj, 65.0, np.random.default_rng(2), 0.0, 0.025)


def test_early_wrist_transit_respects_speed_limits():
    arm = Arm(arm_spec().compile())
    cfg = MotionConfig()
    target = REST.copy()
    target[:5] += [15, 10, -10, 5, 90]
    progress = []
    for early in (False, True):
        planner = Planner(arm, REST, 30, cfg, np.random.default_rng(42))
        planner.transit(target, early_roll=early)
        frames = np.vstack([REST, planner.frames])
        np.testing.assert_allclose(frames[-1, :5], target[:5])
        speed = np.abs(np.diff(frames, axis=0)) * planner.fps
        assert np.all(speed <= np.asarray(cfg.max_joint_speed) + 1e-6)
        progress.append(np.flatnonzero(np.abs(frames[:, 4] - target[4]) < 9)[0] / (len(frames) - 1))
    assert progress[1] < progress[0] - 0.15


@pytest.mark.parametrize("seed", [2, 21])
def test_open_approach_has_clearance_in_recorded_physics(seed):
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.recovery.miss_prob = 0
    ep = prepare(seed, cfg, load_meshes())
    simulate(ep, TASKS[ep.task], cfg.fps)
    geoms = set(ep.info.geoms[ep.instance.roles["object"]])
    a, b = ep.plan.marks["approach_start"], ep.plan.marks["closing"]
    gaps = [ep.arm.jaw_clearances(None, geoms, qpos=q) for q in ep.qpos[a : b + 1]]
    assert np.min(gaps) >= cfg.motion.approach_clearance
    # The target stays still throughout the open approach; the old seed 2 pushed it about 8 mm.
    body = ep.info.object_bodies[ep.instance.roles["object"]]
    data = mujoco.MjData(ep.model)
    positions = []
    for q in ep.qpos[a : b + 1]:
        data.qpos[:] = q
        mujoco.mj_forward(ep.model, data)
        positions.append(data.xpos[body, :2].copy())
    assert np.max(np.linalg.norm(np.array(positions) - positions[0], axis=1)) < 0.001


def test_forced_miss_recovers_without_rewriting_executed_prefix(monkeypatch):
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    cfg.recovery.miss_prob = 1.0
    cfg.recovery.miss_modes = {"high": 1.0}
    cfg.objects.mesh_prob = 0
    cfg.objects.width = (0.035, 0.035)
    cfg.objects.height = (0.025, 0.025)
    ep = prepare(6, cfg, {})
    assert ep.plan is None
    run_until = Rollout.run_until
    flushes = []

    def checked_run(rollout, frame):
        if rollout.cursor:
            np.testing.assert_array_equal(rollout.planner.frames[: rollout.cursor], rollout.commands)
        run_until(rollout, frame)
        flushes.append(frame)

    recover = Planner.recover

    def checked_recover(planner, obj, reason):
        before = len(planner.frames)
        origin, _ = planner.arm.tcp_pose(planner.q)
        roll = planner.q[4]
        recover(planner, obj, reason)
        segment = planner.frames[before:]
        assert segment
        tcp = np.array([planner.arm.tcp_pose(q)[0] for q in segment])
        # Opening and a short local back-off must not become a trip through the neutral pose.
        assert np.max(np.linalg.norm(tcp - origin, axis=1)) <= 0.04
        assert np.max(np.abs(np.array(segment)[:, 4] - roll)) < 1e-5

    monkeypatch.setattr(Planner, "recover", checked_recover)
    monkeypatch.setattr(Rollout, "run_until", checked_run)
    simulate(ep, TASKS[ep.task], cfg.fps)
    assert 2 <= len(ep.plan.attempts) <= cfg.recovery.max_attempts
    assert ep.plan.attempts[0].miss_mode == "high"
    assert ep.plan.attempts[0].success is False
    assert ep.plan.attempts[-1].success is True
    assert ep.plan.retry_frames == sorted(ep.plan.retry_frames)
    assert len(ep.plan.retry_frames) == len(ep.plan.attempts) - 1
    assert any(memory and "missed" in memory for memory in ep.plan.memories)
    assert len(flushes) >= 3
    assert len(ep.commands) == len(ep.qpos) == len(ep.plan.commands)


def test_dropped_object_is_picked_again_within_shared_budget(monkeypatch):
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    cfg.recovery.miss_prob = 0
    cfg.objects.mesh_prob = 0
    cfg.objects.width = (0.04, 0.04)
    cfg.objects.height = (0.03, 0.03)
    calls = []
    task = TASKS["pick_lift"]

    def script(instance, scene, planner, rng):
        obj = find(scene, instance.roles["object"])

        def drop_first(grasp):
            calls.append(grasp)
            if len(calls) == 1:
                planner.gripper(100)
                planner.dwell(0.6)
                planner.flush()
                assert not planner.rollout.held(obj.name)
                raise DroppedError(obj.name)

        task.pick_and_place(obj, planner, drop_first)

    monkeypatch.setattr(task, "script", script)
    ep = prepare(6, cfg, {})
    simulate(ep, task, cfg.fps)
    assert len(calls) == 2
    assert len(ep.plan.attempts) == 2
    assert all(a.success for a in ep.plan.attempts)
    assert len(ep.plan.retry_frames) == 1
    assert any(memory and "dropped" in memory for memory in ep.plan.memories)


def test_attempt_budget_rejects_without_an_extra_attempt(monkeypatch):
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.recovery.max_attempts = 1
    cfg.recovery.miss_prob = 0
    ep = prepare(2, cfg, load_meshes())
    monkeypatch.setattr(Rollout, "lifted", lambda self, name: False)
    with pytest.raises(EpisodeError, match="attempts_exhausted"):
        simulate(ep, TASKS[ep.task], cfg.fps)
    assert len(ep.plan.attempts) == 1
    assert ep.plan.attempts[0].success is False


def test_observed_object_pose_does_not_change_model_joint_references():
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    ep = prepare(2, cfg, load_meshes())
    rollout = Rollout(ep, TASKS[ep.task], cfg.fps)
    original = ep.model.qpos0.copy()
    obj = find(ep.scene, ep.instance.roles["object"])
    body = ep.info.object_bodies[obj.name]
    joint = ep.model.body_jntadr[body]
    address = ep.model.jnt_qposadr[joint]
    rollout.data.qpos[address] += 0.01
    mujoco.mj_forward(ep.model, rollout.data)
    observed = rollout.object_now(obj)
    assert observed.pos[0] == pytest.approx(obj.pos[0] + 0.01, abs=1e-5)
    rollout.sync_scene()
    np.testing.assert_array_equal(ep.model.qpos0, original)
    np.testing.assert_array_equal(ep.arm.scene_qpos, rollout.data.qpos)


def test_tangential_box_width_includes_rotated_corners():
    arm = Arm(arm_spec().compile())
    obj = ObjectParams(
        name="object",
        label="block",
        shape="box",
        half_size=[0.025, 0.015, 0.03],
        look=Look(rgba=[1, 1, 1, 1]),
        mass=0.04,
        friction=1.0,
        pos=[0.28, 0],
        yaw=np.deg2rad(15),
    )
    grasp = tangential_grasp_for(arm, obj, 0, np.random.default_rng(2), 0, 0.025, heading_offset=0)
    expected = 2 * (0.025 * np.sin(obj.yaw) + 0.015 * np.cos(obj.yaw))
    assert grasp.width == pytest.approx(expected)
    assert grasp.width > 2 * obj.half_size[1] + 0.01


def test_path_rounding_removes_corner_without_moving_end_poses():
    path = np.repeat(REST[None], 61, axis=0)
    path[:31, 0] += np.linspace(0, 10, 31)
    path[31:, 0] += 10
    path[31:, 1] += np.linspace(0, 10, 31)[1:]
    smooth, source = rounded_path(path, np.asarray(MotionConfig().max_joint_speed))
    np.testing.assert_allclose(smooth[[0, -1]], path[[0, -1]])
    assert np.all(np.diff(source) >= 0)
    # The original right-angle joint-space corner becomes a continuous change of velocity.
    velocity = np.diff(smooth[:, :5], axis=0)
    assert np.max(np.linalg.norm(np.diff(velocity, axis=0), axis=1)) < 0.05 * np.max(
        np.linalg.norm(velocity, axis=1)
    )
    reference = np.column_stack([np.interp(source, np.arange(len(path)), q) for q in path.T])
    assert np.max(np.abs(smooth - reference)) < 1.0


def test_connected_moves_vary_speed_without_stopping_at_boundary():
    arm = Arm(arm_spec().compile())
    cfg = MotionConfig()
    planner = Planner(arm, REST, 30, cfg, np.random.default_rng(4))
    for delta in (20.0, 25.0):
        start = planner.q.copy()
        travel = np.zeros_like(start)
        travel[0] = delta
        planner._segment(1.5, lambda t, start=start, travel=travel: start + t[:, None] * travel)
        planner.mark(f"pan_{int(delta)}")
    plan = planner.plan()
    speed = np.diff(plan.commands[:, 0]) * planner.fps
    middle = speed[len(speed) // 3 : 2 * len(speed) // 3]
    # A broad varying pulse, with no stop/restart or flat cruise at the primitive boundary.
    assert 0.45 * middle.max() < middle.min() < 0.9 * middle.max()
    assert not np.any((middle[1:-1] < middle[:-2]) & (middle[1:-1] < middle[2:]))
    np.testing.assert_array_equal(plan.commands, planner.plan().commands)
    assert np.all(np.abs(np.diff(plan.commands, axis=0)) * planner.fps <= cfg.max_joint_speed)
    assert plan.marks["pan_25"] == len(plan.commands)
    assert 0 < plan.marks["pan_20"] < plan.marks["pan_25"]
    np.testing.assert_allclose(plan.commands[-1], planner.q)


def test_wrist_camera_rejects_underneath_mount():
    arm = Arm(arm_spec().compile())
    q = REST.copy()
    q[4] = -90
    assert arm.wrist_camera_ok(q)
    q[4] = 0
    assert arm.wrist_camera_ok(q)
    q[4] = 90
    assert not arm.wrist_camera_ok(q)


def test_aligned_close_start_goes_directly_to_grasp():
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    cfg.motion.grasp_styles = {"tangential": 1.0}
    ep = prepare(11, cfg, load_meshes(), script=True)
    grasp = ep.plan.grasp
    # Construct an aligned pose 4 cm before contact, independently of trajectory timing.
    start, error = ep.arm.ik(
        grasp.pos - 0.04 * grasp.approach,
        grasp.approach,
        grasp.jaw,
        ep.plan.commands[ep.plan.marks["grasp_approach_end"] - 1],
        tol=1e-5,
    )
    assert error < 0.004
    cfg.motion.inclined_tilt_prob = 0.0
    cfg.motion.grasp_styles = {grasp.style: 1.0}
    planner = Planner(
        ep.arm, start, cfg.fps, cfg.motion, np.random.default_rng(11), object_geoms=ep.info.geoms
    )
    planner.subtask("pick")
    planner.pick(find(ep.scene, ep.instance.roles["object"]))
    assert "direct_grasp" in planner.marks


def test_look_approach_blends_while_object_remains_visible():
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    cfg.motion.grasp_styles = {"tangential": 1.0}
    ep = prepare(11, cfg, load_meshes(), script=True)
    start = ep.plan.marks["looked"]
    overlap_end = ep.plan.marks["look_overlap_end"]
    approach_end = ep.plan.marks["grasp_approach_end"]
    assert start < overlap_end < approach_end
    name = ep.instance.roles["object"]
    assert all(
        ep.arm.wrist_camera_ok(q, ep.info.object_bodies[name], set(ep.info.geoms[name]))
        for q in ep.plan.commands[start:approach_end]
    )


def test_closing_blends_into_lift_within_speed_limits():
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    cfg.motion.grasp_styles = {"tangential": 1.0}
    ep = prepare(11, cfg, load_meshes(), script=True)
    start, end = ep.plan.marks["grasp_approach_end"], ep.plan.marks["grasped"]
    velocity = np.diff(ep.plan.commands[start : end + 1], axis=0) * cfg.fps
    # The gripper is still closing while the arm starts to lift.
    assert np.any((np.abs(velocity[:, GRIPPER]) > 0.01) & (np.linalg.norm(velocity[:, :5], axis=1) > 0.01))
    assert np.all(np.abs(velocity) <= np.asarray(cfg.motion.max_joint_speed) + 1e-6)


def _gl_available() -> bool:
    try:
        from mujoco.rendering.classic import gl_context

        gl_context.GLContext(8, 8).free()
        return True
    except Exception:
        return False


def test_virtual_calibration_reports_like_the_motors_bus():
    """What the synthetic follower reports equals what a real SO-101 bus computes from the same ticks."""
    motors = {
        joint: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if i == GRIPPER else MotorNormMode.DEGREES)
        for i, joint in enumerate(JOINTS)
    }
    rng = np.random.default_rng(0)
    limits = np.array(JOINT_LIMITS_DEG)
    calibration = VirtualCalibration(CalibrationConfig())
    for _ in range(50):
        calibration.start_episode(rng)
        bus = SerialMotorsBus(port="/dev/null", motors=motors, calibration=calibration.motor_calibration())
        values = rng.uniform(limits[:, 0], limits[:, 1])
        values[GRIPPER] = rng.uniform(0, 100)
        angles = values.copy()
        angles[GRIPPER] = limits[GRIPPER, 0] + values[GRIPPER] / 100 * (
            limits[GRIPPER, 1] - limits[GRIPPER, 0]
        )
        raw = calibration.raw(angles)
        expected = bus._normalize({i + 1: int(r) for i, r in enumerate(raw)})
        np.testing.assert_allclose(calibration.report(values), [expected[i + 1] for i in range(len(JOINTS))])


@pytest.mark.skipif(not _gl_available(), reason="needs an OpenGL context (EGL)")
def test_generate_merges_annotated_shards(tmp_path):
    """Two tiny shards are generated, merged, and every frame renders with the shipped recipes."""
    cfg = GenerateConfig(
        output_dir=tmp_path,
        num_shards=2,
        episodes_per_shard=1,
        num_workers=2,
    )
    cfg.generation.height = cfg.generation.width = 64
    cfg.generation.objects.mesh_prob = 0.0
    cfg.generation.annotation.min_visible_pixels = 4
    generate(cfg)

    merged = tmp_path / "merged"
    dataset = LeRobotDataset(cfg.repo_id, root=merged)
    assert dataset.num_episodes == 2
    assert dataset.meta.robot_type == "so_follower"
    assert dataset.meta.info.tools
    assert dataset.features["observation.state"]["names"] == [f"{j}.pos" for j in JOINTS]
    sidecar = [json.loads(line) for line in (merged / EPISODES_FILE).read_text().splitlines()]
    assert [entry["episode_index"] for entry in sidecar] == [0, 1]

    recipes = {name: load_recipe(f"src/lerobot/configs/recipes/{name}.yaml") for name in RECIPES}
    rendered_frames = dict.fromkeys(RECIPES, 0)
    vqa_frames = 0
    front_boxes = 0
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["observation.images.front"].shape == (3, 64, 64)
        persistent, events = item[LANGUAGE_PERSISTENT], item[LANGUAGE_EVENTS]
        assert persistent
        vqa_frames += bool(events)
        front_answers = [
            e
            for e in events
            if e["role"] == "assistant" and e["style"] == "vqa" and e["camera"] == "observation.images.front"
        ]
        assert len(front_answers) <= 1
        front_boxes += any("detections" in json.loads(e["content"]) for e in front_answers)
        for name, recipe in recipes.items():
            # Rendering raises on ambiguous or malformed rows. None is a sub-recipe without a target at
            # this frame (e.g. memory before the first boundary), which training falls back from.
            rendered = render_sample(
                recipe=recipe,
                persistent=persistent,
                events=events,
                t=float(item["timestamp"]),
                sample_idx=i,
                task=item["task"],
            )
            rendered_frames[name] += rendered is not None
            if events and name == "subtask_mem_vqa_speech":
                assert rendered is not None, f"VQA frame {i} does not render"
    assert vqa_frames > 0
    assert front_boxes >= 2 * len(dataset) / cfg.generation.fps
    for name, count in rendered_frames.items():
        assert count > 0.7 * len(dataset), f"{name} renders {count}/{len(dataset)} frames"


def test_front_stream_does_not_collide_with_general_front_questions():
    from examples.so101_datagen.annotations import annotation_frames

    cfg = GenerationConfig().annotation
    boxes, general = annotation_frames(900, 30, cfg, np.random.default_rng(3))
    assert len(boxes) == 90
    assert len(general) == 30
    assert [general[t] for t in sorted(general)].count("front") == 15
    assert not boxes.intersection(t for t, camera in general.items() if camera == "front")


def test_grasp_question_requires_visible_target_and_uses_projected_point():
    from examples.so101_datagen.annotations import vqa_pair
    from examples.so101_datagen.render import Visible

    cfg = GenerationConfig().annotation
    cfg.question_weights = {"grasp": 1.0}
    visible = {"target": Visible("target", "red block", 80, [0.1, 0.2, 0.4, 0.5], [0.25, 0.35])}
    # A grasp target must be visible; a different visible object cannot stand in for it.
    scene = None
    rng = np.random.default_rng(4)
    assert vqa_pair("front", visible, scene, {}, cfg, rng, grasp=("hidden", "red block", [0.3, 0.4])) is None
    pair = vqa_pair("front", visible, scene, {}, cfg, rng, grasp=("target", "red block", [0.3, 0.4]))
    answer = json.loads(pair[1]["content"])
    assert answer == {"label": "grasp point of the red block", "point_format": "xy", "point": [0.3, 0.4]}


def test_motion_noise_scales_with_k_and_decays_without_gripper_noise():
    from examples.so101_datagen.processors import MotionNoise, to_dict, to_vector

    steps = [MotionNoise(k=(k, k)) for k in (0.0, 0.2, 0.4)]
    for step in steps:
        step.start_episode(np.random.default_rng(8))
    q = REST.copy()
    for frame in range(30):
        q[0] += 0.5
        q[GRIPPER] = frame
        outputs = [to_vector(step.action(to_dict(q))) for step in steps]
        np.testing.assert_array_equal(outputs[0], q)
        np.testing.assert_allclose(outputs[2][:5] - q[:5], 2 * (outputs[1][:5] - q[:5]), atol=1e-10)
        assert all(out[GRIPPER] == q[GRIPPER] for out in outputs)
    initial = steps[2]._offset.copy()
    assert np.linalg.norm(initial) > 0
    for _ in range(45):
        steps[2].action(to_dict(q))
    np.testing.assert_allclose(steps[2]._offset, initial * np.exp(-3), atol=1e-10)
    steps[2].start_episode(np.random.default_rng(8))
    np.testing.assert_array_equal(to_vector(steps[2].action(to_dict(q))), q)


def test_wide_camera_keeps_scene_inside_frustum():
    from examples.so101_datagen.randomization import _corners, _in_view

    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 1.0
    cfg.objects.mesh_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    for seed in range(5):
        ep = prepare(seed, cfg, {})
        assert ep.scene.front_camera.mode == "wide"
        corners = np.concatenate([_corners(item) for item in [*ep.scene.objects, *ep.scene.containers]])
        assert _in_view(ep.scene.front_camera, corners, cfg.width / cfg.height)
        assert 50 <= ep.scene.front_camera.fovy <= 65
        # A front camera must stay forward-facing and oblique, rather than behind or overhead.
        camera = ep.scene.front_camera
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.array(camera.quat))
        backwards = rotation.reshape(3, 3)[:, 2]
        assert abs(np.degrees(np.arctan2(backwards[1], backwards[0]))) <= 50
        assert 25 <= np.degrees(np.arcsin(backwards[2])) <= 65
        assert camera.pos[0] > cfg.front_camera.lookat[0]
        assert np.linalg.norm(np.array(camera.pos) - cfg.front_camera.lookat) < 0.66


@pytest.mark.skipif(not _gl_available(), reason="needs an OpenGL context (EGL)")
def test_camera_projection_rejects_behind_and_offscreen_points():
    from examples.so101_datagen.render import StateRenderer

    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.objects.mesh_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    ep = prepare(2, cfg, {})
    renderer = StateRenderer(ep.model, cfg.height, cfg.width)
    try:
        renderer.set_state(ep.model.qpos0)
        cam = ep.model.camera("front").id
        origin = renderer.data.cam_xpos[cam]
        rot = renderer.data.cam_xmat[cam].reshape(3, 3)
        np.testing.assert_allclose(renderer.project("front", origin - rot[:, 2]), [0.5, 0.5])
        assert renderer.project("front", origin + rot[:, 2]) is None
        assert renderer.project("front", origin - rot[:, 2] + 10 * rot[:, 0]) is None
    finally:
        renderer.close()


def test_obstructed_entry_replans_before_executing_an_attempt(monkeypatch):
    cfg = GenerationConfig()
    cfg.front_camera.wide_prob = 0
    cfg.tasks.weights = {"pick_lift": 1.0}
    cfg.objects.mesh_prob = 0
    cfg.objects.width = (0.035, 0.035)
    cfg.objects.height = (0.025, 0.025)
    cfg.recovery.miss_prob = 0
    original = Planner._off_axis
    rejected = []

    def obstructed(self, approach):
        if not self._straight_entry:
            rejected.append(len(self.frames))
            raise PlanningError("Sampled entry is obstructed")
        return original(self, approach)

    monkeypatch.setattr(Planner, "_off_axis", obstructed)
    ep = prepare(6, cfg, {})
    simulate(ep, TASKS[ep.task], cfg.fps)
    assert rejected
    assert len(ep.plan.attempts) == 1
    assert ep.plan.attempts[0].success
    assert not ep.plan.retry_frames


@pytest.mark.parametrize(
    "style",
    [
        "kitchenware_tray",
        "crate",
        "metal_tray_small",
        "metal_tray_medium",
        "metal_tray_long",
        "wooden_tray",
        "white_dish",
    ],
)
def test_imported_container_has_open_interior_and_solid_support(style):
    from examples.so101_datagen.scene import ContainerParams, _add_container, container_asset

    spec = mujoco.MjSpec()
    container = ContainerParams(
        "bin", "bin", 0.075, 0.04, 0.003, Look([0.1, 0.3, 0.8, 1]), [0, 0], 0, style=style
    )
    _add_container(spec, container)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
    # Falling probes cover the usable interior, including the seams of decomposed collision meshes.
    for x in np.linspace(-container.inner_half_xy[0] + 0.015, container.inner_half_xy[0] - 0.015, 5):
        for y in np.linspace(-0.06, 0.06, 5):
            body = spec.worldbody.add_body(pos=[x, y, 0.1])
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0], mass=0.01)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1500)
    asset = container_asset(style)
    floor = container.wall_height * asset["floor"] / asset["height"]
    np.testing.assert_allclose(data.xpos[2:, 2], floor + 0.006, atol=0.001)


def test_rectangular_tray_layout_clearance_and_containment():
    from examples.so101_datagen.randomization import Footprint, _separated

    tray = Footprint(0, 0, 0.08, (0.15, 0.08), np.pi / 2)
    assert not _separated(tray, Footprint(0, 0.17, 0.02), 0.01)
    assert _separated(tray, Footprint(0.12, 0, 0.02), 0.01)
    assert not _separated(tray, Footprint(0.1, 0, 0.02), 0.01)


@pytest.mark.parametrize("seed", [2, 20])
def test_ordered_transfers_preserve_order_and_previous_goal_on_retry(seed, monkeypatch):
    cfg = GenerationConfig()
    cfg.tasks.weights = {"ordered_transfers": 1}
    cfg.recovery.miss_prob = 0
    ep = prepare(seed, cfg, load_meshes())
    pick_once = Planner._pick_once
    missed = []

    def miss_second(planner, obj):
        if planner.transfer_id == "transfer_1" and not missed:
            missed.append(obj.name)
            planner._miss_mode = "high"
            planner._miss_distance = 0.03
        return pick_once(planner, obj)

    monkeypatch.setattr(Planner, "_pick_once", miss_second)
    simulate(ep, TASKS[ep.task], cfg.fps)
    assert missed
    assert {t.kind for t in ep.instance.transfers} == {"place_in", "place_near"}
    assert [r.transfer_id for r in ep.plan.completed_transfers] == ["transfer_0", "transfer_1"]
    second = [a for a in ep.plan.attempts if a.transfer_id == "transfer_1"]
    assert [a.success for a in second] == [False, True]
    rows = persistent_rows(ep.instance, ep.plan, cfg.annotation)
    assert all("retry" not in atom["content"].lower() for _, atom in rows if atom["style"] == "plan")
    assert TASKS[ep.task].success(ep.instance, ep.model, _final_data(ep), ep.info)
    # Final positions alone cannot certify the wrong execution order or a partial task.
    rollout = Rollout(ep, TASKS[ep.task], cfg.fps)
    rollout.planner = type("Completed", (), {"completed_transfers": ep.plan.completed_transfers[::-1]})()
    with pytest.raises(EpisodeError, match="transfer_order"):
        rollout.finish()
    rollout.planner.completed_transfers = ep.plan.completed_transfers[:1]
    with pytest.raises(EpisodeError, match="transfer_order"):
        rollout.finish()


@pytest.mark.parametrize(
    "style",
    [
        "kitchenware_tray",
        "crate",
        "metal_tray_small",
        "metal_tray_medium",
        "metal_tray_long",
        "wooden_tray",
        "white_dish",
    ],
)
def test_original_container_textures_keep_labels_uvs_and_physics(style):
    from examples.so101_datagen.randomization import _sample_container
    from examples.so101_datagen.scene import _add_container, container_asset

    cfg = GenerationConfig()
    cfg.container.style = style
    variants = []
    for probability in (0, 1):
        cfg.container.original_texture_prob = probability
        colors, labels = set(), set()
        container = _sample_container(np.random.default_rng(2), cfg, "bin", labels, colors)
        assert labels == {container.label}
        assert colors == {container.color}
        if probability:
            assert container.look.texture == {"kind": "container_asset", "name": style}
            asset = container_asset(style)
            assert container.label == asset.get(
                "native_label", f"blue {'tray' if style == 'kitchenware_tray' else 'crate'}"
            )
            assert container.color == asset["native_color"]
            assert container.look.rgba == [1, 1, 1, 1]  # Source albedo must not be tinted again.
            # Repeated native assets keep distinguishable names rather than two "blue tray" labels.
            second = _sample_container(np.random.default_rng(3), cfg, "bin2", labels, colors)
            assert second.label != container.label
            assert container.label in labels and second.label in labels
        spec = mujoco.MjSpec()
        _add_container(spec, container)
        model = spec.compile()
        if probability:
            assert model.ntex == 1
            assert len(model.mesh_texcoord) > 0
            material = model.material("bin_mat").id
            assert model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0
            assert not model.mat_texuniform[material]
        variants.append((container, model))
    a, b = variants
    assert (a[0].inner_half, a[0].wall_height, a[0].yaw) == (b[0].inner_half, b[0].wall_height, b[0].yaw)
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(a[1], field), getattr(b[1], field))


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_original_container_texture_probability_is_validated(probability):
    from examples.so101_datagen.randomization import _sample_container

    cfg = GenerationConfig()
    cfg.container.original_texture_prob = probability
    with pytest.raises(ValueError, match="original_texture_prob"):
        _sample_container(np.random.default_rng(0), cfg, "bin", set(), set())


def test_container_style_mix_preserves_material_names_and_rejects_invalid_weights():
    from examples.so101_datagen.randomization import _sample_container

    cfg = GenerationConfig()
    cfg.container.style_weights = {"metal_tray_small": 1, "wooden_tray": 1, "white_dish": 1, "crate": 0}
    cfg.container.original_texture_prob = 1
    seen = set()
    for seed in range(30):
        container = _sample_container(np.random.default_rng(seed), cfg, "bin", set(), set())
        seen.add(container.style)
        assert max(container.outer_half_xy) * 2 <= cfg.container.max_asset_length
        assert container.label in {"metal tray", "wooden tray", "white dish"}
    assert seen == {"metal_tray_small", "wooden_tray", "white_dish"}
    for weights in ({"unknown": 1}, {"crate": -1}, {"crate": 0}, {"crate": float("nan")}):
        cfg.container.style_weights = weights
        with pytest.raises(ValueError):
            _sample_container(np.random.default_rng(0), cfg, "bin", set(), set())


def test_inside_reference_destination_uses_observed_object_pose():
    from dataclasses import replace
    from types import SimpleNamespace

    from examples.so101_datagen.tasks import footprint

    cfg = GenerationConfig()
    cfg.tasks.weights = {"ordered_transfers": 1}
    ep = prepare(2, cfg, load_meshes())
    first, second = ep.instance.transfers
    assert second.target_name == first.object_name
    assert second.container_name == first.target_name
    container = find(ep.scene, second.container_name)
    reference = find(ep.scene, second.target_name)
    observed_position = np.asarray(first.position) + np.array([0.002, -0.002])
    observed = replace(
        ep.scene,
        objects=[
            replace(obj, pos=observed_position.tolist()) if obj.name == reference.name else obj
            for obj in ep.scene.objects
        ],
    )
    planner = SimpleNamespace(
        rollout=SimpleNamespace(scene_now=lambda: observed), workspace=cfg.layout.radius
    )
    TASKS[ep.task].prepare_transfer(second, ep.scene, planner, np.random.default_rng(0))
    moving = find(ep.scene, second.object_name)
    c, s = np.cos(container.yaw), np.sin(container.yaw)
    local = np.array([[c, s], [-s, c]]) @ (np.array(second.position) - container.pos)
    assert np.all(np.abs(local) + footprint(moving) < container.inner_half_xy)
    distance = np.linalg.norm(np.array(second.position) - observed_position)
    assert (
        footprint(moving) + footprint(reference) <= distance < footprint(moving) + footprint(reference) + 0.04
    )
    assert np.linalg.norm(np.array(second.position) - reference.pos) > 0.08


@pytest.mark.parametrize("style", ["object_aligned", "tangential"])
def test_grasp_support_height_changes_world_pose_without_changing_carry_offset(style):
    from dataclasses import replace

    from examples.so101_datagen.motion import grasp_for

    arm = Arm(arm_spec().compile())
    obj = ObjectParams(
        "object",
        "cylinder",
        "cylinder",
        [0.015, 0.025],
        Look([1, 1, 1, 1]),
        0.04,
        1.0,
        [0.25, 0],
        0,
    )

    def grasp(item):
        if style == "tangential":
            return tangential_grasp_for(arm, item, 30, np.random.default_rng(0), 0.005, 0.025)
        return grasp_for(arm, item, 30, 0, 0.005)

    low, high = grasp(obj), grasp(replace(obj, support_height=0.035))
    np.testing.assert_allclose(high.pos - low.pos, [0, 0, 0.035], atol=1e-12)
    np.testing.assert_allclose(high.point - low.point, [0, 0, 0.035], atol=1e-12)
    assert high.height == pytest.approx(low.height)


def test_object_resting_on_raised_support_does_not_count_as_lifted():
    from types import SimpleNamespace

    from examples.so101_datagen.scene import _add_object
    from examples.so101_datagen.tasks import is_lifted

    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.015], pos=[0.2, 0, 0.015])
    spec.worldbody.add_site(name="gripperframe", pos=[0.2, 0, 0.05])
    obj = ObjectParams(
        "object",
        "block",
        "box",
        [0.01, 0.01, 0.02],
        Look([1, 1, 1, 1]),
        0.04,
        1.0,
        [0.2, 0],
        0,
        support_height=0.03,
    )
    _add_object(spec, obj, {})
    model = spec.compile()
    body = model.body(obj.name).id
    address = model.jnt_qposadr[model.body_jntadr[body]]
    assert model.qpos0[address + 2] == pytest.approx(0.052)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1000)
    info = SimpleNamespace(
        params=SimpleNamespace(objects=[obj], containers=[]), object_bodies={obj.name: body}
    )
    assert data.xpos[body, 2] > obj.half_size[-1] + 0.015
    assert not is_lifted(obj.name, model, data, info)
    data.qpos[address + 2] += 0.02
    mujoco.mj_forward(model, data)
    assert is_lifted(obj.name, model, data, info)


def test_clear_container_preserves_goals_and_retries_above_the_tray_floor(monkeypatch):
    from types import SimpleNamespace

    from examples.so101_datagen.tasks.collect_objects import inside_container

    cfg = GenerationConfig()
    cfg.tasks.weights = {"clear_container": 1}
    cfg.recovery.miss_prob = 0
    ep = prepare(4, cfg, load_meshes())
    task = TASKS[ep.task]
    initial = Rollout(ep, task, cfg.fps)
    tray = ep.scene.containers[0]
    assert tray.wall_height <= cfg.container.clearing_wall_height[1]
    for transfer in ep.instance.transfers:
        obj = find(ep.scene, transfer.object_name)
        assert obj.support_height == tray.thickness
        assert inside_container(obj.name, tray.name, ep.model, initial.data, ep.info)
        assert not task.transfer_success(transfer.id, ep.instance, ep.model, initial.data, ep.info)
    original = Planner._pick_once
    missed = []

    def miss_second(planner, obj):
        if planner.transfer_id == ep.instance.transfers[1].id and not missed:
            missed.append(obj.name)
            planner._miss_mode = "high"
            planner._miss_distance = 0.03
        return original(planner, obj)

    monkeypatch.setattr(Planner, "_pick_once", miss_second)
    simulate(ep, task, cfg.fps)
    assert missed
    assert len(ep.plan.completed_transfers) == 2
    second_attempts = [a for a in ep.plan.attempts if a.transfer_id == ep.instance.transfers[1].id]
    assert [a.success for a in second_attempts] == [False, True]
    data = _final_data(ep)
    assert task.success(ep.instance, ep.model, data, ep.info)
    rollout = Rollout(ep, task, cfg.fps)
    rollout.data.qpos[:] = ep.qpos[-1]
    mujoco.mj_forward(ep.model, rollout.data)
    rollout.planner = SimpleNamespace(completed_transfers=ep.plan.completed_transfers)
    rollout.check_completed()
    for transfer in ep.instance.transfers:
        original_object = find(ep.scene, transfer.object_name)
        current = rollout.object_now(original_object)
        # A cylinder can settle slightly tilted; the observed base must still drop below the tray floor.
        assert current is not None and current.support_height < original_object.support_height
    # A completed object rolling into the next reserved goal causes a local destination refresh.
    first_transfer, next_transfer = ep.instance.transfers
    occupied = rollout.data.xpos[ep.info.object_bodies[first_transfer.object_name], :2].copy()
    next_transfer.position = occupied.tolist()
    task.prepare_transfer(
        next_transfer,
        ep.scene,
        SimpleNamespace(rollout=rollout, workspace=cfg.layout.radius),
        np.random.default_rng(0),
    )
    assert np.linalg.norm(np.array(next_transfer.position) - occupied) > 0.03
    # Both completed placements are necessary. Returning one object to its original tray pose fails.
    first = ep.instance.transfers[0]
    body = ep.info.object_bodies[first.object_name]
    address = ep.model.jnt_qposadr[ep.model.body_jntadr[body]]
    rollout.data.qpos[address : address + 7] = ep.model.qpos0[address : address + 7]
    mujoco.mj_forward(ep.model, rollout.data)
    assert not task.success(ep.instance, ep.model, rollout.data, ep.info)
    with pytest.raises(EpisodeError, match="completed_goal_disturbed"):
        rollout.check_completed()
    # An unrelated object left in the tray also prevents whole-task success.
    data = _final_data(ep)
    moved = {t.object_name for t in ep.instance.transfers}
    other = next(obj for obj in ep.scene.objects if obj.name not in moved)
    body = ep.info.object_bodies[other.name]
    address = ep.model.jnt_qposadr[ep.model.body_jntadr[body]]
    data.qpos[address : address + 3] = [*tray.pos, tray.thickness + other.half_size[-1]]
    mujoco.mj_forward(ep.model, data)
    assert all(
        task.transfer_success(t.id, ep.instance, ep.model, data, ep.info) for t in ep.instance.transfers
    )
    assert not task.success(ep.instance, ep.model, data, ep.info)


def test_imported_containers_require_complete_local_preparation(tmp_path, monkeypatch):
    from examples.so101_datagen import scene

    monkeypatch.setattr(scene, "__file__", str(tmp_path / "scene.py"))
    monkeypatch.setattr(
        scene, "container_asset", lambda style: {"albedo": "albedo.png", "collision_parts": 2}
    )
    root = tmp_path / "container_assets" / "white_dish"
    root.mkdir(parents=True)
    for name in ("visual.obj", "albedo.png", "collision_000.obj"):
        (root / name).touch()
    with pytest.raises(FileNotFoundError, match="prepare_containers.py --styles white_dish"):
        scene.container_asset_directory("white_dish")
    (root / "collision_001.obj").touch()
    assert scene.container_asset_directory("white_dish") == root
