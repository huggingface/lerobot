# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict
from unittest.mock import MagicMock

import draccus
import numpy as np
import pytest

from lerobot.rollout.end_effector import EndEffectorConfig, EndEffectorKinematics, ToolCameraMount, parse_pose
from lerobot.rollout.hybrid import HybridConfig, HybridPlanner, InterventionLimit, PlannerDecision
from lerobot.rollout.inference import PolicyQuery, QueryKind
from lerobot.rollout.inference.hybrid import HybridInferenceEngine
from lerobot.rollout.planner import PlannerConfig


@pytest.fixture
def arm(tmp_path):
    pytest.importorskip("mujoco")
    model = tmp_path / "planar.xml"
    model.write_text("""<mujoco><compiler angle="radian"/>
    <default><joint axis="0 0 1" range="-3 3"/><geom type="sphere" size=".01"/></default>
    <worldbody><body><joint name="a"/><geom/>
    <body pos=".2 0 0"><joint name="b"/><geom/>
    <body pos=".2 0 0"><joint name="c"/><geom/><site name="tip" pos=".08 0 0"/>
    </body></body></body></worldbody></mujoco>""")
    ee = EndEffectorConfig(
        str(model), "tip", ["a", "b", "c"], ["a.pos", "b.pos", "c.pos"], "Fixed model base, z up"
    )
    limits = {k: InterventionLimit(-3, 3, 0.25, 0.2, 0.04) for k in ee.action_keys}
    limits["gripper.pos"] = InterventionLimit(0, 1, 1, 1, 0.04)
    config = HybridConfig(limits=limits, end_effectors={"arm": ee})
    pose = dict(zip(ee.action_keys, [0.3, 0.6, -0.4], strict=True)) | {"gripper.pos": 0.4}
    solver = EndEffectorKinematics(ee)
    return config, pose, solver


def proposal(target, **kwargs):
    return PlannerDecision.parse(
        {
            "mode": "end_effector",
            "scene": "Clear space",
            "reason": "Small correction",
            "instruction": "",
            "targets": {},
            "duration_s": 2,
            "ee_targets": {"arm": target},
            **kwargs,
        }
    )


def test_fk_matches_analytic_planar_chain_and_config_round_trip(arm):
    config, pose, solver = arm
    point, rotation = parse_pose(solver.forward(pose))
    angles = np.cumsum([pose[k] for k in config.end_effectors["arm"].action_keys])
    assert point == pytest.approx(
        [np.dot([0.2, 0.2, 0.08], np.cos(angles)), np.dot([0.2, 0.2, 0.08], np.sin(angles)), 0]
    )
    assert rotation.as_rotvec() == pytest.approx([0, 0, angles[-1]])
    assert draccus.decode(HybridConfig, asdict(config)) == config


def test_camera_mount_composes_translation_and_rotation_in_tool_frame(arm):
    config, pose, solver = arm
    solver.config.camera_mounts["wrist"] = ToolCameraMount([0.1, 0, 0], [1, 0, 0, 0], "Measured fixture")
    pose.update({"a.pos": np.pi / 2, "b.pos": 0, "c.pos": 0})
    result = solver.camera_poses(pose)["wrist"]
    transform = np.asarray(result["T_base_from_camera"])
    assert transform[:3, 3] == pytest.approx([0, 0.58, 0], abs=1e-10)
    assert transform[:3, 0] == pytest.approx([0, 1, 0], abs=1e-10)
    assert np.linalg.inv(transform) @ (transform @ [0.03, 0.02, 0.3, 1]) == pytest.approx(
        [0.03, 0.02, 0.3, 1]
    )
    assert result["estimated"]
    assert draccus.decode(HybridConfig, asdict(config)) == config
    planner = HybridPlanner(PlannerConfig(), "mock", hybrid=config, client=MagicMock())
    blocks = planner.observation_blocks("Current", pose)
    assert any("T_base_from_camera" in block.get("text", "") for block in blocks)


@pytest.mark.parametrize(
    "position,quaternion", [([float("nan"), 0, 0], [1, 0, 0, 0]), ([0, 0, 0], [2, 0, 0, 0])]
)
def test_invalid_camera_mount_rejected(position, quaternion):
    with pytest.raises(ValueError):
        ToolCameraMount(position, quaternion, "Invalid fixture")


def test_proposal_fk_uses_ordered_postprocessed_joint_targets(arm):
    config, pose, _ = arm
    config.review_policy_chunks = True
    planner = HybridPlanner(PlannerConfig(), "mock", hybrid=config, client=MagicMock())
    keys = ["gripper.pos", "c.pos", "a.pos", "b.pos"]
    obs = pose | {
        "_hybrid_proposal": {
            "id": 1,
            "action_keys": keys,
            "actions": [[0.7, 0.0, 0.0, 0.0], [0.2, 0.0, np.pi / 2, 0.0]],
            "execute_steps": 1,
            "fps": 30,
            "task": "Move",
        }
    }
    context = planner.proposal_context(obs)
    poses = context["end_effector_trajectory"]
    assert poses[0]["arm"]["position_m"] == pytest.approx([0.48, 0, 0])
    assert poses[1]["arm"]["position_m"] == pytest.approx([0, 0.48, 0])
    assert context["actions"][0][0] == 0.7  # Gripper stays in native opening units.
    assert len(poses) == 2  # Review includes the suffix, even though only one step is authorized.


def test_fk_ik_round_trip_preserves_uncommanded_gripper(arm):
    config, pose, solver = arm
    target = solver.forward(pose | {"b.pos": pose["b.pos"] + 0.025})
    resolved = proposal(target).resolve_motion(config, pose, {"arm": solver})
    assert solver.reached(target, resolved)
    assert resolved["gripper.pos"] == pose["gripper.pos"]
    changed = proposal(target, targets={"gripper.pos": 1}).resolve_motion(config, pose, {"arm": solver})
    assert changed["gripper.pos"] == 1


@pytest.mark.parametrize(
    "case",
    [
        "translation",
        "rotation",
        "unreachable",
        "speed",
        "joint_delta",
        "joint_speed",
        "raw_joint",
        "unknown",
        "two_arms",
    ],
)
def test_invalid_cartesian_proposals_rejected_atomically(arm, case):
    config, pose, solver = arm
    target = solver.forward(pose | {"b.pos": pose["b.pos"] + 0.025})
    request = proposal(target)
    if case == "translation":
        target["position_m"][0] += 0.5
    elif case == "rotation":
        target["quaternion_wxyz"] = [0, 1, 0, 0]
    elif case == "unreachable":
        target["position_m"][2] += 0.01
    elif case == "speed":
        request = proposal(target, duration_s=0.01)
    elif case == "joint_delta":
        for key in config.end_effectors["arm"].action_keys:
            config.limits[key].max_delta = 0
    elif case == "joint_speed":
        for key in config.end_effectors["arm"].action_keys:
            config.limits[key].max_speed = 0.00001
    elif case == "raw_joint":
        request = proposal(target, targets={"a.pos": 0.3})
    elif case == "unknown":
        request = proposal(target, ee_targets={"unknown": target})
    else:
        request = proposal(target, ee_targets={"arm": target, "other": target})
    with pytest.raises(ValueError):
        request.resolve_motion(config, pose, {"arm": solver})


@pytest.mark.parametrize("value", [[0, 0, 0, 0], [2, 0, 0, 0], [True, 0, 0, 0], [float("nan"), 0, 0, 0]])
def test_bad_quaternions_rejected(value):
    with pytest.raises(ValueError):
        proposal({"position_m": [0, 0, 0], "quaternion_wxyz": value})


def test_cartesian_mode_is_opt_in_and_raw_arm_joint_corrections_disabled(arm):
    config, pose, solver = arm
    target = solver.forward(pose)
    with pytest.raises(ValueError, match="Unknown end effector"):
        proposal(target).resolve_motion(HybridConfig(limits=config.limits), pose, {})
    with pytest.raises(ValueError, match="Use end_effector"):
        PlannerDecision("intervention", "clear", "move", "", {"a.pos": 0.31}, 2).validate_motion(config, pose)


def test_planner_receives_fk_and_explicit_frames(arm):
    config, pose, _ = arm
    planner = HybridPlanner(PlannerConfig(), "planar", hybrid=config, client=MagicMock())
    blocks = planner.observation_blocks("current", pose)
    assert "Measured end-effector poses from FK" in blocks[-1]["text"]
    prompt = planner.request_text(PolicyQuery(QueryKind.NEXT_SUBTASK, "Move"), "Move")
    assert "quaternion_wxyz" in prompt and "Fixed model base, z up" in prompt


def test_ik_executes_only_after_acceptance_and_cartesian_arrival_is_checked(arm, monkeypatch):
    config, pose, solver = arm
    clock = [100.0]
    monkeypatch.setattr("lerobot.rollout.inference.hybrid.time.perf_counter", lambda: clock[0])
    delegate = MagicMock(task="Move", ready=True, failed=False)
    engine = HybridInferenceEngine(delegate, config, list(config.limits), 1)
    engine.resume()
    engine.notify_observation(pose)
    engine.get_action({})
    target = solver.forward(pose | {"b.pos": pose["b.pos"] + 0.025})
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal)
    engine._resolve_query(query, pose, lambda *args: proposal(target), epoch=engine._query_epoch)
    assert engine._target is None
    engine.get_action({})
    assert engine._mode == "intervention"
    clock[0] += 2
    engine.notify_observation(pose)  # Joint tolerance alone would incorrectly count this as reached.
    engine.get_action({})
    assert engine._mode == "intervention"
    engine.notify_observation(engine._target)
    engine.get_action({})
    assert engine._mode == "review"
    delegate.get_action.assert_not_called()


def test_stale_cartesian_reply_after_reset_cannot_move(arm):
    config, pose, solver = arm
    delegate = MagicMock(task="Move", ready=True, failed=False)
    engine = HybridInferenceEngine(delegate, config, list(config.limits), 1)
    engine.resume()
    engine.notify_observation(pose)
    engine.get_action({})
    epoch = engine._query_epoch
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal)
    engine.reset()
    engine._resolve_query(query, pose, lambda *args: proposal(solver.forward(pose)), epoch=epoch)
    assert engine._pending_decision is None
    assert engine._mode == "idle"


def test_ik_starts_from_measured_snapshot_not_pre_settle_command(arm):
    config, pose, solver = arm
    delegate = MagicMock(task="Move", ready=True, failed=False)
    engine = HybridInferenceEngine(delegate, config, list(config.limits), 1)
    engine.resume()
    engine.notify_observation(pose)
    engine.get_action({})
    engine._hold["a.pos"] += 0.1  # Commanded setpoint differs from settled measured position.
    target = solver.forward(pose | {"b.pos": pose["b.pos"] + 0.025})
    engine._resolve_query(
        PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal),
        dict(pose),
        lambda *args: proposal(target),
        epoch=engine._query_epoch,
    )
    action = engine.get_action({})
    assert not engine.terminal
    assert engine._hold["a.pos"] == pose["a.pos"]
    assert action.tolist() == pytest.approx(list(pose.values()), abs=0.003)
    assert solver.reached(target, engine._target)
