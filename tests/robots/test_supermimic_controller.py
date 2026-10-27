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

"""SuperMimicController on a synthetic export: phases, joint mapping, inputs and wiring."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.supermimic import MotionReference, load_export, proprio_observation
from lerobot.policies.supermimic.geometry import quat_to_mat, z_rotation
from lerobot.robots.unitree_g1.config_unitree_g1 import SuperMimicControllerConfig, UnitreeG1Config
from lerobot.robots.unitree_g1.controllers import supermimic_tracking as smt
from lerobot.robots.unitree_g1.g1_utils import NUM_MOTORS, G1_29_JointIndex
from tests.mocks.mock_supermimic import make_clip, make_export


@pytest.fixture
def export(tmp_path):
    return load_export(make_export(tmp_path / "export"))


@pytest.fixture
def clip(tmp_path):
    return make_clip(tmp_path / "clip.csv")


@pytest.fixture
def reference(clip):
    return MotionReference.from_csv([clip], num_joints=29)


@pytest.fixture
def motions_dir(tmp_path):
    directory = tmp_path / "motions"
    directory.mkdir()
    # 4 source frames at 30 fps resample to 6 control frames, 7 to 11.
    make_clip(directory / "b_turn.csv", frames=7, yaw=-0.5)
    make_clip(directory / "a_walk.csv", frames=4)
    return directory


@pytest.fixture
def clips(motions_dir):
    return MotionReference.from_csv(smt.motion_files(motions_dir), num_joints=29)


def remote(*buttons):
    keys = sum(buttons)
    return bytes([0, 0, keys & 0xFF, keys >> 8]) + bytes(36)


def lowstate(q_slots, yaw=0.0, gyro=(0.0, 0.0, 0.0), keys=()):
    quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    return SimpleNamespace(
        motor_state=[SimpleNamespace(q=float(q), dq=0.0) for q in q_slots],
        imu_state=SimpleNamespace(quaternion=quat, gyroscope=list(gyro)),
        wireless_remote=remote(*keys),
    )


def test_dds_joint_names_cover_g1_29():
    from lerobot.policies.supermimic import G1_JOINT_NAMES

    assert sorted(smt.dds_joint_name(name) for name in G1_JOINT_NAMES) == sorted(
        j.name for j in G1_29_JointIndex
    )


def test_gains_and_home_pose_in_dds_order(export, reference):
    controller = smt.SuperMimicController(export, reference)
    knee = export.spec.joint_names.index("right_knee_joint")
    assert controller.kp[G1_29_JointIndex.kRightKnee] == pytest.approx(export.spec.kp[knee])
    assert controller.kd[G1_29_JointIndex.kRightKnee] == pytest.approx(export.spec.kd[knee])
    elbow = export.spec.joint_names.index("left_elbow_joint")
    assert controller.default_angles[G1_29_JointIndex.kLeftElbow] == pytest.approx(
        float(reference.q[0, elbow])
    )
    pose = controller.default_base_pose
    assert pose[:2].tolist() == [0.0, 0.0] and pose[2] == pytest.approx(0.79)


def test_motion_files_sorts_a_directory_and_accepts_one_file(motions_dir, clip, tmp_path):
    assert [path.name for path in smt.motion_files(motions_dir)] == ["a_walk.csv", "b_turn.csv"]
    assert smt.motion_files(clip) == [clip]
    with pytest.raises(FileNotFoundError):
        smt.motion_files(tmp_path / "missing")


def test_hold_then_play_then_done(export, clips):
    controller = smt.SuperMimicController(export, clips)
    state = lowstate(controller.default_angles)
    for _ in range(3):
        controller.run_step({}, state)
    assert controller.phase == smt.Phase.HOLD and controller.frame == 0

    for expected in range(1, 6):
        controller.run_step({smt.PLAY_KEY: 1.0}, state)
        assert controller.phase == smt.Phase.TRACK and controller.frame == expected
    controller.run_step({smt.PLAY_KEY: 1.0}, state)
    assert controller.phase == smt.Phase.DONE and controller.frame == 5
    assert controller.observation_state()[smt.DONE_KEY] == 1.0

    controller.reset()
    assert controller.phase == smt.Phase.HOLD and controller.frame == 0


def test_remote_buttons_cycle_clips_and_play(export, clips):
    controller = smt.SuperMimicController(export, clips)
    home = controller.default_angles

    controller.run_step({}, lowstate(home, keys=[smt.REMOTE_R1]))
    assert controller.clip_name == "b_turn" and controller.frame == 6
    # Holding the button does not repeat it.
    controller.run_step({}, lowstate(home, keys=[smt.REMOTE_R1]))
    assert controller.clip_name == "b_turn" and controller.phase == smt.Phase.HOLD

    controller.run_step({}, lowstate(home, keys=[smt.REMOTE_START]))
    assert controller.phase == smt.Phase.TRACK and controller.frame == 7
    assert controller.observation_state()[smt.FRAME_KEY] == 1.0

    controller.run_step({}, lowstate(home))
    controller.run_step({}, lowstate(home, keys=[smt.REMOTE_R1]))
    assert controller.clip_name == "a_walk" and controller.phase == smt.Phase.HOLD
    controller.run_step({}, lowstate(home))
    controller.run_step({}, lowstate(home, keys=[smt.REMOTE_L1]))
    assert controller.clip_name == "b_turn"
    assert controller.observation_state()[smt.CLIP_KEY] == 1.0


def test_start_replays_a_finished_clip(export, clips):
    controller = smt.SuperMimicController(export, clips, autoplay=True)
    home = controller.default_angles
    for _ in range(6):
        controller.run_step({}, lowstate(home))
    assert controller.phase == smt.Phase.DONE
    controller.run_step({}, lowstate(home, keys=[smt.REMOTE_START]))
    assert controller.phase == smt.Phase.TRACK and controller.frame == 1


def test_auto_advance_wraps_around_the_clips(export, clips):
    controller = smt.SuperMimicController(export, clips, autoplay=True, auto_advance=True)
    home = controller.default_angles
    for _ in range(6):
        controller.run_step({}, lowstate(home))
    assert controller.clip_name == "b_turn" and controller.phase == smt.Phase.TRACK
    for _ in range(11):
        controller.run_step({}, lowstate(home))
    assert controller.clip_name == "a_walk" and controller.frame == 0


def test_start_clip_by_name_sets_the_home_pose(export, clips):
    controller = smt.SuperMimicController(export, clips, start_clip="b_turn")
    first, _ = clips.clip_frames("b_turn")
    elbow = export.spec.joint_names.index("left_elbow_joint")
    assert controller.frame == first
    assert controller.default_angles[G1_29_JointIndex.kLeftElbow] == pytest.approx(
        float(clips.q[first, elbow])
    )
    with pytest.raises(ValueError, match="Unknown clip"):
        smt.SuperMimicController(export, clips, start_clip="nope")


def test_run_step_reproduces_actor_inputs_and_action_mapping(export, reference):
    controller = smt.SuperMimicController(export, reference, autoplay=True)
    spec = export.spec
    q_actor = torch.from_numpy(spec.array("default_q")) + 0.02
    q_slots = np.zeros(NUM_MOTORS, dtype=np.float32)
    q_slots[controller.slots] = q_actor.numpy()
    yaw = 1.0
    targets = controller.run_step({}, lowstate(q_slots, yaw=yaw, gyro=(0.1, -0.2, 0.3)))

    rotation = quat_to_mat(torch.tensor([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=torch.float32))
    aligned = z_rotation(
        torch.tensor(yaw) - torch.atan2(reference.root_R[0, 1, 0], reference.root_R[0, 0, 0])
    )
    obs = proprio_observation(
        rotation,
        torch.tensor([0.1, -0.2, 0.3]),
        q_actor,
        torch.zeros(29),
        torch.zeros(29),
        aligned @ reference.root_R[0],
        torch.from_numpy(spec.array("default_q")),
    )
    expected_action = export.actor(obs.expand(10, -1).unsqueeze(0), reference.goal(torch.tensor([0]), 10))[0]
    expected_target = spec.action_to_target(expected_action.detach().numpy())
    for index, slot in enumerate(controller.slots):
        assert targets[f"{G1_29_JointIndex(slot).name}.q"] == pytest.approx(expected_target[index], abs=1e-5)
    assert set(targets) == {f"{joint.name}.q" for joint in G1_29_JointIndex}
    assert controller.latency_stats()["count"] == 1


def test_controller_rejects_clip_index_outside_reference(export, reference):
    with pytest.raises(ValueError, match="outside"):
        smt.SuperMimicController(export, reference, start_clip=1)


def test_robot_config_requires_export_and_motions():
    with pytest.raises(ValueError, match="supermimic.export"):
        UnitreeG1Config(controller="SuperMimicController")
    with pytest.raises(ValueError, match="motion_fps"):
        SuperMimicControllerConfig(motion_fps=0)
    with pytest.raises(ValueError, match="not supported"):
        UnitreeG1Config(
            controller="SuperMimicController",
            embodiment="g1_23",
            supermimic=SuperMimicControllerConfig(export="e", motions="m"),
        )


def test_make_robot_controller_builds_from_robot_config(tmp_path, motions_dir):
    from lerobot.robots.unitree_g1.unitree_g1 import make_robot_controller

    config = UnitreeG1Config(
        controller="SuperMimicController",
        supermimic=SuperMimicControllerConfig(
            export=str(make_export(tmp_path / "export")),
            motions=str(motions_dir),
            start_clip="b_turn",
            autoplay=True,
        ),
    )
    controller = make_robot_controller(config.controller, config)
    assert isinstance(controller, smt.SuperMimicController)
    assert controller.reference.names == ["a_walk", "b_turn"]
    assert controller.clip_name == "b_turn" and controller.phase == smt.Phase.TRACK
    assert controller.action_ft == {smt.PLAY_KEY: float}
