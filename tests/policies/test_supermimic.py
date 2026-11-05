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

"""SuperMimic shared core: dynamics spec, reference builder, observation and export contract."""

import json
import math

import numpy as np
import pytest
import torch

from lerobot.policies.supermimic import (
    GOAL_FRAME_DIM,
    OBS_DIM,
    MotionReference,
    ObservationHistory,
    g1_29dof_spec,
    load_export,
    proprio_observation,
)
from lerobot.policies.supermimic.export import validate_manifest
from lerobot.policies.supermimic.geometry import mat_to_6d, quat_to_mat, yaw_mat, z_rotation
from tests.mocks.mock_supermimic import make_clip, make_export, make_manifest


def test_g1_spec_gains_follow_armature():
    spec = g1_29dof_spec()
    omega = 10 * 2.0 * 3.1415926535
    knee = spec.joint_names.index("left_knee_joint")
    assert spec.kp[knee] == pytest.approx(spec.armature[knee] * omega**2)
    assert spec.kd[knee] == pytest.approx(4.0 * spec.armature[knee] * omega)
    assert spec.action_scale[knee] == pytest.approx(0.25 * spec.effort_limit[knee] / spec.kp[knee])
    assert spec.decimation == 4


def test_action_to_target_is_unclipped_affine_map():
    spec = g1_29dof_spec()
    action = np.full(29, 100.0, dtype=np.float32)
    offset = np.linspace(-0.1, 0.1, 29, dtype=np.float32)
    target = spec.action_to_target(action, offset)
    np.testing.assert_allclose(
        target, spec.array("default_q") + offset + 100.0 * spec.array("action_scale"), rtol=1e-6
    )


def test_check_manifest_rejects_mismatched_gains():
    spec = g1_29dof_spec()
    manifest = make_manifest()
    spec.check_manifest(manifest)
    manifest["robot"]["kp"][3] *= 1.01
    with pytest.raises(ValueError, match="kp"):
        spec.check_manifest(manifest)


def test_validate_manifest_rejects_reordered_observation():
    manifest = make_manifest()
    fields = manifest["inputs"]["observation_fields"]
    fields[0], fields[1] = fields[1], fields[0]
    with pytest.raises(ValueError, match="observation layout"):
        validate_manifest(manifest, g1_29dof_spec())


def test_load_export_checks_checksum_and_golden_vectors(tmp_path):
    directory = make_export(tmp_path / "export")
    export = load_export(directory)
    assert export.history_length == 10 and export.goal_frames == 10
    assert export.check_golden_vectors() < 1e-6

    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["artifacts"]["actor"]["sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_export(directory)


def test_reference_resamples_to_control_rate(tmp_path):
    clip = make_clip(tmp_path / "clip.csv", frames=61, fps=30.0, speed=0.5, yaw=0.3)
    reference = MotionReference.from_csv([clip], num_joints=29)
    assert reference.num_frames == math.floor(2.0 / 0.02) + 1
    middle = reference.num_frames // 2
    torch.testing.assert_close(reference.root_v[middle, :2].norm(), torch.tensor(0.5), atol=1e-4, rtol=0)
    torch.testing.assert_close(
        reference.root_v_local[middle], torch.tensor([0.5, 0.0, 0.0]), atol=1e-4, rtol=0
    )


def test_goal_layout_and_clamping(tmp_path):
    reference = MotionReference.from_csv([make_clip(tmp_path / "clip.csv")], num_joints=29)
    last = reference.num_frames - 1
    goal = reference.goal(torch.tensor([0, last]), 10).reshape(2, 10, GOAL_FRAME_DIM)
    # Displacement of frame f+k in the reference heading frame: straight ahead along x.
    assert goal[0, 4, 0] == pytest.approx(5 * 0.02 * 0.5, abs=1e-4)
    assert goal[0, :, 1].abs().max() < 1e-4
    # Past the clip end every future frame is the last one: zero displacement.
    assert goal[1, :, :3].abs().max() == 0
    torch.testing.assert_close(goal[0, 0, 15:44], reference.q[1])
    torch.testing.assert_close(goal[0, 0, 44:73], 0.1 * reference.qd[1])


def test_static_goal_holds_pose(tmp_path):
    reference = MotionReference.from_csv([make_clip(tmp_path / "clip.csv")], num_joints=29)
    goal = reference.static_goal(torch.tensor([7]), 10).reshape(10, GOAL_FRAME_DIM)
    assert goal[:, :3].abs().max() == 0 and goal[:, 9:15].abs().max() == 0 and goal[:, 44:].abs().max() == 0
    torch.testing.assert_close(goal[:, 15:44], reference.q[7].expand(10, -1))
    heading = yaw_mat(reference.root_R[7])
    torch.testing.assert_close(goal[0, 3:9], mat_to_6d(heading.T @ reference.root_R[7]))


def test_proprio_observation_layout():
    spec = g1_29dof_spec()
    default_q = torch.from_numpy(spec.array("default_q"))
    rotation = z_rotation(torch.tensor(0.4)) @ quat_to_mat(torch.tensor([0.995, 0.0998, 0.0, 0.0]))
    q, dq, gyro, previous = (
        default_q + 0.1,
        torch.full((29,), 0.5),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.ones(29),
    )
    obs = proprio_observation(rotation, gyro, q, dq, previous, rotation, default_q)
    assert obs.shape == (OBS_DIM,)
    torch.testing.assert_close(obs[:29], torch.full((29,), 0.1), atol=1e-6, rtol=0)
    torch.testing.assert_close(obs[58:61], gyro)
    torch.testing.assert_close(obs[61:64], -rotation[2])
    torch.testing.assert_close(obs[93:99], mat_to_6d(torch.eye(3)), atol=1e-6, rtol=0)


def test_observation_history_seeds_and_rolls():
    history = ObservationHistory(3, obs_dim=2)
    first = history.push(torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(first, torch.ones(3, 2))
    rolled = history.push(torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(rolled[:, 0], torch.tensor([1.0, 1.0, 2.0]))
    history.reset()
    torch.testing.assert_close(history.push(torch.tensor([3.0, 3.0])), torch.full((3, 2), 3.0))
