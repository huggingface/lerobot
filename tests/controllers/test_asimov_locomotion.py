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

"""The Asimov locomotion controller against a stand-in ONNX file with Menlo's metadata."""

import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from lerobot.controllers.asimov import locomotion  # noqa: E402
from lerobot.robots.asimov.asimov import ASIMOV_JOINTS  # noqa: E402

LEG_DEFAULTS = [-0.15, 0.0, 0.0, 0.45, -0.3, 0.0, 0.15, 0.0, 0.0, -0.45, 0.3, 0.0]
ACTION_SCALE = [0.25] * 5 + [0.08] + [0.25] * 5 + [0.08]


def metadata(**overrides):
    clip = [[None, None]] * 5 + [[-0.09, 0.09]] + [[None, None]] * 5 + [[-0.09, 0.09]]
    meta = {
        "joint_names": list(locomotion.POLICY_JOINTS),
        "observation_names": list(locomotion.POLICY_OBSERVATIONS),
        "obs_dim": 45,
        "action_dim": 12,
        "policy_rate_hz": 50.0,
        "action_scale": ACTION_SCALE,
        "action_offset": LEG_DEFAULTS,
        "action_clip": clip,
        "joint_stiffness": [150.0] * 4 + [224.42, 35.2] + [150.0] * 4 + [224.42, 35.2],
        "joint_damping": [5.0] * 4 + [10.201, 1.6] + [5.0] * 4 + [10.201, 1.6],
        "robot_model_check": "matched",
    }
    meta.update(overrides)
    return {key: value if isinstance(value, str) else json.dumps(value) for key, value in meta.items()}


def write_policy(path, weights, meta):
    """actions = obs @ weights, with Menlo's input/output names and metadata."""
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "w"], ["actions"])],
        "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 45])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 12])],
        [numpy_helper.from_array(weights.astype(np.float32), "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for key, value in meta.items():
        model.metadata_props.add(key=key, value=value)
    onnx.save(model, path)


@pytest.fixture
def make_controller(tmp_path, monkeypatch):
    def make(weights=None, **meta):
        path = tmp_path / "policy.onnx"
        write_policy(path, np.zeros((45, 12)) if weights is None else weights, metadata(**meta))
        monkeypatch.setattr(locomotion, "hf_hub_download", lambda repo_id, filename: str(path))
        return locomotion.AsimovLocomotionController("test/policy")

    return make


def test_gains_and_default_pose_cover_the_whole_body(make_controller):
    controller = make_controller()
    assert controller.control_dt == pytest.approx(0.02)
    assert controller.default_pose[:12] == pytest.approx(LEG_DEFAULTS)
    assert controller.default_pose[ASIMOV_JOINTS.index("L_Elbow")] == pytest.approx(0.4)
    assert (controller.kp > 0).all() and (controller.kd > 0).all()


def test_rejects_a_checkpoint_with_another_layout(make_controller):
    with pytest.raises(ValueError, match="joint_names"):
        make_controller(joint_names=list(reversed(locomotion.POLICY_JOINTS)))
    with pytest.raises(ValueError, match="obs_dim"):
        make_controller(obs_dim=47)


def test_observation_layout(make_controller):
    controller = make_controller()
    q = np.zeros(len(ASIMOV_JOINTS))
    q[:12] = LEG_DEFAULTS
    q[:12] += np.arange(12) / 100  # Offsets from the default pose, in policy joint order.
    dq = np.zeros(len(ASIMOV_JOINTS))
    dq[:12] = np.arange(12)
    waist = ASIMOV_JOINTS.index("Waist_Yaw")
    q[waist], dq[waist] = np.pi / 2, 1.0
    controller.last_action[:] = 7.0

    obs = controller.observation(
        q, dq, gyro=[1.0, 0.0, 0.0], gravity=[1.0, 0.0, 0.0], command=[0.3, 0.1, 0.2]
    )

    # Pelvis-frame IMU turned into the torso frame: +90 deg waist yaw maps pelvis x to torso -y.
    assert obs[0:3] == pytest.approx(np.array([0.0, -1.0, 1.0]) * 0.25, abs=1e-6)
    assert obs[3:6] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
    assert obs[6:9] == pytest.approx([0.3, 0.1, 0.2])
    slots = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
    assert obs[9:21] == pytest.approx(np.array(slots) / 100)
    assert obs[21:33] == pytest.approx(np.array(slots) * 0.1)
    assert obs[33:45] == pytest.approx([7.0] * 12)


def test_step_scales_clips_and_passes_upper_body_through(make_controller):
    weights = np.zeros((45, 12))
    weights[6, :] = 10.0  # Every action follows the vx command.
    controller = make_controller(weights)
    q = controller.default_pose.astype(np.float64)
    upper = controller.default_pose.copy()
    upper[ASIMOV_JOINTS.index("L_Elbow")] = 1.2

    targets = controller.run_step(q, np.zeros_like(q), [0, 0, 0], [0, 0, -1], [0.1, 0, 0], upper)

    assert controller.last_action == pytest.approx([1.0] * 12)
    assert targets[0] == pytest.approx(LEG_DEFAULTS[0] + 0.25)
    assert targets[5] == pytest.approx(0.08)
    assert targets[12:] == pytest.approx(upper[12:])
    controller.run_step(q, np.zeros_like(q), [0, 0, 0], [0, 0, -1], [1.0, 0, 0], upper)
    assert controller.run_step(q, np.zeros_like(q), [0, 0, 0], [0, 0, -1], [1.0, 0, 0], upper)[
        5
    ] == pytest.approx(0.09)
    controller.reset()
    assert not controller.last_action.any()
