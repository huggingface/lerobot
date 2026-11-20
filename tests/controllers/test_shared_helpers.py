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

"""Tests for the checkpoint loading and state reading the controllers share. No network."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from lerobot.controllers.unitree_g1.proprio import read_proprio  # noqa: E402
from lerobot.robots.unitree_g1.g1_utils import NUM_MOTORS, get_gravity_orientation  # noqa: E402


@pytest.fixture
def onnx_file(tmp_path, monkeypatch):
    """A one-layer ONNX policy with JSON constants in its metadata, served in place of the Hub."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["obs"], ["action"])],
        "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    helper.set_model_props(model, {"kp": json.dumps([1.0, 2.0]), "kd": json.dumps([0.5])})
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)

    import lerobot.controllers.hub as hub

    downloads = []
    monkeypatch.setattr(
        hub, "hf_hub_download", lambda repo_id, filename: downloads.append((repo_id, filename)) or str(path)
    )
    return downloads


class TestLoadOnnxPolicy:
    def test_opens_the_session_on_one_thread(self, onnx_file):
        from lerobot.controllers.hub import load_onnx_policy

        session, constants = load_onnx_policy("org/policy", "policy.onnx")

        assert onnx_file == [("org/policy", "policy.onnx")]
        assert constants == {}
        options = session.get_session_options()
        assert (options.intra_op_num_threads, options.inter_op_num_threads) == (1, 1)
        obs = np.arange(3, dtype=np.float32).reshape(1, 3)
        np.testing.assert_array_equal(session.run(None, {"obs": obs})[0], obs)

    def test_reads_metadata_constants_as_float32(self, onnx_file):
        from lerobot.controllers.hub import load_onnx_policy

        _, constants = load_onnx_policy("org/policy", "policy.onnx", metadata_keys=("kp", "kd"))

        np.testing.assert_array_equal(constants["kp"], [1.0, 2.0])
        assert constants["kp"].dtype == np.float32
        np.testing.assert_array_equal(constants["kd"], [0.5])

    def test_missing_metadata_names_the_keys(self, onnx_file):
        from lerobot.controllers.hub import load_onnx_policy

        with pytest.raises(ValueError, match=r"missing \['default_angles'\]"):
            load_onnx_policy("org/policy", "policy.onnx", metadata_keys=("kp", "default_angles"))


def test_read_proprio_keeps_dds_slot_order():
    motors = [SimpleNamespace(q=0.1 * i, dq=0.01 * i) for i in range(NUM_MOTORS)]
    quat = [0.9, 0.1, 0.2, 0.3]
    lowstate = SimpleNamespace(
        motor_state=motors, imu_state=SimpleNamespace(quaternion=quat, gyroscope=[1, 2, 3])
    )

    proprio = read_proprio(lowstate)

    np.testing.assert_allclose(proprio.q, 0.1 * np.arange(NUM_MOTORS), rtol=1e-6)
    np.testing.assert_allclose(proprio.dq, 0.01 * np.arange(NUM_MOTORS), rtol=1e-6)
    assert proprio.q.dtype == proprio.dq.dtype == proprio.ang_vel.dtype == np.float32
    np.testing.assert_array_equal(proprio.ang_vel, [1, 2, 3])
    np.testing.assert_allclose(proprio.quat, quat, rtol=1e-6)
    np.testing.assert_array_equal(proprio.gravity, get_gravity_orientation(quat))
