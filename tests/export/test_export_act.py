#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from pathlib import Path

import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_act_policy_and_batch,
    to_numpy,
)


class TestACTExport:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.onnx").exists()

    @pytest.mark.slow
    def test_onnx_forward_pass(self, tmp_path: Path):
        from lerobot.export import export_policy, load

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        runtime = load(package_path, backend="onnx", device="cpu")

        obs_numpy = to_numpy(batch)
        action_chunk = runtime.predict_action_chunk(obs_numpy)

        assert action_chunk.ndim == 2
        assert action_chunk.shape[0] == policy.config.chunk_size
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action_chunk.shape[1] == action_dim

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import export_policy, load

        policy, batch = create_act_policy_and_batch()

        with torch.no_grad():
            torch.manual_seed(42)
            pytorch_output = policy.predict_action_chunk(batch)

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy)

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-3,
            atol=1e-4,
            msg="ACT ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_action_chunking_wrapper(self, tmp_path: Path):
        from lerobot.export import export_policy, load
        from lerobot.export.runtime import ActionChunkingWrapper

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        runtime = load(package_path, backend="onnx", device="cpu")
        wrapper = ActionChunkingWrapper(runtime)

        obs_numpy = to_numpy(batch)
        wrapper.reset()
        action = wrapper.select_action(obs_numpy)

        assert action.ndim == 1
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action.shape[0] == action_dim


class TestACTBackends:
    def test_onnx_backend_initialization(self, tmp_path: Path):
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        from lerobot.export.backends.onnx import ONNXBackend

        model_path = package_path / "artifacts" / "model.onnx"
        backend = ONNXBackend(model_path, device="cpu")

        assert backend.input_names is not None
        assert backend.output_names is not None
        assert len(backend.input_names) > 0
        assert len(backend.output_names) > 0

    def test_openvino_backend_initialization(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        from lerobot.export.backends.openvino import OpenVINOBackend

        model_path = package_path / "artifacts" / "model.onnx"
        backend = OpenVINOBackend(model_path, device="cpu")

        assert backend.input_names is not None
        assert backend.output_names is not None
        assert len(backend.input_names) > 0
        assert len(backend.output_names) > 0

    @pytest.mark.slow
    def test_openvino_numerical_parity_with_onnx(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        from lerobot.export.backends.onnx import ONNXBackend
        from lerobot.export.backends.openvino import OpenVINOBackend

        model_path = package_path / "artifacts" / "model.onnx"

        onnx_backend = ONNXBackend(model_path, device="cpu")
        openvino_backend = OpenVINOBackend(model_path, device="cpu")

        obs_numpy = to_numpy(batch)
        inputs = {k: v for k, v in obs_numpy.items() if k in onnx_backend.input_names}

        onnx_outputs = onnx_backend.run(inputs)
        openvino_outputs = openvino_backend.run(inputs)

        for name in onnx_outputs:
            assert_numerical_parity(
                openvino_outputs[name],
                onnx_outputs[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"OpenVINO output '{name}' does not match ONNX output",
            )


class TestACTRuntime:
    @pytest.mark.slow
    def test_create_runtime_returns_single_shot(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.runtime import SingleShotRuntime, create_runtime

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        runtime = create_runtime(package_path, backend="onnx", device="cpu")

        assert isinstance(runtime, SingleShotRuntime)
