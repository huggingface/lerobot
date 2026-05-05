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

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_act_policy_and_batch,
    to_numpy,
)


def _read_manifest(package_path: Path) -> dict[str, Any]:
    with (package_path / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


class TestACTExport:
    @pytest.mark.slow
    def test_export_creates_valid_package_manifest_runner_type_class_path(self, tmp_path: Path):
        policy, batch = create_act_policy_and_batch()

        package_path = policy.export(
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.onnx").exists()

        manifest = _read_manifest(package_path)
        assert manifest["model"]["runner"]["type"] == "single_pass"
        assert manifest["policy"]["source"]["class_path"] == "lerobot.policies.act.modeling_act.ACTPolicy"
        assert "backend" not in manifest["model"]

    @pytest.mark.slow
    def test_onnx_forward_pass(self, tmp_path: Path):
        from lerobot.export import ExportedPolicy, load_exported_policy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        assert isinstance(runtime, ExportedPolicy)

        obs_numpy = to_numpy(batch)
        action_chunk = runtime.predict_action_chunk(obs_numpy)

        assert action_chunk.ndim == 2
        assert action_chunk.shape[0] == policy.config.chunk_size
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action_chunk.shape[1] == action_dim

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        with torch.no_grad():
            torch.manual_seed(42)
            pytorch_output = policy.predict_action_chunk(batch)

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy)

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="ACT ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_onnx_numerical_parity_with_normalization(self, tmp_path: Path):
        """End-to-end parity with normalization ON: runtime Normalizer must match manual normalize/denormalize."""
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()
        state_mean = np.full(6, 0.1, dtype=np.float32)
        state_std = np.full(6, 0.5, dtype=np.float32)
        image_mean = np.full((3, 1, 1), 0.25, dtype=np.float32)
        image_std = np.full((3, 1, 1), 0.5, dtype=np.float32)
        action_mean = np.full(6, 0.2, dtype=np.float32)
        action_std = np.full(6, 0.4, dtype=np.float32)

        policy.config.stats = {
            "observation.state": {"mean": state_mean.tolist(), "std": state_std.tolist()},
            "observation.images.top": {"mean": image_mean.tolist(), "std": image_std.tolist()},
            "action": {"mean": action_mean.tolist(), "std": action_std.tolist()},
        }

        normalized_batch = {
            "observation.state": (batch["observation.state"] - torch.from_numpy(state_mean))
            / torch.from_numpy(state_std),
            "observation.images.top": (batch["observation.images.top"] - torch.from_numpy(image_mean))
            / torch.from_numpy(image_std),
        }
        with torch.no_grad():
            torch.manual_seed(42)
            pytorch_normalized_output = policy.predict_action_chunk(normalized_batch)
        pytorch_denormalized = pytorch_normalized_output.cpu().numpy() * action_std + action_mean

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
            include_normalization=True,
        )

        manifest = _read_manifest(package_path)
        preprocessors = manifest["model"]["preprocessors"] or []
        postprocessors = manifest["model"]["postprocessors"] or []
        assert manifest["model"]["runner"]["type"] == "single_pass"
        assert manifest["policy"]["source"]["class_path"] == "lerobot.policies.act.modeling_act.ACTPolicy"
        assert "backend" not in manifest["model"]
        assert [processor["type"] for processor in preprocessors] == ["normalize"]
        assert [processor["type"] for processor in postprocessors] == ["denormalize"]
        assert preprocessors, "Expected manifest to declare normalize preprocessors when stats are present"
        assert postprocessors, (
            "Expected manifest to declare denormalize postprocessors when stats are present"
        )
        assert (package_path / "stats.safetensors").exists()

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy)

        pytorch_np = pytorch_denormalized
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="ACT ONNX output (with runtime normalization) does not match PyTorch normalize→model→denormalize",
        )

    @pytest.mark.slow
    def test_select_action_is_default_api(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
        )

        exported_policy = load_exported_policy(package_path, backend="onnx", device="cpu")

        obs_numpy = to_numpy(batch)
        exported_policy.reset()
        action = exported_policy.select_action(obs_numpy)

        assert action.ndim == 1
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action.shape[0] == action_dim


class TestACTBackends:
    def test_onnx_backend_initialization(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.backends import BACKENDS
        from lerobot.export.backends.onnx import _ONNXRuntimeSession

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        manifest = _read_manifest(package_path)
        session = BACKENDS["onnx"].open(package_path / "artifacts", manifest, device="cpu")
        assert isinstance(session, _ONNXRuntimeSession)

        outputs = session.run("model", to_numpy(batch))
        assert outputs
        assert all(isinstance(value, np.ndarray) for value in outputs.values())

    def test_openvino_backend_initialization(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy
        from lerobot.export.backends import BACKENDS
        from lerobot.export.backends.openvino import _OpenVINORuntimeSession

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        manifest = _read_manifest(package_path)
        session = BACKENDS["openvino"].open(package_path / "artifacts", manifest, device="cpu")
        assert isinstance(session, _OpenVINORuntimeSession)

        outputs = session.run("model", to_numpy(batch))
        assert outputs
        assert all(isinstance(value, np.ndarray) for value in outputs.values())

    @pytest.mark.slow
    def test_openvino_numerical_parity_with_onnx(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy
        from lerobot.export.backends import BACKENDS

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        manifest = _read_manifest(package_path)
        onnx_session = BACKENDS["onnx"].open(package_path / "artifacts", manifest, device="cpu")
        ov_session = BACKENDS["openvino"].open(package_path / "artifacts", manifest, device="cpu")

        obs_numpy = to_numpy(batch)
        onnx_outputs = onnx_session.run("model", obs_numpy)
        openvino_outputs = ov_session.run("model", obs_numpy)

        for name in onnx_outputs:
            assert_numerical_parity(
                openvino_outputs[name],
                onnx_outputs[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"OpenVINO output '{name}' does not match ONNX output",
            )

    @pytest.mark.slow
    def test_openvino_numerical_parity_with_pytorch(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        with torch.no_grad():
            torch.manual_seed(42)
            pytorch_output = policy.predict_action_chunk(batch)

        package_path = policy.to_openvino(
            tmp_path / "act_package",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="openvino", device="cpu")
        obs_numpy = to_numpy(batch)
        ov_output = runtime.predict_action_chunk(obs_numpy)

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            ov_output,
            pytorch_np,
            rtol=1e-4,
            atol=1e-4,
            msg="ACT OpenVINO output does not match PyTorch output",
        )


class TestACTRuntime:
    @pytest.mark.slow
    def test_from_exported_loads_user_facing_policy(self, tmp_path: Path):
        from lerobot.export import ExportedPolicy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
        )

        runtime = policy.from_exported(package_path, backend="onnx", device="cpu")

        assert isinstance(runtime, ExportedPolicy)
