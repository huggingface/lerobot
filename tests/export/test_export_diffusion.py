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

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")
diffusers = pytest.importorskip("diffusers")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_diffusion_policy_and_batch,
    to_numpy,
)


class TestDiffusionExport:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        from lerobot.export import export_policy

        policy, batch = create_diffusion_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.onnx").exists()

    @pytest.mark.slow
    def test_detected_as_iterative(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.manifest import IterativeConfig, Manifest

        policy, batch = create_diffusion_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
        )

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_iterative
        assert isinstance(manifest.inference, IterativeConfig)
        assert manifest.inference.num_steps > 0

    @pytest.mark.slow
    def test_scheduler_config_exported(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.manifest import IterativeConfig, Manifest

        policy, batch = create_diffusion_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
        )

        manifest = Manifest.load(package_path / "manifest.json")
        assert isinstance(manifest.inference, IterativeConfig)
        assert manifest.inference.scheduler == "ddim"
        assert manifest.inference.num_train_timesteps == policy.config.num_train_timesteps
        assert manifest.inference.beta_schedule == policy.config.beta_schedule
        assert manifest.inference.prediction_type == policy.config.prediction_type

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import export_policy, load
        from lerobot.utils.constants import OBS_IMAGES

        policy, batch = create_diffusion_policy_and_batch()

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.horizon, policy.config.action_feature.shape[0])

        stacked_batch = {
            "observation.state": batch["observation.state"],
            OBS_IMAGES: batch["observation.images.top"].unsqueeze(2),
        }

        with torch.no_grad():
            global_cond = policy.diffusion._prepare_global_conditioning(stacked_batch)
            pytorch_output = policy.diffusion.conditional_sample(1, global_cond=global_cond, noise=noise)

        package_path = export_policy(
            policy,
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load(package_path, backend="onnx", device="cpu")

        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-2,
            atol=1e-3,
            msg="Diffusion ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load

        policy, batch = create_diffusion_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.horizon, policy.config.action_feature.shape[0]).astype(
            np.float32
        )

        obs_numpy = to_numpy(batch)

        runtime_onnx = load(package_path, backend="onnx", device="cpu")
        onnx_output = runtime_onnx.predict_action_chunk(obs_numpy, noise=noise)

        runtime_openvino = load(package_path, backend="openvino", device="cpu")
        openvino_output = runtime_openvino.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            openvino_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Diffusion OpenVINO output does not match ONNX output",
        )


class TestDiffusionRuntime:
    @pytest.mark.slow
    def test_create_runtime_returns_iterative(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.runtime import IterativeRuntime, create_runtime

        policy, batch = create_diffusion_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
        )

        runtime = create_runtime(package_path, backend="onnx", device="cpu")

        assert isinstance(runtime, IterativeRuntime)
