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
transformers = pytest.importorskip("transformers")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_pi0_policy_and_batch,
    create_smolvla_policy_and_batch,
    skip_if_pi0_transformers_unavailable,
    to_numpy,
)


class TestPI0Export:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        skip_if_pi0_transformers_unavailable()
        from lerobot.export import export_policy
        from lerobot.export.manifest import Manifest, TwoPhaseConfig

        policy, batch = create_pi0_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi0_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "encoder.onnx").exists()
        assert (package_path / "artifacts" / "denoise.onnx").exists()

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_two_phase
        assert isinstance(manifest.inference, TwoPhaseConfig)
        assert manifest.inference.num_steps == policy.config.num_inference_steps

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        skip_if_pi0_transformers_unavailable()
        from lerobot.export import export_policy, load
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        policy, batch = create_pi0_policy_and_batch(device="cuda")

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                lang_tokens=batch[OBS_LANGUAGE_TOKENS],
                lang_masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                state=torch.nn.functional.pad(batch[OBS_STATE], (0, 32 - 14)),
                noise=noise,
            )

        package_path = export_policy(
            policy,
            tmp_path / "pi0_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-2,
            atol=1e-2,
            msg="PI0 ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        skip_if_pi0_transformers_unavailable()
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load
        from lerobot.utils.constants import OBS_STATE

        policy, batch = create_pi0_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi0_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.chunk_size, policy.config.max_action_dim).astype(np.float32)

        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))

        runtime_onnx = load(package_path, backend="onnx", device="cpu")
        onnx_output = runtime_onnx.predict_action_chunk(obs_numpy, noise=noise)

        runtime_openvino = load(package_path, backend="openvino", device="cpu")
        openvino_output = runtime_openvino.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            openvino_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="PI0 OpenVINO output does not match ONNX output",
        )


class TestSmolVLAExport:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.manifest import Manifest, TwoPhaseConfig

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "encoder.onnx").exists()
        assert (package_path / "artifacts" / "denoise.onnx").exists()

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_two_phase
        assert isinstance(manifest.inference, TwoPhaseConfig)
        assert manifest.inference.num_steps == policy.config.num_steps

    @pytest.mark.slow
    def test_runtime_forward_pass(self, tmp_path: Path):
        from lerobot.export import export_policy, load
        from lerobot.export.runtime import TwoPhaseRuntime

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
        )

        runtime = load(package_path, backend="onnx", device="cpu")
        assert isinstance(runtime, TwoPhaseRuntime)

        obs_numpy = to_numpy(batch)
        action_chunk = runtime.predict_action_chunk(obs_numpy)

        assert action_chunk.ndim == 2
        assert action_chunk.shape[0] == policy.config.chunk_size
        assert action_chunk.shape[1] == policy.config.max_action_dim

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import export_policy, load
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                lang_tokens=batch[OBS_LANGUAGE_TOKENS],
                lang_masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                state=torch.nn.functional.pad(batch[OBS_STATE], (0, 32 - 14)),
                noise=noise,
            )

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-2,
            atol=1e-2,
            msg="SmolVLA ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load
        from lerobot.utils.constants import OBS_STATE

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.chunk_size, policy.config.max_action_dim).astype(np.float32)

        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))

        runtime_onnx = load(package_path, backend="onnx", device="cpu")
        onnx_output = runtime_onnx.predict_action_chunk(obs_numpy, noise=noise)

        runtime_openvino = load(package_path, backend="openvino", device="cpu")
        openvino_output = runtime_openvino.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            openvino_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="SmolVLA OpenVINO output does not match ONNX output",
        )
