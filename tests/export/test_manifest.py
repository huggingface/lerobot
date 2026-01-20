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
"""Tests for manifest schema validation and serialization.

These tests run without any optional dependencies (no onnxruntime, openvino, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TestManifestSchema:
    def test_manifest_save_load_roundtrip(self, tmp_path: Path):
        from lerobot.export.manifest import (
            ActionSpec,
            ExportMetadata,
            IOSpec,
            Manifest,
            PolicyInfo,
            PolicySource,
            TensorSpec,
        )

        manifest = Manifest(
            policy=PolicyInfo(
                name="test_policy",
                source=PolicySource(repo_id="test/repo"),
            ),
            artifacts={"onnx": "artifacts/model.onnx"},
            io=IOSpec(
                inputs=[TensorSpec(name="input", dtype="float32", shape=["B", 6])],
                outputs=[TensorSpec(name="output", dtype="float32", shape=["B", 10, 6])],
            ),
            action=ActionSpec(dim=6, chunk_size=10, n_action_steps=10),
            metadata=ExportMetadata(created_at="2025-01-01T00:00:00Z", created_by="test"),
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)

        assert loaded.policy.name == "test_policy"
        assert loaded.is_single_pass
        assert loaded.artifacts["onnx"] == "artifacts/model.onnx"
        assert loaded.action.dim == 6

    def test_two_phase_manifest_save_load_roundtrip(self, tmp_path: Path):
        from lerobot.export.manifest import (
            ActionSpec,
            ExportMetadata,
            IOSpec,
            Manifest,
            PolicyInfo,
            PolicySource,
            TensorSpec,
            TwoPhaseConfig,
        )

        two_phase_config = TwoPhaseConfig(
            num_steps=10,
            encoder_artifact="onnx_encoder",
            denoise_artifact="onnx_denoise",
            num_layers=18,
            num_kv_heads=8,
            head_dim=256,
        )

        manifest = Manifest(
            policy=PolicyInfo(
                name="test_pi0_policy",
                source=PolicySource(repo_id="test/pi0"),
            ),
            artifacts={
                "onnx_encoder": "artifacts/encoder.onnx",
                "onnx_denoise": "artifacts/denoise.onnx",
            },
            io=IOSpec(
                inputs=[
                    TensorSpec(name="image_0", dtype="float32", shape=["B", 3, 224, 224]),
                    TensorSpec(name="state", dtype="float32", shape=["B", 32]),
                ],
                outputs=[TensorSpec(name="action", dtype="float32", shape=["B", 50, 32])],
            ),
            action=ActionSpec(dim=32, chunk_size=50, n_action_steps=50),
            metadata=ExportMetadata(created_at="2025-01-01T00:00:00Z", created_by="test"),
            inference=two_phase_config,
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)

        assert loaded.policy.name == "test_pi0_policy"
        assert loaded.is_two_phase
        assert isinstance(loaded.inference, TwoPhaseConfig)
        assert loaded.inference.num_steps == 10
        assert loaded.inference.encoder_artifact == "onnx_encoder"
        assert loaded.inference.denoise_artifact == "onnx_denoise"
        assert loaded.inference.num_layers == 18
        assert loaded.inference.num_kv_heads == 8
        assert loaded.inference.head_dim == 256


class TestNormalizer:
    def test_normalizer_standard(self, tmp_path: Path):
        from lerobot.export.manifest import NormalizationConfig, NormalizationType
        from lerobot.export.normalize import Normalizer, save_stats_safetensors

        stats = {
            "observation.state": {
                "mean": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "std": np.array([1.0, 2.0, 0.5], dtype=np.float32),
            }
        }

        stats_path = tmp_path / "stats.safetensors"
        save_stats_safetensors(stats, stats_path)

        config = NormalizationConfig(
            type=NormalizationType.STANDARD,
            artifact="stats.safetensors",
            input_features=["observation.state"],
            output_features=["action"],
        )
        normalizer = Normalizer.from_safetensors(stats_path, config)

        observation = {"observation.state": np.array([[0.0, 3.0, 3.0]], dtype=np.float32)}
        normalized = normalizer.normalize_inputs(observation)

        expected = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(normalized["observation.state"], expected, rtol=1e-5)
