#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for converged policy_package manifest schema.

These tests run without any optional dependencies (no onnxruntime, openvino, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestManifestSchema:
    """Tests for the Manifest dataclass and serialization."""

    def test_single_pass_roundtrip(self, tmp_path: Path):
        from lerobot.export.manifest import (
            CameraConfig,
            HardwareConfig,
            Manifest,
            Metadata,
            ModelConfig,
            PolicyInfo,
            PolicySource,
            ProcessorSpec,
            RobotConfig,
            TensorSpec,
        )

        manifest = Manifest(
            policy=PolicyInfo(
                name="act",
                source=PolicySource(
                    repo_id="lerobot/act_aloha",
                    class_path="lerobot.policies.act.modeling_act.ACTPolicy",
                ),
            ),
            model=ModelConfig(
                n_obs_steps=1,
                runner={"type": "single_pass", "chunk_size": 100, "n_action_steps": 100},
                artifacts={"model": "model.onnx"},
                preprocessors=[
                    ProcessorSpec(
                        type="normalize",
                        mode="mean_std",
                        artifact="stats.safetensors",
                        features=["observation.state"],
                    )
                ],
                postprocessors=[
                    ProcessorSpec(
                        type="denormalize",
                        mode="mean_std",
                        artifact="stats.safetensors",
                        features=["action"],
                    )
                ],
            ),
            hardware=HardwareConfig(
                robots=[
                    RobotConfig(
                        name="main",
                        state=TensorSpec(shape=[14], dtype="float32"),
                        action=TensorSpec(shape=[14], dtype="float32"),
                    )
                ],
                cameras=[CameraConfig(name="top", shape=[3, 480, 640], dtype="uint8")],
            ),
            metadata=Metadata(created_at="2026-04-01T00:00:00Z", created_by="test"),
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)

        assert loaded.format == "policy_package"
        assert loaded.version == "1.0"
        assert loaded.policy.name == "act"
        assert loaded.policy.source.repo_id == "lerobot/act_aloha"
        assert loaded.policy.source.class_path == "lerobot.policies.act.modeling_act.ACTPolicy"
        assert loaded.runner_type == "single_pass"
        assert loaded.model.runner["type"] == "single_pass"
        assert loaded.model.runner["chunk_size"] == 100
        assert loaded.model.artifacts["model"] == "model.onnx"
        assert "backend" not in loaded.model.to_dict()
        assert len(loaded.model.preprocessors) == 1
        assert loaded.model.preprocessors[0].type == "normalize"
        assert loaded.model.preprocessors[0].mode == "mean_std"
        assert len(loaded.model.postprocessors) == 1
        assert loaded.hardware.robots[0].name == "main"
        assert loaded.hardware.robots[0].state.shape == [14]
        assert loaded.hardware.cameras[0].name == "top"
        assert loaded.metadata.created_by == "test"

    def test_iterative_roundtrip(self, tmp_path: Path):
        from lerobot.export.manifest import (
            Manifest,
            ModelConfig,
            PolicyInfo,
        )

        manifest = Manifest(
            policy=PolicyInfo(name="diffusion"),
            model=ModelConfig(
                n_obs_steps=2,
                runner={
                    "type": "iterative",
                    "horizon": 16,
                    "n_action_steps": 8,
                    "action_dim": 6,
                    "num_inference_steps": 100,
                    "scheduler": "ddpm",
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                    "beta_schedule": "squaredcos_cap_v2",
                    "prediction_type": "epsilon",
                    "clip_sample": True,
                    "clip_sample_range": 1.0,
                },
                artifacts={"model": "model.onnx"},
            ),
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)

        assert loaded.runner_type == "iterative"
        assert loaded.model.runner["scheduler"] == "ddpm"
        assert loaded.model.runner["horizon"] == 16
        assert loaded.model.runner["action_dim"] == 6
        assert loaded.model.runner["beta_start"] == 0.0001

    def test_kv_cache_flow_roundtrip(self, tmp_path: Path):
        from lerobot.export.manifest import (
            Manifest,
            ModelConfig,
            PolicyInfo,
        )

        manifest = Manifest(
            policy=PolicyInfo(name="pi0"),
            model=ModelConfig(
                n_obs_steps=1,
                runner={
                    "type": "kv_cache_flow",
                    "chunk_size": 50,
                    "n_action_steps": 50,
                    "action_dim": 32,
                    "num_inference_steps": 10,
                    "scheduler": "euler",
                    "num_layers": 18,
                    "num_kv_heads": 8,
                    "head_dim": 256,
                    "input_mapping": {},
                },
                artifacts={
                    "encoder": "encoder.onnx",
                    "denoise": "denoise.onnx",
                },
            ),
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)

        assert loaded.model.runner["num_layers"] == 18
        assert loaded.model.runner["scheduler"] == "euler"
        assert loaded.model.artifacts["encoder"] == "encoder.onnx"
        assert loaded.model.artifacts["denoise"] == "denoise.onnx"

    def test_minimal_manifest(self, tmp_path: Path):
        """Minimal manifest with only required fields."""
        from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo

        manifest = Manifest(
            policy=PolicyInfo(name="test"),
            model=ModelConfig(
                n_obs_steps=1,
                runner={"type": "single_pass"},
                artifacts={"model": "model.onnx"},
            ),
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded = Manifest.load(manifest_path)
        assert loaded.policy.name == "test"
        assert loaded.hardware is None
        assert loaded.metadata is None
        assert loaded.model.preprocessors is None

    def test_invalid_format_raises(self):
        from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo

        with pytest.raises(ValueError, match="Invalid format"):
            Manifest(
                format="wrong_format",
                policy=PolicyInfo(name="test"),
                model=ModelConfig(
                    n_obs_steps=1,
                    runner={"type": "single_pass"},
                    artifacts={"model": "m.onnx"},
                ),
            )

    def test_missing_runner_type_raises(self):
        from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo

        with pytest.raises(ValueError, match="'type' key"):
            Manifest(
                policy=PolicyInfo(name="test"),
                model=ModelConfig(
                    n_obs_steps=1,
                    runner={"chunk_size": 10},  # missing 'type'
                    artifacts={"model": "m.onnx"},
                ),
            )

    def test_empty_artifacts_raises(self):
        from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo

        with pytest.raises(ValueError, match="At least one artifact"):
            Manifest(
                policy=PolicyInfo(name="test"),
                model=ModelConfig(
                    n_obs_steps=1,
                    runner={"type": "single_pass"},
                    artifacts={},
                ),
            )

    def test_single_pass_runner_type_property(self):
        from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo

        m = Manifest(
            policy=PolicyInfo(name="test"),
            model=ModelConfig(
                n_obs_steps=1,
                runner={"type": "single_pass", "chunk_size": 8},
                artifacts={"model": "m.onnx"},
            ),
        )

        assert m.runner_type == "single_pass"

    def test_runner_type_property(self):
        from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo

        m = Manifest(
            policy=PolicyInfo(name="test"),
            model=ModelConfig(
                n_obs_steps=1,
                runner={"type": "iterative", "scheduler": "euler"},
                artifacts={"model": "m.onnx"},
            ),
        )
        assert m.runner_type == "iterative"

    def test_converged_act_fixture_validates_unchanged(self):
        from lerobot.export.manifest import Manifest

        fixture_path = Path(__file__).parent / "fixtures" / "manifest_act_converged.json"
        fixture = json.loads(fixture_path.read_text())

        manifest = Manifest.from_dict(fixture)

        assert manifest.to_dict() == fixture


class TestNormalizer:
    """Tests for the Normalizer with ProcessorSpec-based construction."""

    def test_normalizer_mean_std(self, tmp_path: Path):
        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer, save_stats_safetensors

        stats = {
            "observation.state": {
                "mean": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                "std": np.array([1.0, 2.0, 0.5], dtype=np.float32),
            }
        }

        stats_path = tmp_path / "stats.safetensors"
        save_stats_safetensors(stats, stats_path)

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="mean_std",
                artifact="stats.safetensors",
                features=["observation.state"],
            )
        ]
        postprocessors = [
            ProcessorSpec(
                type="denormalize",
                mode="mean_std",
                artifact="stats.safetensors",
                features=["action"],
            )
        ]

        normalizer = Normalizer.from_specs(preprocessors, postprocessors, tmp_path)
        assert normalizer is not None

        observation = {"observation.state": np.array([[0.0, 3.0, 3.0]], dtype=np.float32)}
        normalized = normalizer.normalize_inputs(observation)

        expected = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(normalized["observation.state"], expected, rtol=1e-5)

    def test_normalizer_roundtrip(self, tmp_path: Path):
        """Normalize then denormalize recovers original values."""
        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer, save_stats_safetensors

        stats = {
            "observation.state": {
                "mean": np.array([1.0, 2.0], dtype=np.float32),
                "std": np.array([0.5, 1.0], dtype=np.float32),
            },
            "action": {
                "mean": np.array([3.0, 4.0], dtype=np.float32),
                "std": np.array([2.0, 0.5], dtype=np.float32),
            },
        }

        stats_path = tmp_path / "stats.safetensors"
        save_stats_safetensors(stats, stats_path)

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="mean_std",
                artifact="stats.safetensors",
                features=["observation.state"],
            )
        ]
        postprocessors = [
            ProcessorSpec(
                type="denormalize", mode="mean_std", artifact="stats.safetensors", features=["action"]
            )
        ]

        normalizer = Normalizer.from_specs(preprocessors, postprocessors, tmp_path)

        original_action = np.array([[5.0, 4.5]], dtype=np.float32)
        # Manually normalize
        normalized = (original_action - np.array([3.0, 4.0])) / np.array([2.0, 0.5])
        # Denormalize via normalizer
        recovered = normalizer.denormalize_outputs(normalized, key="action")
        np.testing.assert_allclose(recovered, original_action, rtol=1e-5)

    def test_normalizer_from_specs_returns_none_when_no_specs(self, tmp_path: Path):
        from lerobot.export.normalize import Normalizer

        result = Normalizer.from_specs(None, None, tmp_path)
        assert result is None

        result = Normalizer.from_specs([], [], tmp_path)
        assert result is None

    def test_normalizer_from_specs_raises_when_non_identity_lacks_artifact(self, tmp_path: Path):
        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="mean_std",
                artifact=None,
                features=["observation.state"],
            )
        ]

        with pytest.raises(ValueError, match="non-identity normalization specs but no stats artifact"):
            Normalizer.from_specs(preprocessors, None, tmp_path)

    def test_normalizer_from_specs_allows_identity_only_without_artifact(self, tmp_path: Path):
        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="identity",
                artifact=None,
                features=["observation.image"],
            )
        ]

        result = Normalizer.from_specs(preprocessors, None, tmp_path)
        assert result is None

    def test_normalizer_quantiles_roundtrip(self, tmp_path: Path):
        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer, save_stats_safetensors

        stats = {
            "observation.state": {
                "q01": np.array([-2.0, 0.0], dtype=np.float32),
                "q99": np.array([2.0, 4.0], dtype=np.float32),
            },
            "action": {
                "q01": np.array([-1.0, 0.0], dtype=np.float32),
                "q99": np.array([1.0, 2.0], dtype=np.float32),
            },
        }
        save_stats_safetensors(stats, tmp_path / "stats.safetensors")

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="quantiles",
                artifact="stats.safetensors",
                features=["observation.state"],
            )
        ]
        postprocessors = [
            ProcessorSpec(
                type="denormalize",
                mode="quantiles",
                artifact="stats.safetensors",
                features=["action"],
            )
        ]

        normalizer = Normalizer.from_specs(preprocessors, postprocessors, tmp_path)
        assert normalizer is not None

        observation = {"observation.state": np.array([[-2.0, 4.0], [2.0, 0.0]], dtype=np.float32)}
        normalized = normalizer.normalize_inputs(observation)
        expected = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        np.testing.assert_allclose(normalized["observation.state"], expected, rtol=1e-5)

        action_scaled = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        recovered = normalizer.denormalize_outputs(action_scaled, key="action")
        expected_action = np.array([[-1.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(recovered, expected_action, rtol=1e-5)

    def test_normalizer_quantile10_uses_q10_q90(self, tmp_path: Path):
        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer, save_stats_safetensors

        stats = {
            "observation.state": {
                "q10": np.array([0.0], dtype=np.float32),
                "q90": np.array([10.0], dtype=np.float32),
            }
        }
        save_stats_safetensors(stats, tmp_path / "stats.safetensors")

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="quantile10",
                artifact="stats.safetensors",
                features=["observation.state"],
            )
        ]
        normalizer = Normalizer.from_specs(preprocessors, None, tmp_path)
        assert normalizer is not None

        observation = {"observation.state": np.array([[0.0, 5.0, 10.0]], dtype=np.float32)}
        normalized = normalizer.normalize_inputs(observation)
        expected = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(normalized["observation.state"], expected, rtol=1e-5)

    def test_normalizer_quantiles_requires_q01_q99(self, tmp_path: Path):
        import pytest

        from lerobot.export.manifest import ProcessorSpec
        from lerobot.export.normalize import Normalizer, save_stats_safetensors

        stats = {
            "observation.state": {
                "mean": np.array([0.0], dtype=np.float32),
                "std": np.array([1.0], dtype=np.float32),
            }
        }
        save_stats_safetensors(stats, tmp_path / "stats.safetensors")

        preprocessors = [
            ProcessorSpec(
                type="normalize",
                mode="quantiles",
                artifact="stats.safetensors",
                features=["observation.state"],
            )
        ]
        normalizer = Normalizer.from_specs(preprocessors, None, tmp_path)
        assert normalizer is not None

        observation = {"observation.state": np.array([[0.5]], dtype=np.float32)}
        with pytest.raises(ValueError, match="q01.*q99|quantile"):
            normalizer.normalize_inputs(observation)

    def test_save_stats_safetensors_uses_slash_delimited_keys(self, tmp_path: Path):
        from safetensors.numpy import load_file

        from lerobot.export.normalize import save_stats_safetensors

        stats = {
            "observation.state": {
                "mean": np.array([0.0, 1.0], dtype=np.float32),
                "std": np.array([2.0, 3.0], dtype=np.float32),
            }
        }

        stats_path = tmp_path / "stats.safetensors"
        save_stats_safetensors(stats, stats_path)

        raw = load_file(str(stats_path))

        assert sorted(raw) == ["observation.state/mean", "observation.state/std"]
