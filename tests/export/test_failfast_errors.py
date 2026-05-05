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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_normalizer_raises_when_stats_missing_for_key():
    from lerobot.export.normalize import Normalizer

    normalizer = Normalizer({"observation.state": "mean_std"}, {"action": "mean_std"}, stats={})

    with pytest.raises(ValueError, match=r"missing normalization stats for key 'observation\.state'"):
        normalizer.normalize_inputs({"observation.state": np.zeros((2,), dtype=np.float32)})

    with pytest.raises(ValueError, match=r"missing normalization stats for key 'action'"):
        normalizer.denormalize_outputs(np.zeros((2,), dtype=np.float32), key="action")


def test_export_policy_raises_when_normalization_stats_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.export import exporter

    class DummyPolicy:
        def __init__(self) -> None:
            state_feature = PolicyFeature(type=FeatureType.STATE, shape=(6,))
            config = SimpleNamespace(
                repo_id=None,
                revision=None,
                n_obs_steps=1,
                input_features={"observation.state": state_feature},
                output_features={},
                robot_state_feature=state_feature,
                env_state_feature=None,
                image_features={},
                normalization_mapping={"STATE": NormalizationMode.MEAN_STD},
            )
            self.config = config

        def export_assets(self, output_dir: Path) -> dict[str, str]:
            del output_dir
            return {}

        def export_stats(self, output_dir: Path, *, include_normalization: bool) -> str | None:
            del output_dir
            if include_normalization:
                raise ValueError(
                    "cannot export policy DummyPolicy: normalization stats required but not available"
                )
            return None

        def export_processor_specs(self, *, include_normalization: bool, stats_artifact, assets=None):
            del include_normalization, stats_artifact, assets
            return [], []

    class DummyRunner:
        type = "single_pass"

        @staticmethod
        def export(policy, example_batch):
            return {}, {"type": "single_pass"}

    class DummyBackend:
        runtime_only = False

        @staticmethod
        def serialize(modules, artifacts_dir, opset_version=17):
            return {"model": "model.onnx"}

    monkeypatch.setattr(exporter, "_select_runner", lambda policy: DummyRunner)
    monkeypatch.setattr(exporter, "BACKENDS", {"onnx": DummyBackend()})
    with pytest.raises(
        ValueError,
        match=r"cannot export policy DummyPolicy: normalization stats required but not available",
    ):
        exporter.export_policy(DummyPolicy(), tmp_path / "package", example_batch={})


def test_manifest_load_wraps_parse_errors_with_path(tmp_path: Path):
    from lerobot.export.manifest import Manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json")

    with pytest.raises(ValueError, match=rf"failed to parse manifest at {manifest_path}:"):
        Manifest.load(manifest_path)
