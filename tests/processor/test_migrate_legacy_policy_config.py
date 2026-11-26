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

import json
import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import ModelCard
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file, save_file

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.processor import migrate_policy_normalization as migration
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


@pytest.mark.parametrize("source", ["local", "hub"])
def test_migrate_legacy_shape_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_dir = tmp_path / "migrated"
    policy = get_policy_class("tdmpc")(
        make_policy_config(
            "tdmpc",
            device="cpu",
            input_features={
                OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (4,)),
                OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            },
            output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        )
    )
    original_weights = policy.state_dict()
    legacy_weights = {name: weight.clone() for name, weight in original_weights.items()}
    legacy_weights["normalize_inputs.buffer_observation_environment_state.min"] = torch.ones(4)
    legacy_weights["normalize_inputs.buffer_observation_environment_state.max"] = torch.full((4,), 3.0)
    legacy_weights["unnormalize_outputs.buffer_action.min"] = torch.tensor([10.0, 20.0])
    legacy_weights["unnormalize_outputs.buffer_action.max"] = torch.tensor([14.0, 28.0])
    save_file(legacy_weights, checkpoint / "model.safetensors")
    raw_config = json.dumps(
        {
            "device": "cpu",
            "input_shapes": {OBS_ENV_STATE: [4], OBS_STATE: [2]},
            "output_shapes": {ACTION: [2]},
            "input_normalization_modes": {OBS_ENV_STATE: "min_max", OBS_STATE: "identity"},
            "output_normalization_modes": {ACTION: "min_max"},
        }
    )
    (checkpoint / "config.json").write_text(raw_config, encoding="utf-8")
    (checkpoint / "config.yaml").write_text("policy:\n  name: tdmpc\n", encoding="utf-8")

    downloads = []
    if source == "hub":
        pretrained_path = "example/legacy-tdmpc"

        def download(repo_id: str, filename: str, revision: str | None = None) -> str:
            assert repo_id == pretrained_path
            assert revision == "checkpoint-revision"
            downloads.append(filename)
            if filename == "train_config.json":
                raise EntryNotFoundError("No training config")
            return str(checkpoint / filename)

        monkeypatch.setattr(migration, "hf_hub_download", download)
    else:
        pretrained_path = str(checkpoint)

    # Keep migration offline. Card validation is handled separately by the Hub library.
    monkeypatch.setattr(ModelCard, "validate", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_policy_normalization.py",
            "--pretrained-path",
            pretrained_path,
            "--revision",
            "checkpoint-revision",
            "--output-dir",
            str(output_dir),
        ],
    )

    migration.main()

    assert (checkpoint / "config.json").read_text(encoding="utf-8") == raw_config
    if source == "hub":
        assert set(downloads) == {"config.json", "config.yaml", "model.safetensors", "train_config.json"}
    migrated_config = policy.config.from_pretrained(output_dir)
    assert migrated_config.input_features[OBS_ENV_STATE] == PolicyFeature(FeatureType.ENV, (4,))
    assert migrated_config.normalization_mapping["ENV"] == NormalizationMode.MIN_MAX
    assert migrated_config.normalization_mapping["STATE"] == NormalizationMode.IDENTITY
    migrated_weights = load_file(output_dir / "model.safetensors")
    assert migrated_weights.keys() == original_weights.keys()
    for name, weight in original_weights.items():
        torch.testing.assert_close(migrated_weights[name], weight, atol=0, rtol=0)
    processor_stats = {}
    for path in output_dir.glob("policy_preprocessor*.safetensors"):
        processor_stats.update(load_file(path))
    torch.testing.assert_close(processor_stats[f"{OBS_ENV_STATE}.min"], torch.ones(4))
    torch.testing.assert_close(processor_stats[f"{OBS_ENV_STATE}.max"], torch.full((4,), 3.0))

    preprocessor, postprocessor = make_pre_post_processors(migrated_config, pretrained_path=output_dir)
    observation = preprocessor(
        {OBS_ENV_STATE: torch.tensor([1.0, 2.0, 3.0, 1.5]), OBS_STATE: torch.tensor([0.25, -0.75])}
    )
    torch.testing.assert_close(observation[OBS_ENV_STATE], torch.tensor([[-1.0, 0.0, 1.0, -0.5]]))
    torch.testing.assert_close(observation[OBS_STATE], torch.tensor([[0.25, -0.75]]))
    action = postprocessor(torch.tensor([[-0.5, 0.5]]))
    torch.testing.assert_close(action, torch.tensor([[11.0, 26.0]]))
