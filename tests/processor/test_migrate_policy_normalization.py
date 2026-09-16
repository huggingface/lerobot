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
"""End-to-end tests for the `migrate_policy_normalization` script.

Regression coverage for #4649: the script used to feed raw JSON config values
straight into the config constructor, so tuple-typed fields (e.g. diffusion's
`crop_shape`) kept their JSON list form. `draccus.encode` then raised inside
`save_pretrained`, leaving a 0-byte `config.json` and no `model.safetensors` in
the output directory.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import migrate_policy_normalization as migrate
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


class _StubModelCard:
    """Stands in for huggingface_hub's ModelCard so the test stays offline."""

    def __init__(self):
        self.data = SimpleNamespace(datasets=None, license=None, tags=[])

    def save(self, path):
        Path(path).write_text("# stub model card\n", encoding="utf-8")


def _make_old_format_checkpoint(root: Path) -> Path:
    """Synthesize a minimal old-format (pre-processor) diffusion checkpoint."""
    model_dir = root / "old_model"
    model_dir.mkdir()
    # Tuple-typed fields ship as JSON lists in old configs — the exact values that
    # crashed draccus.encode when passed to the config constructor unconverted.
    config = {
        "type": "diffusion",
        "input_features": {
            OBS_IMAGE: {"type": "VISUAL", "shape": [3, 96, 96]},
            OBS_STATE: {"type": "STATE", "shape": [2]},
        },
        "output_features": {ACTION: {"type": "ACTION", "shape": [2]}},
        "crop_shape": [84, 84],
        "down_dims": [64, 128],
        "pretrained_backbone_weights": None,
        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Normalization buffers the script extracts into processors, plus one stray
    # parameter so the cleaned state dict is not empty.
    state_dict = {
        "normalize_inputs.buffer_observation_image.mean": torch.zeros(3, 1, 1),
        "normalize_inputs.buffer_observation_image.std": torch.ones(3, 1, 1),
        "normalize_inputs.buffer_observation_state.min": torch.zeros(2),
        "normalize_inputs.buffer_observation_state.max": torch.ones(2),
        "unnormalize_outputs.buffer_action.min": torch.zeros(2),
        "unnormalize_outputs.buffer_action.max": torch.ones(2),
        "model.diffusion_model.input_proj.weight": torch.randn(4, 4),
    }
    save_file(state_dict, str(model_dir / "model.safetensors"))
    return model_dir


def test_migration_writes_loadable_config_and_weights(tmp_path, monkeypatch):
    model_dir = _make_old_format_checkpoint(tmp_path)
    out_dir = tmp_path / "migrated"

    monkeypatch.setattr(sys, "argv", [
        "migrate_policy_normalization",
        "--pretrained-path",
        str(model_dir),
        "--output-dir",
        str(out_dir),
    ])
    # Keep the run offline: card rendering/validation hits the Hub API.
    monkeypatch.setattr(
        "lerobot.common.train_utils.generate_model_card", lambda cfg: _StubModelCard()
    )

    migrate.main()

    # The #4649 regression: config.json used to be 0 bytes and weights were never written.
    assert (out_dir / "config.json").stat().st_size > 0
    assert (out_dir / "model.safetensors").stat().st_size > 0
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        assert (out_dir / name).stat().st_size > 0

    # Tuple fields round-trip through draccus encode/parse without crashing.
    saved = json.loads((out_dir / "config.json").read_text())
    assert saved["type"] == "diffusion"
    assert saved["crop_shape"] == [84, 84]
    assert saved["down_dims"] == [64, 128]

    # The migrated directory loads as a typed config on its own.
    parsed = PreTrainedConfig.from_pretrained(out_dir)
    assert parsed.type == "diffusion"
    assert parsed.crop_shape == (84, 84)
    assert parsed.down_dims == (64, 128)
