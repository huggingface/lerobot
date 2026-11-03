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

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.utils.hub import find_latest_hub_checkpoint, hub_safe_id


def _patch_list_files(monkeypatch, files):
    api = MagicMock()
    api.list_repo_files.return_value = files
    # HfApi is imported into lerobot.utils.hub at module load, so patch it there.
    monkeypatch.setattr("lerobot.utils.hub.HfApi", lambda *a, **k: api)
    return api


def test_find_latest_hub_checkpoint_picks_highest_step(monkeypatch):
    _patch_list_files(
        monkeypatch,
        [
            "README.md",
            "checkpoints/000500/pretrained_model/model.safetensors",
            "checkpoints/000500/training_state/training_step.json",
            "checkpoints/020000/pretrained_model/model.safetensors",
            "checkpoints/001000/training_state/training_step.json",
        ],
    )
    # Numeric max, not lexicographic — "020000" beats "001000"/"000500".
    assert find_latest_hub_checkpoint("u/run") == "checkpoints/020000"


def test_find_latest_hub_checkpoint_ignores_non_step_entries(monkeypatch):
    _patch_list_files(
        monkeypatch,
        ["checkpoints/last/pretrained_model/model.safetensors", "config.json"],
    )
    # "last" (a symlink target name) is not a numeric step → no resolvable checkpoint.
    assert find_latest_hub_checkpoint("u/run") is None


def test_find_latest_hub_checkpoint_none_when_no_checkpoints(monkeypatch):
    _patch_list_files(monkeypatch, ["config.json", "model.safetensors"])
    assert find_latest_hub_checkpoint("u/run") is None


def test_hub_safe_id_preserves_repo_ids():
    # Repo ids must keep their forward slash on every platform.
    assert hub_safe_id("lerobot/smolvla_base") == "lerobot/smolvla_base"
    assert hub_safe_id(Path("lerobot/smolvla_base")) == "lerobot/smolvla_base"


@pytest.mark.skipif(sys.platform != "win32", reason="backslash separators only occur on Windows")
def test_hub_safe_id_normalizes_windows_separators():
    # On Windows, Path("lerobot/smolvla_base") stringifies to "lerobot\\smolvla_base",
    # which huggingface_hub rejects as a repo id (#2552). hub_safe_id restores posix form.
    assert str(Path("lerobot/smolvla_base")) == "lerobot\\smolvla_base"
    assert hub_safe_id(Path("lerobot/smolvla_base")) == "lerobot/smolvla_base"
    assert hub_safe_id("lerobot\\smolvla_base") == "lerobot/smolvla_base"


def test_peft_adapter_records_posix_base_model_id():
    # lerobot-train stores --policy.path as a Path; the PEFT adapter_config.json written from it must
    # hold a Hub repo id that loads on every platform, not "lerobot\\act_base" from a Windows run.
    pytest.importorskip("peft")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        pretrained_backbone_weights=None,
        device="cpu",
    )
    config.pretrained_path = Path("lerobot/act_base")
    peft_model = ACTPolicy(config).wrap_with_peft(peft_cli_overrides={"target_modules": ["linear1"]})
    assert peft_model.peft_config["default"].base_model_name_or_path == "lerobot/act_base"
