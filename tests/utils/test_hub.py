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

from unittest.mock import MagicMock

import pytest

from lerobot.utils.hub import find_latest_hub_checkpoint, sanitize_checkpoint_consent


@pytest.mark.parametrize("remove", [False, True])
def test_checkpoint_consent_is_cleared_regardless_of_config_structure(remove):
    config = {"custom": [{"nested": {"trust_remote_code": True, "model": "user/model"}}], "other": None}

    assert sanitize_checkpoint_consent(config, remove=remove)

    expected = {"model": "user/model"} if remove else {"trust_remote_code": False, "model": "user/model"}
    assert config == {"custom": [{"nested": expected}], "other": None}


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


def test_split_hub_reference_preserves_local_paths(tmp_path, monkeypatch):
    from lerobot.utils.hub import split_hub_reference

    path = tmp_path / "tokenizer@local"
    path.mkdir()
    monkeypatch.chdir(tmp_path)
    for reference in (path, str(path), path.name):
        assert split_hub_reference(reference) == (reference, None)


@pytest.mark.parametrize("reference", ["org/tokenizer@", "@commit"])
def test_split_hub_reference_rejects_incomplete_pins(reference):
    from lerobot.utils.hub import split_hub_reference

    with pytest.raises(ValueError, match="repo@revision"):
        split_hub_reference(reference)
