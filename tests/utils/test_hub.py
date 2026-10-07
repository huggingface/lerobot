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

from lerobot.utils.hub import find_latest_hub_checkpoint


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
