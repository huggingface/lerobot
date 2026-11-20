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

"""
Tests for the Hub loading path of the `migrate_policy_normalization` script.
"""

import json

import httpx
import pytest
import torch
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)
from safetensors.torch import save_file

from lerobot.processor import migrate_policy_normalization as migrate

REPO_ID = "user/old-format-policy"
POLICY_CONFIG = {"type": "act", "chunk_size": 10}
TRAIN_CONFIG = {"repo_id": "user/some-dataset", "seed": 1000}


def _http_error(error_cls: type[HfHubHTTPError], status_code: int, filename: str) -> HfHubHTTPError:
    """Builds the error `huggingface_hub` raises for an HTTP `status_code` on `filename`."""
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
    response = httpx.Response(status_code, request=httpx.Request("HEAD", url))
    return error_cls(f"{status_code} Client Error.", response=response)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Replaces `hf_hub_download` with a fake Hub that holds `model.safetensors` and `config.json`.

    `fake_hub.train_config_error` is raised for `train_config.json`. When it is None, the fake Hub also
    holds a `train_config.json` file. No network access is needed.
    """

    class FakeHub:
        train_config_error: Exception | None = None

    hub = FakeHub()

    save_file(
        {"normalize_inputs.buffer_observation_state.mean": torch.zeros(2)}, tmp_path / "model.safetensors"
    )
    (tmp_path / "config.json").write_text(json.dumps(POLICY_CONFIG))
    (tmp_path / "train_config.json").write_text(json.dumps(TRAIN_CONFIG))

    def fake_hf_hub_download(repo_id, filename, revision=None, **kwargs):
        assert repo_id == REPO_ID
        if filename == "train_config.json" and hub.train_config_error is not None:
            raise hub.train_config_error
        return str(tmp_path / filename)

    monkeypatch.setattr(migrate, "hf_hub_download", fake_hf_hub_download)
    return hub


def test_load_model_from_hub_reads_train_config_when_present(fake_hub):
    state_dict, config, train_config = migrate.load_model_from_hub(REPO_ID)

    assert "normalize_inputs.buffer_observation_state.mean" in state_dict
    assert config == POLICY_CONFIG
    assert train_config == TRAIN_CONFIG


@pytest.mark.parametrize(
    "make_error",
    [
        # The Hub answers 404 because the repo has no `train_config.json`.
        pytest.param(lambda: _http_error(RemoteEntryNotFoundError, 404, "train_config.json"), id="remote"),
        # The file is not in the local cache: offline mode, or a network or server error on an uncached file.
        pytest.param(lambda: LocalEntryNotFoundError("train_config.json is not in the cache"), id="local"),
    ],
)
def test_load_model_from_hub_without_train_config_continues(fake_hub, capsys, make_error):
    """A repo without the optional `train_config.json` must not stop the migration."""
    fake_hub.train_config_error = make_error()

    state_dict, config, train_config = migrate.load_model_from_hub(REPO_ID)

    assert "normalize_inputs.buffer_observation_state.mean" in state_dict
    assert config == POLICY_CONFIG
    assert train_config is None
    assert (
        "train_config.json not found - continuing without training configuration" in capsys.readouterr().out
    )


def test_load_model_from_hub_repository_errors_still_raise(fake_hub):
    """Only a missing `train_config.json` is optional. A 401 (repository not found or no access) must raise."""
    fake_hub.train_config_error = _http_error(RepositoryNotFoundError, 401, "train_config.json")

    with pytest.raises(RepositoryNotFoundError):
        migrate.load_model_from_hub(REPO_ID)
