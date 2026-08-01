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
"""`WandBLogger.download_dataset_artifact` / `record_dataset_artifact_lineage` are thin, testable
wrappers around `lerobot.integrations.wandb_artifacts`. Mocked one layer up (at the
`lerobot.integrations.wandb_artifacts.download_artifact` boundary the logger imports), the same
boundary `tests/integrations/wandb_artifacts/test_store.py` mocks the W&B SDK at, so neither the real
W&B SDK nor a network call is ever exercised here.
"""

from unittest.mock import MagicMock

import pytest

# `validate_dataset_directory` (reached via the logger's lazy import) pulls in `datasets`/`pandas`,
# so these tests only run once the dataset extra is installed.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.integrations.wandb_artifacts as wandb_artifacts
from lerobot.common.wandb_utils import WandBLogger


def _make_logger(run: MagicMock) -> WandBLogger:
    logger = WandBLogger.__new__(WandBLogger)
    logger._run = run
    return logger


def test_download_dataset_artifact_delegates_to_store_download_artifact(monkeypatch, tmp_path):
    run = MagicMock()
    logger = _make_logger(run)
    download_root = tmp_path / "wandb_dataset"

    expected = wandb_artifacts.MaterializedArtifact(
        requested_ref="team/proj/pick-cube:latest",
        resolved_ref="team/proj/pick-cube:v3",
        local_path=download_root,
        version="v3",
        digest="deadbeef",
        metadata={},
    )
    fake_download_artifact = MagicMock(return_value=expected)
    monkeypatch.setattr(wandb_artifacts, "download_artifact", fake_download_artifact)

    result = logger.download_dataset_artifact("team/proj/pick-cube:latest", download_root)

    assert result is expected
    fake_download_artifact.assert_called_once_with(
        run,
        "team/proj/pick-cube:latest",
        expected_type="dataset",
        download_root=download_root,
        validator=wandb_artifacts.validate_dataset_directory,
    )


def test_download_dataset_artifact_propagates_failures_without_a_network_call(monkeypatch, tmp_path):
    logger = _make_logger(MagicMock())

    def _fake_download_artifact(*_args, **_kwargs):
        raise ValueError("staged directory failed validation")

    monkeypatch.setattr(wandb_artifacts, "download_artifact", _fake_download_artifact)

    try:
        logger.download_dataset_artifact("team/proj/pick-cube:latest", tmp_path / "out")
    except ValueError as e:
        assert "failed validation" in str(e)
    else:
        raise AssertionError("expected ValueError to propagate")


def test_record_dataset_artifact_lineage_writes_config_and_summary():
    run = MagicMock()
    run.summary = {}  # MagicMock's auto-mocked __setitem__/__getitem__ don't share state; use a real dict.
    logger = _make_logger(run)

    logger.record_dataset_artifact_lineage("team/proj/pick-cube:latest", "team/proj/pick-cube:v3")

    run.config.update.assert_called_once_with(
        {
            "dataset_artifact_requested_ref": "team/proj/pick-cube:latest",
            "dataset_artifact_resolved_ref": "team/proj/pick-cube:v3",
        },
        allow_val_change=True,
    )
    assert run.summary["dataset_artifact_requested_ref"] == "team/proj/pick-cube:latest"
    assert run.summary["dataset_artifact_resolved_ref"] == "team/proj/pick-cube:v3"
