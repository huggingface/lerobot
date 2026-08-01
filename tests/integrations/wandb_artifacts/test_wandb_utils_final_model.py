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
"""`WandBLogger.log_final_model` (issue #5: publish the final trained checkpoint as its own
versioned W&B Artifact). Mocked one layer up, the same way `test_wandb_utils_dataset_artifact.py`
mocks it, at the `lerobot.integrations.wandb_artifacts.upload_directory` boundary the logger imports
— so neither the real W&B SDK nor a network call is ever exercised here. `inspect_model_directory`
itself is exercised for real: it's local-only file inspection, not a network/SDK call.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

import lerobot.integrations.wandb_artifacts as wandb_artifacts
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def _make_logger(
    run: MagicMock, wandb_cfg: WandBConfig | None = None, group: str = "policy-act-seed-0"
) -> WandBLogger:
    logger = WandBLogger.__new__(WandBLogger)
    logger._run = run
    logger.cfg = wandb_cfg if wandb_cfg is not None else WandBConfig(model_artifact_name="my-policy")
    logger._group = group
    return logger


def _write_checkpoint(checkpoint_dir: Path, *, policy_type: str = "act") -> Path:
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_model_dir.mkdir(parents=True)
    (pretrained_model_dir / CONFIG_NAME).write_text(json.dumps({"type": policy_type}))
    (pretrained_model_dir / SAFETENSORS_SINGLE_FILE).write_bytes(b"weights")
    return pretrained_model_dir


def test_log_final_model_delegates_to_store_upload_directory(monkeypatch, tmp_path):
    run = MagicMock()
    wandb_cfg = WandBConfig(model_artifact_name="my-policy", model_artifact_aliases=["candidate", "v42"])
    logger = _make_logger(run, wandb_cfg)
    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    pretrained_model_dir = _write_checkpoint(checkpoint_dir)

    expected = wandb_artifacts.MaterializedArtifact(
        requested_ref="team/proj/my-policy",
        resolved_ref="team/proj/my-policy:v0",
        local_path=pretrained_model_dir,
        version="v0",
        digest="deadbeef",
        metadata={},
    )
    fake_upload_directory = MagicMock(return_value=expected)
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)

    result = logger.log_final_model(
        checkpoint_dir,
        step=1000,
        robot_type="so101",
        camera_keys=["cam_high", "cam_low"],
    )

    assert result is expected
    fake_upload_directory.assert_called_once()
    args, kwargs = fake_upload_directory.call_args
    assert args == (run, pretrained_model_dir)
    assert kwargs["name"] == "my-policy"
    assert kwargs["artifact_type"] == "model"
    assert kwargs["aliases"] == ["candidate", "v42"]
    assert kwargs["registry_collection"] is None
    metadata = kwargs["metadata"]
    assert metadata["policy_type"] == "act"
    assert metadata["final_step"] == 1000
    assert metadata["robot_type"] == "so101"
    assert metadata["camera_keys"] == ["cam_high", "cam_low"]
    assert "git_commit" in metadata
    assert "lerobot_version" in metadata
    # No dataset Artifact was trained from: no lineage fields.
    assert "dataset_artifact_requested_ref" not in metadata


def test_log_final_model_includes_dataset_lineage_when_trained_from_a_dataset_artifact(monkeypatch, tmp_path):
    run = MagicMock()
    logger = _make_logger(run)
    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    _write_checkpoint(checkpoint_dir)

    fake_upload_directory = MagicMock(
        return_value=wandb_artifacts.MaterializedArtifact(
            requested_ref="x", resolved_ref="x:v0", local_path=tmp_path, version="v0", digest="d", metadata={}
        )
    )
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)

    dataset_artifact = wandb_artifacts.MaterializedArtifact(
        requested_ref="team/proj/pick-cube:latest",
        resolved_ref="team/proj/pick-cube:v3",
        local_path=tmp_path,
        version="v3",
        digest="cafebabe",
        metadata={},
    )

    logger.log_final_model(checkpoint_dir, step=1, dataset_artifact=dataset_artifact)

    metadata = fake_upload_directory.call_args.kwargs["metadata"]
    assert metadata["dataset_artifact_requested_ref"] == "team/proj/pick-cube:latest"
    assert metadata["dataset_artifact_resolved_ref"] == "team/proj/pick-cube:v3"
    assert metadata["dataset_artifact_digest"] == "cafebabe"


def test_log_final_model_links_to_registry_when_registered_model_name_set(monkeypatch, tmp_path):
    run = MagicMock()
    wandb_cfg = WandBConfig(model_artifact_name="my-policy", registered_model_name="prod-policy")
    logger = _make_logger(run, wandb_cfg)
    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    _write_checkpoint(checkpoint_dir)

    fake_upload_directory = MagicMock(
        return_value=wandb_artifacts.MaterializedArtifact(
            requested_ref="x", resolved_ref="x:v0", local_path=tmp_path, version="v0", digest="d", metadata={}
        )
    )
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)

    logger.log_final_model(checkpoint_dir, step=1)

    assert fake_upload_directory.call_args.kwargs["registry_collection"] == "prod-policy"


def test_log_final_model_collection_name_falls_back_to_group_when_only_registered_model_name_set(
    monkeypatch, tmp_path
):
    run = MagicMock()
    wandb_cfg = WandBConfig(registered_model_name="prod-policy")
    logger = _make_logger(run, wandb_cfg, group="policy:act-seed:0")
    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    _write_checkpoint(checkpoint_dir)

    fake_upload_directory = MagicMock(
        return_value=wandb_artifacts.MaterializedArtifact(
            requested_ref="x", resolved_ref="x:v0", local_path=tmp_path, version="v0", digest="d", metadata={}
        )
    )
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)

    logger.log_final_model(checkpoint_dir, step=1)

    # ":" is not valid in a wandb artifact name (get_safe_wandb_artifact_name), so it must be scrubbed.
    assert fake_upload_directory.call_args.kwargs["name"] == "policy_act-seed_0"


def test_log_final_model_aliases_never_reach_log_policy_periodic_upload(monkeypatch, tmp_path):
    """Aliases (`wandb.model_artifact_aliases`) apply only to the final model Artifact: `log_policy`
    (the periodic per-checkpoint path) must keep calling `wandb.log_artifact` with no aliases at all.
    """
    run = MagicMock()
    wandb_cfg = WandBConfig(model_artifact_name="my-policy", model_artifact_aliases=["candidate", "v42"])
    logger = _make_logger(run, wandb_cfg)
    fake_wandb = MagicMock()
    logger._wandb = fake_wandb

    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    _write_checkpoint(checkpoint_dir)

    logger.log_policy(checkpoint_dir)

    fake_wandb.log_artifact.assert_called_once()
    args, kwargs = fake_wandb.log_artifact.call_args
    assert kwargs == {}  # no `aliases=` kwarg at all: log_policy never applies model_artifact_aliases
    assert len(args) == 1  # just the Artifact object

    # Meanwhile the final-model path (mocked at the store boundary) does apply them.
    fake_upload_directory = MagicMock(
        return_value=wandb_artifacts.MaterializedArtifact(
            requested_ref="x", resolved_ref="x:v0", local_path=tmp_path, version="v0", digest="d", metadata={}
        )
    )
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)
    logger.log_final_model(checkpoint_dir, step=1)
    assert fake_upload_directory.call_args.kwargs["aliases"] == ["candidate", "v42"]


def _write_adapter_only_checkpoint(checkpoint_dir: Path, *, base_model: str = "lerobot/pi0_base") -> Path:
    """A PEFT checkpoint: adapter weights only, base model resolved elsewhere at load time."""
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_model_dir.mkdir(parents=True)
    (pretrained_model_dir / CONFIG_NAME).write_text(json.dumps({"type": "pi0"}))
    (pretrained_model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model})
    )
    (pretrained_model_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    return pretrained_model_dir


def test_log_final_model_refuses_to_register_an_adapter_only_checkpoint(monkeypatch, tmp_path):
    """A PEFT run must not put an unrollable version into the Registry, where a team looks for
    something deployable. The Artifact is still published — only the Registry claim is withheld.
    """
    run = MagicMock()
    wandb_cfg = WandBConfig(model_artifact_name="my-policy", registered_model_name="prod-policy")
    logger = _make_logger(run, wandb_cfg)
    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    _write_adapter_only_checkpoint(checkpoint_dir)

    fake_upload_directory = MagicMock(
        return_value=wandb_artifacts.MaterializedArtifact(
            requested_ref="x", resolved_ref="x:v0", local_path=tmp_path, version="v0", digest="d", metadata={}
        )
    )
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)

    logger.log_final_model(checkpoint_dir, step=1)

    kwargs = fake_upload_directory.call_args.kwargs
    assert kwargs["registry_collection"] is None  # not linked
    assert kwargs["name"] == "my-policy"  # still uploaded, under the requested collection
    assert kwargs["metadata"]["is_self_contained"] is False
    assert "lerobot/pi0_base" in kwargs["metadata"]["registry_link_refused_reason"]


def test_log_final_model_adapter_only_without_registry_request_is_unchanged(monkeypatch, tmp_path):
    """No Registry link was asked for, so there is nothing to refuse and nothing to explain."""
    run = MagicMock()
    logger = _make_logger(run, WandBConfig(model_artifact_name="my-policy"))
    checkpoint_dir = tmp_path / "checkpoints" / "000001"
    _write_adapter_only_checkpoint(checkpoint_dir)

    fake_upload_directory = MagicMock(
        return_value=wandb_artifacts.MaterializedArtifact(
            requested_ref="x", resolved_ref="x:v0", local_path=tmp_path, version="v0", digest="d", metadata={}
        )
    )
    monkeypatch.setattr(wandb_artifacts, "upload_directory", fake_upload_directory)

    logger.log_final_model(checkpoint_dir, step=1)

    kwargs = fake_upload_directory.call_args.kwargs
    assert kwargs["registry_collection"] is None
    assert "registry_link_refused_reason" not in kwargs["metadata"]
