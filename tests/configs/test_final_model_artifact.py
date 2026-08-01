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
"""Validation of `wandb.model_artifact_name` / `wandb.registered_model_name` in
`TrainPipelineConfig.validate()` (issue #5: publishing the final trained model as its own
versioned W&B Artifact)."""

import draccus
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers --policy.type act)


def _parse(*args: str) -> TrainPipelineConfig:
    # draccus can't build the nested `dataset` field at all with zero `--dataset.*` flags (see
    # tests/configs/test_dataset_artifact_ref.py), so every call here supplies a plain repo_id;
    # none of these tests are about `dataset.repo_id`/`artifact_ref` itself.
    return draccus.parse(
        TrainPipelineConfig,
        args=["--policy.type", "act", "--policy.push_to_hub", "false", "--dataset.repo_id", "u/d", *args],
    )


def test_both_names_unset_is_unchanged_behavior():
    cfg = _parse()
    cfg.validate()
    assert cfg.wandb.model_artifact_name is None
    assert cfg.wandb.registered_model_name is None
    assert cfg.wandb.model_artifact_aliases == ["latest"]


def test_model_artifact_name_requires_save_checkpoint():
    cfg = _parse(
        "--wandb.enable", "true", "--wandb.model_artifact_name", "my-policy", "--save_checkpoint", "false"
    )
    with pytest.raises(ValueError, match="save_checkpoint=True"):
        cfg.validate()


def test_registered_model_name_requires_save_checkpoint():
    cfg = _parse(
        "--wandb.enable",
        "true",
        "--wandb.registered_model_name",
        "my-policy",
        "--save_checkpoint",
        "false",
    )
    with pytest.raises(ValueError, match="save_checkpoint=True"):
        cfg.validate()


def test_model_artifact_name_with_save_checkpoint_passes_validation():
    cfg = _parse(
        "--wandb.enable", "true", "--wandb.model_artifact_name", "my-policy", "--save_checkpoint", "true"
    )
    cfg.validate()


def test_registered_model_name_rejected_when_offline():
    cfg = _parse(
        "--wandb.enable",
        "true",
        "--wandb.registered_model_name",
        "my-policy",
        "--save_checkpoint",
        "true",
        "--wandb.mode",
        "offline",
    )
    with pytest.raises(ValueError, match="online"):
        cfg.validate()


def test_model_artifact_name_rejected_when_offline():
    # P1: `upload_directory` calls `logged.wait()` and reads server-assigned version/digest/
    # qualified-name, none of which offline mode can produce, so this must fail in `validate()`
    # rather than after a full training run.
    cfg = _parse(
        "--wandb.enable",
        "true",
        "--wandb.model_artifact_name",
        "my-policy",
        "--save_checkpoint",
        "true",
        "--wandb.mode",
        "offline",
    )
    with pytest.raises(ValueError, match="online"):
        cfg.validate()


def test_registered_model_name_with_online_mode_passes_validation():
    cfg = _parse(
        "--wandb.enable",
        "true",
        "--wandb.registered_model_name",
        "my-policy",
        "--save_checkpoint",
        "true",
        "--wandb.mode",
        "online",
    )
    cfg.validate()


def test_model_artifact_name_requires_wandb_enable():
    # P1: `train()` never constructs a `WandBLogger` when `wandb.enable=false`, so the requested
    # artifact would otherwise be silently skipped after a full training run.
    cfg = _parse("--wandb.model_artifact_name", "my-policy", "--save_checkpoint", "true")
    with pytest.raises(ValueError, match="wandb.enable"):
        cfg.validate()


def test_registered_model_name_requires_wandb_enable():
    cfg = _parse("--wandb.registered_model_name", "my-policy", "--save_checkpoint", "true")
    with pytest.raises(ValueError, match="wandb.enable"):
        cfg.validate()
