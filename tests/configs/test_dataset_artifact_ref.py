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
"""Validation of `dataset.repo_id` / `dataset.artifact_ref` in `TrainPipelineConfig.validate()`."""

import json

import draccus
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers --policy.type act)


def _parse(*args: str) -> TrainPipelineConfig:
    # push_to_hub=false sidesteps an unrelated, pre-existing validation (repo_id required to push to
    # the Hub) so these tests isolate the repo_id/artifact_ref behavior under test.
    return draccus.parse(
        TrainPipelineConfig, args=["--policy.type", "act", "--policy.push_to_hub", "false", *args]
    )


def test_neither_repo_id_nor_artifact_ref_fails_fast():
    # draccus can't build the nested `dataset` field at all with zero `--dataset.*` flags (a
    # pre-existing quirk, not new here) so touch an unrelated field to force construction, then let
    # our own explicit check catch the "neither set" case in validate().
    cfg = _parse("--dataset.eval_split", "0.0")
    with pytest.raises(ValueError, match="Exactly one of"):
        cfg.validate()


def test_both_repo_id_and_artifact_ref_rejected():
    cfg = _parse("--dataset.repo_id", "u/d", "--dataset.artifact_ref", "team/proj/name:latest")
    with pytest.raises(ValueError, match="Exactly one of"):
        cfg.validate()


def test_repo_id_only_is_unchanged_behavior():
    cfg = _parse("--dataset.repo_id", "u/d")
    cfg.validate()
    assert cfg.dataset.repo_id == "u/d"
    assert cfg.dataset.artifact_ref is None


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--wandb.project", "proj"],  # wandb.enable left at its false default
        ["--wandb.enable", "true", "--wandb.project", ""],
    ],
    ids=["missing_enable", "missing_project"],
)
def test_artifact_ref_requires_wandb_enable_and_project(extra_args):
    cfg = _parse("--dataset.artifact_ref", "team/proj/name:latest", *extra_args)
    with pytest.raises(ValueError, match="requires `wandb.enable=true`"):
        cfg.validate()


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_artifact_ref_requires_online_wandb_mode(mode):
    cfg = _parse(
        "--dataset.artifact_ref",
        "team/proj/name:latest",
        "--wandb.enable",
        "true",
        "--wandb.project",
        "proj",
        "--wandb.mode",
        mode,
    )
    with pytest.raises(ValueError, match="requires `wandb.mode=online`"):
        cfg.validate()


def test_artifact_ref_rejected_for_remote_jobs():
    cfg = _parse(
        "--dataset.artifact_ref",
        "team/proj/name:latest",
        "--wandb.enable",
        "true",
        "--wandb.project",
        "proj",
        "--job.target",
        "a10g-small",
    )
    with pytest.raises(ValueError, match="not supported for remote"):
        cfg.validate()


def test_artifact_ref_with_wandb_enabled_and_local_job_passes_validation():
    cfg = _parse(
        "--dataset.artifact_ref",
        "team/proj/name:latest",
        "--wandb.enable",
        "true",
        "--wandb.project",
        "proj",
    )
    cfg.validate()
    assert cfg.dataset.repo_id is None
    assert cfg.dataset.artifact_ref == "team/proj/name:latest"


def test_artifact_checkpoint_round_trips_without_special_casing(tmp_path):
    """A checkpoint written mid-run resumes and re-validates as-is: nothing materialization does to
    the config can make it fail its own exclusive-source validation.
    """
    cfg = _parse(
        "--dataset.artifact_ref",
        "team/proj/name:latest",
        "--wandb.enable",
        "true",
        "--wandb.project",
        "proj",
    )
    cfg.validate()

    # Materialization repoints `root` at the downloaded copy; `repo_id` is never written.
    cfg.dataset.root = tmp_path / "run" / "wandb_dataset"
    cfg.save_pretrained(tmp_path)

    with open(tmp_path / TRAIN_CONFIG_NAME) as f:
        saved = json.load(f)
    assert saved["dataset"]["artifact_ref"] == "team/proj/name:latest"
    assert saved["dataset"]["repo_id"] is None

    reloaded = TrainPipelineConfig.from_pretrained(tmp_path)
    reloaded.validate()
    assert reloaded.dataset.repo_id is None
    assert reloaded.dataset.artifact_ref == "team/proj/name:latest"


def test_local_id_derives_a_constructor_name_only_for_artifact_runs():
    """`local_id` is what `LeRobotDataset` gets constructed with: the Hub repo when there is one,
    the artifact's collection name when there isn't, and never a value written back to `repo_id`.
    """
    hub = DatasetConfig(repo_id="user/dataset")
    assert hub.local_id == "user/dataset"

    artifact = DatasetConfig(artifact_ref="team/proj/pick-cube:latest")
    assert artifact.local_id == "pick-cube"
    assert artifact.repo_id is None

    assert DatasetConfig().local_id is None
