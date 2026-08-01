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

import draccus
import pytest

from lerobot.configs.train import TrainPipelineConfig
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
