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

import draccus
import pytest

from lerobot.configs import JobConfig
from lerobot.configs.train import TrainPipelineConfig


def test_jobconfig_defaults_are_local():
    cfg = JobConfig()
    assert cfg.target is None
    assert cfg.is_remote is False
    assert cfg.image == "huggingface/lerobot-gpu:latest"
    assert cfg.timeout == "2d"
    assert cfg.detach is False


def test_jobconfig_local_string_is_not_remote():
    assert JobConfig(target="local").is_remote is False


def test_jobconfig_flavor_is_remote():
    assert JobConfig(target="a10g-small").is_remote is True


def test_train_config_parses_job_target():
    parsed = draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "u/d", "--policy.type", "act", "--job.target", "a10g-small"],
    )
    assert parsed.job.target == "a10g-small"
    assert parsed.job.is_remote is True
    assert parsed.save_checkpoint_to_hub is False


def test_save_checkpoint_to_hub_requires_repo_id():
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id",
            "u/d",
            "--policy.type",
            "act",
            "--policy.push_to_hub",
            "false",
            "--save_checkpoint_to_hub",
            "true",
        ],
    )
    with pytest.raises(ValueError, match="requires --policy.repo_id"):
        cfg.validate()
