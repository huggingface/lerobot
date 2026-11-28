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


from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.rl.algorithms.sac import SACAlgorithmConfig  # noqa: E402
from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: E402


@pytest.fixture
def rl_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainRLServerPipelineConfig:
    monkeypatch.setattr("sys.argv", ["lerobot-train"])
    return TrainRLServerPipelineConfig(
        policy=GaussianActorConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "run",
    )


@pytest.mark.parametrize("with_dataset", [False, True])
def test_rl_config_validates_with_optional_offline_dataset(
    rl_config: TrainRLServerPipelineConfig, with_dataset: bool
) -> None:
    if with_dataset:
        rl_config.dataset = DatasetConfig(repo_id="local/offline")

    rl_config.validate()

    assert isinstance(rl_config.algorithm, SACAlgorithmConfig)
    assert rl_config.algorithm.policy_config is rl_config.policy
    assert rl_config.optimizer is not None


def test_rl_config_still_rejects_multiple_offline_datasets(rl_config: TrainRLServerPipelineConfig) -> None:
    rl_config.dataset = DatasetConfig(repo_id=["local/first", "local/second"])
    with pytest.raises(NotImplementedError, match="LeRobotMultiDataset"):
        rl_config.validate()


@pytest.mark.parametrize("with_dataset", [False, True])
def test_rl_config_eval_steps_requires_held_out_dataset(
    rl_config: TrainRLServerPipelineConfig, with_dataset: bool
) -> None:
    if with_dataset:
        rl_config.dataset = DatasetConfig(repo_id="local/offline")
    rl_config.eval_steps = 1
    with pytest.raises(ValueError, match="eval_steps > 0 requires dataset.eval_split > 0.0"):
        rl_config.validate()


def test_rl_config_still_validates_offline_image_transforms(rl_config: TrainRLServerPipelineConfig) -> None:
    rl_config.dataset = DatasetConfig(repo_id="local/offline")
    rl_config.dataset.image_transforms.enable = True
    rl_config.dataset.image_transforms.backend = "gpu"
    with pytest.raises(ValueError, match="needs an accelerator device"):
        rl_config.validate()
