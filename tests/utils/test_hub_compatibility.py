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

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import draccus
import pytest
from huggingface_hub import hf_hub_download

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.rewards.sarm.configuration_sarm import SARMConfig
from lerobot.rl.algorithms.sac.configuration_sac import SACAlgorithmConfig
from lerobot.utils.hub import HubMixin


@pytest.mark.parametrize(
    "module_name, config_factory",
    [
        ("lerobot.configs.policies", lambda: ACTConfig(device="cpu")),
        ("lerobot.configs.rewards", lambda: SARMConfig(device="cpu")),
        ("lerobot.rl.algorithms.configs", SACAlgorithmConfig),
        (
            "lerobot.configs.train",
            lambda: draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "user/dataset"]),
        ),
    ],
)
def test_remote_config_download_uses_supported_hub_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    config_factory: Callable[[], HubMixin],
) -> None:
    config = config_factory()
    config._save_pretrained(tmp_path)
    calls = []
    signature = inspect.signature(hf_hub_download)

    def download(**kwargs: Any) -> str:
        # Check against the installed Hub's public API, including on the minimum supported version.
        signature.bind(**kwargs)
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(importlib.import_module(module_name), "hf_hub_download", download)
    config_class_name = {
        "lerobot.configs.policies": "PreTrainedConfig",
        "lerobot.configs.rewards": "RewardModelConfig",
        "lerobot.rl.algorithms.configs": "RLAlgorithmConfig",
        "lerobot.configs.train": "TrainPipelineConfig",
    }[module_name]
    config_class = getattr(importlib.import_module(module_name), config_class_name)
    loaded = config_class.from_pretrained(
        "user/model",
        revision="checkpoint",
        token="test-token",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        resume_download=True,
        proxies={"https": "http://localhost:3128"},
    )

    assert type(loaded) is type(config)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "user/model"
    assert calls[0]["revision"] == "checkpoint"
    assert calls[0]["token"] == "test-token"
    assert calls[0]["cache_dir"] == tmp_path / "cache"
    assert calls[0]["local_files_only"] is True


@pytest.mark.dataset
@pytest.mark.parametrize("upload_large_folder", [False, True])
@pytest.mark.parametrize("push_videos", [False, True])
@pytest.mark.parametrize("branch", [None, "staging"])
def test_dataset_push_uses_upload_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upload_large_folder: bool,
    push_videos: bool,
    branch: str | None,
) -> None:
    from huggingface_hub import HfApi

    from lerobot.datasets import lerobot_dataset as dataset_module
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.datasets.storage import DEFAULT_STORAGE_FORMAT

    # Avoid recording a real episode: exercise publishing with an API that exposes only supported methods.
    dataset = dataset_module.LeRobotDataset.__new__(dataset_module.LeRobotDataset)
    dataset.repo_id = "user/dataset"
    dataset.root = tmp_path
    dataset.revision = "main"
    dataset.meta = SimpleNamespace(storage_format=DEFAULT_STORAGE_FORMAT, info={})
    api = MagicMock(spec_set=HfApi)
    card = MagicMock()
    monkeypatch.setattr(dataset_module, "HfApi", lambda: api)
    monkeypatch.setattr(dataset_module, "create_lerobot_dataset_card", lambda **kwargs: card)

    dataset.push_to_hub(
        branch=branch,
        allow_patterns="**/*.parquet",
        push_videos=push_videos,
        upload_large_folder=upload_large_folder,
    )

    api.create_repo.assert_called_once_with(
        repo_id="user/dataset", private=None, repo_type="dataset", exist_ok=True
    )
    if branch:
        api.create_branch.assert_called_once_with(
            repo_id="user/dataset", branch=branch, revision="main", repo_type="dataset", exist_ok=True
        )
    else:
        api.create_branch.assert_not_called()
    api.upload_folder.assert_called_once_with(
        repo_id="user/dataset",
        folder_path=tmp_path,
        repo_type="dataset",
        revision=branch,
        allow_patterns="**/*.parquet",
        ignore_patterns=["images/"] + ([] if push_videos else ["videos/"]),
    )
    card.push_to_hub.assert_called_once_with(repo_id="user/dataset", repo_type="dataset", revision=branch)
    api.create_tag.assert_called_once_with(
        "user/dataset",
        tag=CODEBASE_VERSION,
        revision=branch,
        repo_type="dataset",
    )


def test_remote_policy_download_loads_config_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

    config = ACTConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    original = ACTPolicy(config)
    original.save_pretrained(tmp_path)
    calls = []
    signature = inspect.signature(hf_hub_download)

    def download(**kwargs: Any) -> str:
        signature.bind(**kwargs)
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr("lerobot.configs.policies.hf_hub_download", download)
    monkeypatch.setattr("lerobot.policies.pretrained.hf_hub_download", download)
    loaded = ACTPolicy.from_pretrained(
        "user/policy",
        revision="checkpoint",
        resume_download=True,
        proxies={"https": "http://localhost:3128"},
        token="test-token",
    )

    assert not loaded.training
    assert [call["filename"] for call in calls] == ["config.json", "model.safetensors"]
    assert all(call["revision"] == "checkpoint" and call["token"] == "test-token" for call in calls)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
