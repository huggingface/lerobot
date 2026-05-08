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

"""Tests for the reward model base classes and registry."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel


@RewardModelConfig.register_subclass(name="_dummy_hub_reward")
@dataclass
class _DummyHubRewardConfig(RewardModelConfig):
    def get_optimizer_preset(self):
        return AdamWConfig(lr=1e-4)


class _DummyHubReward(PreTrainedRewardModel):
    config_class = _DummyHubRewardConfig
    name = "_dummy_hub_reward"

    def __init__(self, config):
        super().__init__(config)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def compute_reward(self, batch):
        return self.bias.expand(1)


def test_reward_model_config_registry():
    """Verify that classifier and sarm are registered."""
    known = RewardModelConfig.get_known_choices()
    assert "reward_classifier" in known
    assert "sarm" in known


def test_reward_model_config_lookup():
    """Verify that we can look up configs by name."""
    cls = RewardModelConfig.get_choice_class("reward_classifier")
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    assert cls is RewardClassifierConfig


def test_factory_get_reward_model_class():
    """Test the get_reward_model_class factory."""
    from lerobot.rewards.factory import get_reward_model_class

    cls = get_reward_model_class("sarm")
    from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel

    assert cls is SARMRewardModel


def test_factory_unknown_raises():
    """Unknown name should raise ValueError."""
    from lerobot.rewards.factory import get_reward_model_class

    with pytest.raises(ValueError, match="not available"):
        get_reward_model_class("nonexistent_reward_model")


def test_pretrained_reward_model_requires_config_class():
    """Subclass without config_class should fail."""
    with pytest.raises(TypeError, match="must define 'config_class'"):

        class BadModel(PreTrainedRewardModel):
            name = "bad"

            def compute_reward(self, batch):
                pass


def test_pretrained_reward_model_requires_name():
    """Subclass without name should fail."""
    with pytest.raises(TypeError, match="must define 'name'"):

        class BadModel(PreTrainedRewardModel):
            config_class = RewardModelConfig

            def compute_reward(self, batch):
                pass


def test_non_trainable_forward_raises():
    """Non-trainable model should raise on forward()."""
    from dataclasses import dataclass

    from lerobot.optim.optimizers import AdamWConfig

    @dataclass
    class DummyConfig(RewardModelConfig):
        def get_optimizer_preset(self):
            return AdamWConfig(lr=1e-4)

    class DummyReward(PreTrainedRewardModel):
        config_class = DummyConfig
        name = "dummy_test"

        def compute_reward(self, batch):
            return torch.zeros(1)

    config = DummyConfig()
    model = DummyReward(config)

    with pytest.raises(NotImplementedError, match="not trainable"):
        model.forward({"x": torch.zeros(1)})


# ---------------------------------------------------------------------------
# Trainable vs zero-shot (general-purpose) reward models.
# The proposal explicitly supports models like TOPReward that wrap a pretrained
# VLM and produce a reward signal without any training step. These tests pin
# the contract that lets such models coexist with trainable ones.
# ---------------------------------------------------------------------------


def test_is_trainable_false_when_forward_not_overridden():
    """A reward model that only implements ``compute_reward`` is zero-shot."""
    model, _ = _make_dummy_reward_model()
    assert model.is_trainable is False


def test_is_trainable_true_when_forward_overridden():
    """Overriding ``forward`` flips ``is_trainable`` to True."""

    class _TrainableReward(_DummyHubReward):
        name = "_trainable_dummy_reward"

        def forward(self, batch):
            loss = (self.bias**2).sum()
            return loss, {}

    # Register a fresh config subclass so the subclass check passes.
    @RewardModelConfig.register_subclass(name="_trainable_dummy_reward")
    @dataclass
    class _TrainableConfig(_DummyHubRewardConfig):
        pass

    _TrainableReward.config_class = _TrainableConfig
    model = _TrainableReward(_TrainableConfig())
    assert model.is_trainable is True


# ---------------------------------------------------------------------------
# RewardModelConfig.from_pretrained
# ---------------------------------------------------------------------------


def test_reward_model_config_from_pretrained_raises_when_config_missing(tmp_path):
    """``from_pretrained`` must surface a clear ``FileNotFoundError`` when the
    target directory exists but does not contain ``config.json``, instead of
    crashing later inside ``draccus.parse``.
    """
    # tmp_path exists but has no config.json
    with pytest.raises(FileNotFoundError, match="config.json not found"):
        RewardModelConfig.from_pretrained(tmp_path)


def test_reward_model_config_from_pretrained_roundtrip(tmp_path):
    """Round-trip: save a RewardClassifierConfig, reload it, fields must match."""
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    original = RewardClassifierConfig(
        num_classes=3,
        hidden_dim=128,
        latent_dim=64,
        num_cameras=1,
        learning_rate=5e-4,
    )
    original._save_pretrained(tmp_path)

    loaded = RewardModelConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, RewardClassifierConfig)
    assert loaded.num_classes == 3
    assert loaded.hidden_dim == 128
    assert loaded.latent_dim == 64
    assert loaded.num_cameras == 1
    assert loaded.learning_rate == 5e-4


# ---------------------------------------------------------------------------
# TrainPipelineConfig — reward model training path
# ---------------------------------------------------------------------------


def test_train_pipeline_config_path_fields_includes_reward_model():
    """``--reward_model.path=local/dir`` requires ``reward_model`` to be listed
    as a draccus path-field on ``TrainPipelineConfig``."""
    from lerobot.configs.train import TrainPipelineConfig

    fields = TrainPipelineConfig.__get_path_fields__()
    assert "policy" in fields
    assert "reward_model" in fields


def test_train_pipeline_config_trainable_config_returns_reward_model_when_set():
    """When only ``reward_model`` is set, ``trainable_config`` (used by the
    trainer for e.g. ``.device``) must return it — not ``None`` from ``policy``."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    reward_cfg = RewardClassifierConfig(device="cpu")
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        reward_model=reward_cfg,
    )

    assert cfg.is_reward_model_training is True
    assert cfg.trainable_config is reward_cfg
    # This is what lerobot_train.py uses to decide force_cpu; ``cfg.policy.device``
    # would AttributeError here because policy is None.
    assert cfg.trainable_config.device == "cpu"


def test_train_pipeline_config_trainable_config_returns_policy_when_set():
    """Mirror of the reward-model case: when only ``policy`` is set,
    ``trainable_config`` must return it."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    policy_cfg = DiffusionConfig(device="cpu")
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy_cfg,
    )

    assert cfg.is_reward_model_training is False
    assert cfg.trainable_config is policy_cfg
    assert cfg.trainable_config.device == "cpu"


def test_train_pipeline_config_from_pretrained_migrates_legacy_rabc_fields(tmp_path):
    """Legacy top-level RA-BC fields should be migrated into ``sample_weighting``."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=DiffusionConfig(device="cpu"),
    )
    cfg._save_pretrained(tmp_path)

    config_path = tmp_path / TRAIN_CONFIG_NAME
    with open(config_path) as f:
        payload = json.load(f)

    payload.pop("sample_weighting", None)
    payload.update(
        {
            "use_rabc": True,
            "rabc_progress_path": "hf://datasets/user/repo/sarm_progress.parquet",
            "rabc_kappa": 0.05,
            "rabc_epsilon": 1e-5,
            "rabc_head_mode": "dense",
        }
    )
    with open(config_path, "w") as f:
        json.dump(payload, f)

    loaded = TrainPipelineConfig.from_pretrained(tmp_path)

    assert loaded.sample_weighting is not None
    assert loaded.sample_weighting.type == "rabc"
    assert loaded.sample_weighting.progress_path == "hf://datasets/user/repo/sarm_progress.parquet"
    assert loaded.sample_weighting.kappa == 0.05
    assert loaded.sample_weighting.epsilon == 1e-5
    assert loaded.sample_weighting.head_mode == "dense"


def test_train_pipeline_config_from_pretrained_strips_legacy_rabc_when_disabled(tmp_path):
    """Legacy RA-BC fields should be ignored when ``use_rabc`` was false."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=DiffusionConfig(device="cpu"),
    )
    cfg._save_pretrained(tmp_path)

    config_path = tmp_path / TRAIN_CONFIG_NAME
    with open(config_path) as f:
        payload = json.load(f)

    payload.pop("sample_weighting", None)
    payload.update(
        {
            "use_rabc": False,
            "rabc_progress_path": "hf://datasets/user/repo/sarm_progress.parquet",
            "rabc_kappa": 0.05,
            "rabc_epsilon": 1e-5,
            "rabc_head_mode": "dense",
        }
    )
    with open(config_path, "w") as f:
        json.dump(payload, f)

    loaded = TrainPipelineConfig.from_pretrained(tmp_path)

    assert loaded.sample_weighting is None


# ---------------------------------------------------------------------------
# PreTrainedRewardModel hub upload: push_model_to_hub + generate_model_card.
# We test the generation side (offline) fully, and the upload side with HfApi
# mocked so nothing actually hits the network.
# ---------------------------------------------------------------------------


def _make_dummy_reward_model(**config_kwargs):
    return _DummyHubReward(_DummyHubRewardConfig(**config_kwargs)), _DummyHubRewardConfig


@pytest.fixture
def _offline_model_card(monkeypatch):
    """``ModelCard.validate`` does a live ``POST`` to huggingface.co — bypass it
    so tests can run offline."""
    from huggingface_hub import ModelCard

    monkeypatch.setattr(ModelCard, "validate", lambda self, *a, **kw: None)


def test_reward_model_generate_model_card_renders_expected_fields(_offline_model_card):
    """``generate_model_card`` must produce a card with the right metadata and
    body, using the dedicated reward-model template."""
    model, _ = _make_dummy_reward_model(
        license="mit",
        tags=["robot", "sim"],
    )

    card = model.generate_model_card(
        dataset_repo_id="user/my_dataset",
        model_type=model.config.type,
        license=model.config.license,
        tags=model.config.tags,
    )

    # Metadata (YAML header) — ModelCardData fields.
    assert card.data.license == "mit"
    assert card.data.library_name == "lerobot"
    assert card.data.pipeline_tag == "robotics"
    assert "reward-model" in card.data.tags
    assert model.config.type in card.data.tags
    assert card.data.model_name == model.config.type
    assert card.data.datasets == "user/my_dataset"

    # Body — specific to the reward-model template, NOT the policy one.
    body = str(card)
    assert "Reward Model Card" in body
    assert "This reward model has been trained" in body
    assert "--reward_model.type=" in body  # reward-model-specific usage block


def test_reward_model_generate_model_card_uses_default_license(_offline_model_card):
    """When config.license is None the card falls back to apache-2.0."""
    model, _ = _make_dummy_reward_model()

    card = model.generate_model_card(
        dataset_repo_id="user/my_dataset",
        model_type=model.config.type,
        license=model.config.license,
        tags=None,
    )

    assert card.data.license == "apache-2.0"


def test_reward_model_push_model_to_hub_uploads_expected_files(monkeypatch, _offline_model_card):
    """``push_model_to_hub`` must:
    1. create the repo,
    2. assemble a temp folder with weights + config.json + train_config.json + README.md,
    3. call ``api.upload_folder`` on that folder.
    All network calls are mocked.
    """
    from huggingface_hub.constants import CONFIG_NAME

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig

    model, _ = _make_dummy_reward_model(
        repo_id="user/my_reward",
        license="apache-2.0",
    )
    # Point the reward model's train config at a dummy dataset repo.
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/my_dataset"),
        reward_model=model.config,
    )

    uploaded: dict = {}
    fake_commit_info = SimpleNamespace(repo_url=SimpleNamespace(url="https://huggingface.co/user/my_reward"))

    class _FakeHfApi:
        def create_repo(self, repo_id, private=None, exist_ok=False):
            uploaded["create_repo_id"] = repo_id
            uploaded["create_private"] = private
            return SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message, **_kwargs):
            uploaded["upload_repo_id"] = repo_id
            uploaded["upload_repo_type"] = repo_type
            uploaded["commit_message"] = commit_message
            # Snapshot files assembled in the temp folder — this is the real
            # contract we care about.
            uploaded["files"] = sorted(p.name for p in Path(folder_path).iterdir())
            return fake_commit_info

    from lerobot.rewards import pretrained as reward_pretrained

    monkeypatch.setattr(reward_pretrained, "HfApi", lambda *a, **kw: _FakeHfApi())

    model.push_model_to_hub(train_cfg)

    assert uploaded["create_repo_id"] == "user/my_reward"
    assert uploaded["upload_repo_id"] == "user/my_reward"
    assert uploaded["upload_repo_type"] == "model"
    assert uploaded["commit_message"] == "Upload reward model weights, train config and readme"
    # Minimum required files that must be uploaded with a reward model.
    assert CONFIG_NAME in uploaded["files"]  # config.json
    assert TRAIN_CONFIG_NAME in uploaded["files"]  # train_config.json
    assert "README.md" in uploaded["files"]
    assert any(name.endswith(".safetensors") for name in uploaded["files"])
