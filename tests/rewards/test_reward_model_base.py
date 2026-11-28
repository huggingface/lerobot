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
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lerobot.common.train_utils import generate_model_card
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig


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


@pytest.mark.parametrize(
    ("name", "module_path", "class_name"),
    [
        ("reward_classifier", "lerobot.rewards.classifier.modeling_classifier", "Classifier"),
        ("sarm", "lerobot.rewards.sarm.modeling_sarm", "SARMRewardModel"),
        ("topreward", "lerobot.rewards.topreward.modeling_topreward", "TOPRewardModel"),
    ],
)
def test_factory_get_reward_model_class(name, module_path, class_name):
    """Built-in reward models resolve by config_class, whatever their class name."""
    import importlib

    from lerobot.rewards.factory import get_reward_model_class

    assert get_reward_model_class(name) is getattr(importlib.import_module(module_path), class_name)


def test_factory_get_reward_model_class_matches_plugin_config_class(monkeypatch):
    import lerobot.rewards.factory as reward_factory

    class PluginConfig(RewardModelConfig):
        pass

    class OtherConfig(RewardModelConfig):
        pass

    PluginConfig.__module__ = "example.configuration_plugin"

    class OtherModel(PreTrainedRewardModel):
        config_class = OtherConfig
        name = "other"

    class PluginModel(PreTrainedRewardModel):
        config_class = PluginConfig
        name = "plugin"

    def make_module(name, **classes):
        module = ModuleType(name)
        vars(module).update(classes)
        return module

    captured = {}

    def import_module(name):
        captured.update(module=name)
        return make_module(
            name, PreTrainedRewardModel=PreTrainedRewardModel, OtherModel=OtherModel, PluginModel=PluginModel
        )

    monkeypatch.setattr(RewardModelConfig, "get_known_choices", lambda: {"plugin": PluginConfig})
    monkeypatch.setattr(RewardModelConfig, "get_choice_class", lambda name: PluginConfig)
    monkeypatch.setattr(reward_factory.importlib, "import_module", import_module)

    assert reward_factory.get_reward_model_class("plugin") is PluginModel
    assert captured == {"module": "example.modeling_plugin"}

    monkeypatch.setattr(
        reward_factory.importlib, "import_module", lambda name: make_module(name, OtherModel=OtherModel)
    )
    with pytest.raises(
        ValueError, match="config_class = PluginConfig.*not found in 'example.modeling_plugin'"
    ):
        reward_factory.get_reward_model_class("plugin")


def test_factory_get_reward_model_class_reports_missing_modeling_module(monkeypatch):
    import lerobot.rewards.factory as reward_factory

    class PluginConfig(RewardModelConfig):
        pass

    PluginConfig.__module__ = "example.configuration_plugin"

    def raise_missing_module(module_path):
        raise ModuleNotFoundError(f"No module named {module_path!r}", name=module_path)

    monkeypatch.setattr(RewardModelConfig, "get_known_choices", lambda: {"plugin": PluginConfig})
    monkeypatch.setattr(RewardModelConfig, "get_choice_class", lambda name: PluginConfig)
    monkeypatch.setattr(reward_factory.importlib, "import_module", raise_missing_module)

    with pytest.raises(ValueError, match="Reward model class for 'plugin' is not implemented"):
        reward_factory.get_reward_model_class("plugin")


def test_factory_unknown_raises():
    """Unknown name should raise ValueError listing the registered reward models."""
    from lerobot.rewards.factory import get_reward_model_class

    with pytest.raises(ValueError, match="Unknown reward model name 'nonexistent_reward_model'.*sarm"):
        get_reward_model_class("nonexistent_reward_model")


def _robometer_config(**kwargs):
    from lerobot.rewards.robometer import RobometerConfig

    return RobometerConfig(
        device="cpu", vlm_config={"model_type": "fake", "text_config": {"vocab_size": 10}}, **kwargs
    )


def _sarm_config(**kwargs):
    from lerobot.rewards.sarm import SARMConfig

    return SARMConfig(device="cpu", **kwargs)


def _topreward_config(**kwargs):
    from lerobot.rewards.topreward import TOPRewardConfig

    return TOPRewardConfig(device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("make_config", "scoring_options"),
    [
        (_robometer_config, {"num_subsampled_frames": 2}),
        (_sarm_config, {"head_mode": "both", "stride": 5}),
        (_topreward_config, {"num_samples": 3}),
    ],
    ids=["robometer", "sarm", "topreward"],
)
def test_saved_reward_model_configs_keep_scoring_options(tmp_path, make_config, scoring_options):
    make_config(**scoring_options).save_pretrained(tmp_path)

    loaded = RewardModelConfig.from_pretrained(tmp_path)
    assert {name: getattr(loaded, name) for name in scoring_options} == scoring_options


def test_make_frame_scorer_uses_reward_model_naming_convention(monkeypatch):
    import lerobot.rewards.factory as reward_factory

    class PluginConfig:
        type = "plugin"

    PluginConfig.__module__ = "example.configuration_plugin"
    model = SimpleNamespace(config=PluginConfig())
    expected = object()
    captured = {}

    def make_plugin_frame_scorer(*, model, batch_size):
        captured.update(model=model, batch_size=batch_size)
        return expected

    def import_module(name):
        captured.update(module=name)
        return SimpleNamespace(make_plugin_frame_scorer=make_plugin_frame_scorer)

    monkeypatch.setattr(reward_factory.importlib, "import_module", import_module)

    scorer = reward_factory.make_frame_scorer(model, batch_size=4)

    assert scorer is expected
    assert captured == {"module": "example.scoring_plugin", "model": model, "batch_size": 4}


def test_make_frame_scorer_preserves_nested_import_errors(monkeypatch):
    import lerobot.rewards.factory as reward_factory

    class PluginConfig:
        type = "plugin"

    PluginConfig.__module__ = "example.configuration_plugin"
    model = SimpleNamespace(config=PluginConfig())

    def raise_missing_dependency(_module_path):
        raise ModuleNotFoundError("No module named 'plugin_dependency'", name="plugin_dependency")

    monkeypatch.setattr(reward_factory.importlib, "import_module", raise_missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="plugin_dependency"):
        reward_factory.make_frame_scorer(model)


def test_make_frame_scorer_reports_missing_scoring_module(monkeypatch):
    import lerobot.rewards.factory as reward_factory

    class PluginConfig:
        type = "plugin"

    PluginConfig.__module__ = "example.configuration_plugin"
    model = SimpleNamespace(config=PluginConfig())
    module_path = "example.scoring_plugin"

    def raise_missing_module(_module_path):
        raise ModuleNotFoundError(f"No module named {module_path!r}", name=module_path)

    monkeypatch.setattr(reward_factory.importlib, "import_module", raise_missing_module)

    with pytest.raises(ValueError, match="Frame scorer.*not implemented"):
        reward_factory.make_frame_scorer(model)


def test_make_frame_scorer_reports_missing_factory_function(monkeypatch):
    import lerobot.rewards.factory as reward_factory

    class PluginConfig:
        type = "plugin"

    PluginConfig.__module__ = "example.configuration_plugin"
    model = SimpleNamespace(config=PluginConfig())
    monkeypatch.setattr(reward_factory.importlib, "import_module", lambda name: SimpleNamespace())

    with pytest.raises(ValueError, match="Frame scorer.*not implemented"):
        reward_factory.make_frame_scorer(model)


def test_pretrained_reward_model_compute_reward_points_to_native_methods():
    model = _DummyHubReward(_DummyHubRewardConfig())
    with pytest.raises(NotImplementedError, match=r"_DummyHubReward\.compute_reward\(\) was removed"):
        model.compute_reward({})


def test_pretrained_reward_model_requires_config_class():
    """Subclass without config_class should fail."""
    with pytest.raises(TypeError, match="must define 'config_class'"):

        class BadModel(PreTrainedRewardModel):
            name = "bad"


def test_pretrained_reward_model_requires_name():
    """Subclass without name should fail."""
    with pytest.raises(TypeError, match="must define 'name'"):

        class BadModel(PreTrainedRewardModel):
            config_class = RewardModelConfig


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
    """A reward model that does not override ``forward`` is zero-shot."""
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


def test_reward_model_from_pretrained_applies_config_overrides(tmp_path: Path) -> None:
    original = _DummyHubReward(_DummyHubRewardConfig(device="cpu", private=False))
    with torch.no_grad():
        original.bias.fill_(0.75)
    original.save_pretrained(tmp_path)

    loaded = _DummyHubReward.from_pretrained(tmp_path, cli_overrides=["--private=true"], strict=True)

    assert loaded.config.private is True
    assert torch.equal(loaded.bias, original.bias)
    assert loaded.training is False


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
    """Legacy top-level RA-BC fields should be migrated into ``sample_weighting``, except the progress file."""
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
    assert loaded.sample_weighting.kappa == 0.05
    assert loaded.sample_weighting.epsilon == 1e-5
    assert loaded.sample_weighting.score_name is None
    assert loaded.sample_weighting.signal_name is None


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


@pytest.mark.parametrize("weighting_type", ["uniform", "rabc"])
def test_train_pipeline_config_from_pretrained_strips_progress_file_fields(tmp_path, weighting_type, caplog):
    """A ``sample_weighting`` saved with ``progress_path`` and ``head_mode`` should load without them."""
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

    payload["sample_weighting"] = {
        "type": weighting_type,
        "score_name": "sarm",
        "signal_name": "sarm.sparse_progress",
        "progress_path": None,
        "head_mode": "sparse",
        "kappa": 0.02,
        "epsilon": 1e-6,
        "extra_params": {},
    }
    with open(config_path, "w") as f:
        json.dump(payload, f)

    loaded = TrainPipelineConfig.from_pretrained(tmp_path)

    assert loaded.sample_weighting is not None
    assert loaded.sample_weighting.type == weighting_type
    assert loaded.sample_weighting.kappa == 0.02
    assert loaded.sample_weighting.score_name == "sarm"
    assert loaded.sample_weighting.signal_name == "sarm.sparse_progress"
    assert "whose RA-BC read a progress file" not in caplog.text


# ---------------------------------------------------------------------------
# PreTrainedRewardModel hub upload: publish_trained_model + generate_model_card.
# We test the generation side (offline) fully, and the upload side with HfApi
# mocked so nothing actually hits the network.
# ---------------------------------------------------------------------------


def _make_dummy_reward_model(**config_kwargs):
    return _DummyHubReward(_DummyHubRewardConfig(**config_kwargs)), _DummyHubRewardConfig


def _make_train_cfg(dataset_repo_id: str):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    return TrainPipelineConfig(dataset=DatasetConfig(repo_id=dataset_repo_id))


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
        repo_id="user/my_reward_model",
    )

    card = generate_model_card(model.config, cfg=_make_train_cfg("user/my_dataset"))

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
    assert 'RewardModelConfig.from_pretrained("user/my_reward_model")' in body


def test_reward_model_generate_model_card_leaves_an_unknown_license_out(_offline_model_card):
    """When config.license is None and there is no Hub parent, the card states no license."""
    model, _ = _make_dummy_reward_model()

    card = generate_model_card(model.config, cfg=_make_train_cfg("user/my_dataset"))

    assert card.data.license is None


def test_inference_only_reward_model_card_does_not_advertise_training(_offline_model_card):
    card = generate_model_card(TOPRewardConfig(), cfg=_make_train_cfg("user/my_dataset"))
    body = str(card)

    assert "inference-only" in body
    assert "### Train from scratch" not in body
    assert "compute_log_probability" in body


def test_publish_trained_model_uploads_expected_reward_files(monkeypatch, _offline_model_card):
    """Publishing a reward model through ``publish_trained_model`` must:
    1. create the repo,
    2. push the model through ``HubMixin.push_to_hub`` (weights + config.json),
    3. upload a bundle sidecar with train_config.json + the reward-specific README.md.
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

    import lerobot.common.train_utils as train_utils
    import lerobot.utils.hub as hub_module
    from lerobot.common.train_utils import publish_trained_model

    all_files: set[str] = set()

    class _RecordingFakeHfApi(_FakeHfApi):
        def __init__(self, *args, **kwargs):
            pass

        def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message, **_kwargs):
            result = super().upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=folder_path,
                commit_message=commit_message,
            )
            all_files.update(uploaded["files"])
            return result

    monkeypatch.setattr(train_utils, "HfApi", _RecordingFakeHfApi)
    monkeypatch.setattr(hub_module, "HfApi", _RecordingFakeHfApi)

    publish_trained_model(train_cfg, model, None, None, dataset_meta=None)

    assert uploaded["create_repo_id"] == "user/my_reward"
    assert uploaded["upload_repo_id"] == "user/my_reward"
    assert uploaded["upload_repo_type"] == "model"
    # Minimum required files across the publish commits.
    assert CONFIG_NAME in all_files  # config.json (model commit)
    assert TRAIN_CONFIG_NAME in all_files  # train_config.json (bundle commit)
    assert "README.md" in all_files  # reward-specific card (bundle commit)
    assert any(name.endswith(".safetensors") for name in all_files)  # weights (model commit)


def test_from_pretrained_records_loaded_checkpoint(tmp_path):
    """Score provenance relies on the config naming the checkpoint actually loaded."""
    model, _ = _make_dummy_reward_model(pretrained_path="some/other-model")
    model.save_pretrained(tmp_path)

    loaded = _DummyHubReward.from_pretrained(tmp_path)

    assert loaded.config.pretrained_path == str(tmp_path)
    assert loaded.config.pretrained_revision is None


def test_save_pretrained_writes_nothing_off_main_rank(tmp_path, monkeypatch):
    """save_checkpoint calls save_pretrained on every rank; the
    reward serializer must gate its writes so DDP replicas do not race on the same files."""
    import lerobot.distributed.utils as dist_utils

    model, _ = _make_dummy_reward_model()
    monkeypatch.setattr(dist_utils, "is_main_process", lambda: False)
    model.save_pretrained(tmp_path)
    assert not any(tmp_path.iterdir())


def test_reward_model_push_model_to_hub_shim_warns_and_publishes(monkeypatch, _offline_model_card):
    """The deprecated ``push_model_to_hub`` stays callable, delegating to the publisher."""
    from huggingface_hub.constants import CONFIG_NAME

    import lerobot.common.train_utils as train_utils
    import lerobot.utils.hub as hub_module
    from lerobot.configs.train import TRAIN_CONFIG_NAME

    all_files: set[str] = set()

    class _FakeHfApi:
        def __init__(self, *args, **kwargs):
            pass

        def create_repo(self, repo_id, private=None, exist_ok=False, **kwargs):
            return SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, *, repo_id, folder_path, **_kwargs):
            all_files.update(p.name for p in Path(folder_path).iterdir())
            return SimpleNamespace(repo_url=SimpleNamespace(url=f"https://huggingface.co/{repo_id}"))

    monkeypatch.setattr(train_utils, "HfApi", _FakeHfApi)
    monkeypatch.setattr(hub_module, "HfApi", _FakeHfApi)

    model, _ = _make_dummy_reward_model(repo_id="user/my_reward")
    with pytest.warns(FutureWarning, match="push_model_to_hub is deprecated"):
        model.push_model_to_hub(_make_train_cfg("user/my_dataset"))

    assert CONFIG_NAME in all_files
    assert TRAIN_CONFIG_NAME in all_files
    assert "README.md" in all_files


def test_make_reward_model_loads_resume_weights_and_keeps_the_parent(monkeypatch):
    """A resume loads the checkpoint, while `pretrained_path` keeps naming the fine-tuned-from model."""
    import lerobot.rewards.factory as reward_factory

    cfg = SimpleNamespace(
        type="mock", device="cpu", pretrained_path="user/base-reward", pretrained_revision=None
    )
    seen_while_building = []

    def from_pretrained(**kwargs):
        seen_while_building.append(cfg.pretrained_path)
        model = torch.nn.Linear(1, 1)
        model.config = cfg
        return model

    monkeypatch.setattr(
        reward_factory, "get_reward_model_class", lambda _: SimpleNamespace(from_pretrained=from_pretrained)
    )

    model = reward_factory.make_reward_model(cfg, pretrained_path="run/checkpoints/000002/pretrained_model")

    assert seen_while_building == ["run/checkpoints/000002/pretrained_model"]
    assert cfg.pretrained_path == "user/base-reward"
    assert model.config.pretrained_path == "user/base-reward"


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_reward_checkpoint_consent_is_runtime_only(tmp_path, trust_remote_code):
    reward = _DummyHubReward(_DummyHubRewardConfig(device="cpu", trust_remote_code=True))
    reward.save_pretrained(tmp_path)
    loaded = _DummyHubReward.from_pretrained(tmp_path, trust_remote_code=trust_remote_code)
    assert loaded.config.trust_remote_code is trust_remote_code
    torch.testing.assert_close(loaded.bias, reward.bias)


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize(
    ("module_name", "class_name", "loader_name", "config_type"),
    [
        ("topreward.modeling_topreward", "TOPRewardModel", "Qwen3VLForConditionalGeneration", "topreward"),
        ("robometer.modeling_robometer", "RobometerRewardModel", "AutoModelForImageTextToText", "robometer"),
        ("topreward.processor_topreward", "TOPRewardEncoderProcessorStep", "AutoProcessor", None),
        ("robometer.processor_robometer", "RobometerEncoderProcessorStep", "AutoProcessor", None),
        ("robometer.configuration_robometer", "RobometerConfig", "AutoConfig", None),
        ("classifier.modeling_classifier", "Classifier", "AutoModel", "reward_classifier"),
    ],
)
def test_reward_loaders_require_caller_consent(
    monkeypatch, trust_remote_code, module_name, class_name, loader_name, config_type
):
    import importlib
    import sys

    module = importlib.import_module(f"lerobot.rewards.{module_name}")

    class LoaderReachedError(Exception):
        pass

    def from_pretrained(*args, **kwargs):
        assert kwargs["trust_remote_code"] is trust_remote_code
        raise LoaderReachedError

    loader = SimpleNamespace(from_pretrained=from_pretrained)
    if loader_name == "AutoModel":
        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=loader))
    else:
        monkeypatch.setattr(module, loader_name, loader)
    if hasattr(module, "require_package"):
        monkeypatch.setattr(module, "require_package", lambda *args, **kwargs: None)
    constructor = getattr(module, class_name)
    with pytest.raises(LoaderReachedError):
        if config_type is None:
            constructor(trust_remote_code=trust_remote_code)
        else:
            kwargs = {"vlm_config": {"text_config": {"vocab_size": 10}}} if config_type == "robometer" else {}
            config = RewardModelConfig.get_choice_class(config_type)(
                device="cpu", pretrained_path=None, trust_remote_code=trust_remote_code, **kwargs
            )
            constructor(config)


@pytest.mark.parametrize(
    "config_consent,consent", [(False, None), (True, None), (True, False), (False, True)]
)
@pytest.mark.parametrize("topreward", [False, True])
def test_explicit_reward_consent_overrides_supplied_config(
    monkeypatch, tmp_path, consent, config_consent, topreward
):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    model_cls = TOPRewardModel if topreward else _DummyHubReward
    config = model_cls.config_class(device="cpu", trust_remote_code=config_consent)

    class ConstructionReachedError(Exception):
        pass

    def construct(self, received):
        assert received.trust_remote_code is (config_consent if consent is None else consent)
        assert (received is config) is (consent is None)
        raise ConstructionReachedError

    monkeypatch.setattr(model_cls, "__init__", construct)
    kwargs = {} if consent is None else {"trust_remote_code": consent}
    with pytest.raises(ConstructionReachedError):
        model_cls.from_pretrained(tmp_path, config=config, **kwargs)
    assert config.trust_remote_code is config_consent
