#!/usr/bin/env python

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.dm05 import modeling_dm05 as dm05_modeling
from lerobot.policies.dm05.configuration_dm05 import DM05Config
from lerobot.policies.dm05.conversion_dm05 import DM05LerobotBatchConverter
from lerobot.policies.dm05.modeling_dm05 import DM05Policy
from lerobot.policies.dm05.processor_dm05 import DM05TaskProcessor, make_dm05_pre_post_processors
from lerobot.policies.dm05.tokenization_dm05 import DM05Tokenization, get_camera_labels
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


class _IdentityNormalizer:
    def normalize_state(self, state):
        return state.to(torch.float32)

    def normalize_action(self, action):
        return action.to(torch.float32)


class _FakeTokenizer:
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()

    def tokenize_robot(self, **kwargs):
        return {
            "input_ids": torch.tensor([1, 2], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([1, 2], dtype=torch.long),
            "token_type_ids": torch.tensor([0, 0], dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }

    def tokenize_robot_infer(self, **kwargs):
        return {
            "input_ids": torch.tensor([1], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "token_type_ids": torch.tensor([0], dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }


class _PadZeroEosOneTokenizer:
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 1})()

    def tokenize_robot(self, **kwargs):
        ids = torch.tensor([5, 6] if kwargs["prompt"] == "long" else [7], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
            "token_type_ids": torch.zeros_like(ids),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }

    def tokenize_robot_infer(self, **kwargs):
        ids = torch.tensor([5, 6] if kwargs["prompt"] == "long" else [7], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "token_type_ids": torch.zeros_like(ids),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }


class _FakeProcessor:
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()


class _FakeAutoProcessor:
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "unk_token_id": -1,
            "convert_tokens_to_ids": lambda self, token: 7,
        },
    )()


class _FakeCoreConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(model_type="dexbotic_dm05", vlm_config={}, action_config={})


class _FakeCoreModel:
    from_pretrained_calls: list[tuple[str, dict]] = []

    def __init__(self, config):
        self.config = config
        self.embed_tokens = torch.nn.Embedding(2, 2)
        self.model = SimpleNamespace(
            vlm=SimpleNamespace(
                model=SimpleNamespace(
                    language_model=SimpleNamespace(embed_tokens=self.embed_tokens),
                ),
            ),
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        cls.from_pretrained_calls.append((str(pretrained_name_or_path), kwargs))
        config = kwargs.get("config")
        if config is None:
            config = _FakeCoreConfig(model_type="dexbotic_dm05")
        return cls(config)

    def to(self, *args, **kwargs):
        return self


class _FakeTokenization:
    def __init__(self, **kwargs):
        pass


class _FakeSaveCoreConfig:
    def to_dict(self):
        return {
            "model_type": "dexbotic_dm05",
            "vlm_config": {"image_token_index": 42},
            "action_config": {"hidden_size": 8},
        }


class _FakeSaveModel:
    def __init__(self):
        self.config = _FakeSaveCoreConfig()
        self.prepare_called = False

    def prepare_config_for_save(self):
        self.prepare_called = True


def _dm05_config(**kwargs) -> DM05Config:
    config = DM05Config(
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        max_state_dim=32,
        max_action_dim=32,
        **kwargs,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    return config


def _convert(
    config: DM05Config,
    batch: dict,
    include_labels: bool = True,
    tokenization_cls: type = _FakeTokenizer,
) -> dict:
    converter = DM05LerobotBatchConverter(
        config=config,
        tokenization_cls=lambda **_: tokenization_cls(),
        processor=_FakeProcessor(),
        normalizer=_IdentityNormalizer(),
    )
    return converter.convert_lerobot_batch(batch, include_labels=include_labels)


def _core_config_payload() -> dict:
    return {
        "model_type": "dexbotic_dm05",
        "vlm_config": {"image_token_index": 42},
        "action_config": {"hidden_size": 8},
    }


def _write_lerobot_dm05_checkpoint(path, **kwargs) -> DM05Config:
    config_kwargs = {
        "pretrained_name_or_path": ".",
        "processor_name_or_path": ".",
        "norm_stats_path": "norm_stats.json",
        "core_config": _core_config_payload(),
    }
    config_kwargs.update(kwargs)
    config = _dm05_config(**config_kwargs)
    config.save_pretrained(path)
    (path / "model.safetensors").write_bytes(b"")
    (path / "processor_config.json").write_text("{}\n")
    (path / "norm_stats.json").write_text("{}\n")
    return config


def _patch_dm05_runtime(monkeypatch):
    _FakeCoreModel.from_pretrained_calls = []
    monkeypatch.setattr(
        dm05_modeling,
        "import_dm05_core",
        lambda: (_FakeCoreConfig, _FakeCoreModel, _FakeTokenization),
    )
    monkeypatch.setattr(dm05_modeling, "resolve_dm05_normalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(dm05_modeling, "_build_dm05_norm_dataset_from_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: _FakeAutoProcessor(),
    )


def test_dm05_config_defaults_and_validation():
    config = make_policy_config(policy_type="dm05", chunk_size=10, n_action_steps=10)

    assert isinstance(config, DM05Config)
    assert config.pretrained_name_or_path == "Dexmal/DM05"
    assert (
        config.chunk_size,
        config.n_action_steps,
        config.action_mode,
        config.vlm_gradient_checkpointing,
        config.ae_gradient_checkpointing,
        config.ae_gradient_checkpointing_layers,
    ) == (10, 10, "absolute", True, True, 1)

    override_config = _dm05_config(
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        ae_gradient_checkpointing_layers=2,
    )
    assert (
        override_config.vlm_gradient_checkpointing,
        override_config.ae_gradient_checkpointing,
        override_config.ae_gradient_checkpointing_layers,
    ) == (False, False, 2)

    for kwargs, match in [
        ({"action_mode": "delta"}, "action_mode"),
        ({"compile_suffix": "silent"}, "compile_suffix"),
        ({"compile_suffix_warmup_steps": -1}, "compile_suffix_warmup_steps"),
    ]:
        with pytest.raises(ValueError, match=match):
            DM05Config(**kwargs)


def test_dm05_tokenization_builds_opendm_style_user_content_without_random_branches():
    tokenization = DM05Tokenization(
        processor=_FakeProcessor(),
        n_bins=256,
        add_state=False,
    )
    images = [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))]
    meta = {
        "robot_type": "franka",
        "control_mode": "relative",
        "dataset_meta": {"image_keys": ["images_1", "observation.images.left_wrist"]},
    }

    user_content = tokenization._build_user_content(
        prompt="Pick up the mug.",
        images=images,
        state=np.array([-1.0, 1.0], dtype=np.float32),
        meta_data=meta,
        speed_text="0.5",
    )

    assert user_content[0]["text"] == (
        "Robot: franka\nControl mode: relative\nOverall speed: 0.5\nTask: Pick up the mug.\nHead image: "
    )
    assert user_content[1]["type"] == "image"
    assert user_content[2]["text"] == "Left wrist image: "
    assert user_content[3]["type"] == "image"
    assert all("State:" not in item.get("text", "") for item in user_content)

    state_tokenization = DM05Tokenization(processor=_FakeProcessor(), n_bins=256, add_state=True)
    state_content = state_tokenization._build_user_content(
        prompt="Pick up the mug.",
        images=[images[0]],
        state=np.array([-1.0, 1.0], dtype=np.float32),
        meta_data={"dataset_meta": {"image_keys": ["images_1"]}},
        speed_text=None,
    )

    assert state_content[-1]["text"] == "State: 0 255\n"
    assert get_camera_labels({"dataset_meta": {"image_keys": ["images_1", "images_2"]}}, 2) == [
        "Head image: ",
        "Left wrist image: ",
    ]


def test_dm05_processors_roundtrip(tmp_path):
    processor = DM05TaskProcessor(default_task="Default instruction.")

    assert processor.complementary_data({})["task"] == "Default instruction."

    config = _dm05_config()
    preprocessor, postprocessor = make_dm05_pre_post_processors(config=config)

    preprocessor.save_pretrained(tmp_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(tmp_path, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)

    assert any(isinstance(step, DM05TaskProcessor) for step in loaded_preprocessor.steps)
    assert not list(tmp_path.glob("*normalizer_processor.safetensors"))

    default_processed = loaded_preprocessor(
        {
            OBS_STATE: torch.zeros(3),
            "observation.images.front": torch.zeros(3, 2, 2),
        }
    )
    action = torch.tensor([[4.0, 5.0, 6.0]])

    assert default_processed["task"] == "Execute the robot action."
    assert default_processed[OBS_STATE].shape == (1, 3)
    assert default_processed["observation.images.front"].shape == (1, 3, 2, 2)
    assert torch.equal(loaded_postprocessor(action), action)


def test_dm05_policy_save_pretrained_bundles_assets_norm_and_core_config(monkeypatch, tmp_path):
    source_path = tmp_path / "source"
    save_path = tmp_path / "saved"
    source_path.mkdir()
    (source_path / "tokenizer.json").write_text("{}\n")
    norm_payload = {"norm_stats": {"default": {"min": -1, "max": 1}}}
    (source_path / "norm_stats.json").write_text(json.dumps(norm_payload) + "\n")

    saved_model_paths = []
    monkeypatch.setattr(
        dm05_modeling,
        "save_model_as_safetensor",
        lambda model, path: saved_model_paths.append((model, Path(path))),
    )

    policy = object.__new__(DM05Policy)
    policy.config = _dm05_config(
        pretrained_name_or_path=str(source_path),
        processor_name_or_path=str(source_path),
        norm_stats_path=str(source_path / "norm_stats.json"),
    )
    policy.config._dm05_checkpoint_dir = str(source_path)
    policy.model = _FakeSaveModel()

    policy._save_pretrained(save_path)

    saved_config = json.loads((save_path / "config.json").read_text())
    assert (save_path / "tokenizer.json").is_file()
    assert json.loads((save_path / "norm_stats.json").read_text()) == norm_payload
    assert saved_config["pretrained_name_or_path"] == "."
    assert saved_config["processor_name_or_path"] == "."
    assert saved_config["norm_stats_path"] == "norm_stats.json"
    assert saved_config["core_config"]["model_type"] == "dexbotic_dm05"
    assert policy.model.prepare_called is True
    assert saved_model_paths == [(policy.model, save_path / "model.safetensors")]
    assert not any(key.startswith("_dm05_") for key in saved_config)


def test_dm05_policy_save_pretrained_accepts_policy_prefixed_state_dict(tmp_path):
    save_path = tmp_path / "saved"
    model = torch.nn.Linear(2, 3)
    policy = object.__new__(DM05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = _dm05_config()
    policy.model = model
    state_dict = {f"model.{key}": value.detach().clone() for key, value in model.state_dict().items()}

    policy.save_pretrained(save_path, state_dict=state_dict)

    saved = load_file(save_path / dm05_modeling.SAFETENSORS_SINGLE_FILE)
    assert set(saved) == set(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(saved[key], value)


def test_dm05_sft_base_and_lerobot_checkpoint_loading(monkeypatch, tmp_path):
    base_path = tmp_path / "dm05_base"
    checkpoint_path = tmp_path / "lerobot_checkpoint"
    base_path.mkdir()
    _write_lerobot_dm05_checkpoint(checkpoint_path)
    _patch_dm05_runtime(monkeypatch)

    base_config = _dm05_config(
        pretrained_name_or_path=str(base_path),
        processor_name_or_path=str(base_path),
    )
    DM05Policy(base_config, dataset_meta=object())
    base_load = _FakeCoreModel.from_pretrained_calls[-1]

    assert base_load[0] == str(base_path)
    assert "config" not in base_load[1]
    assert (
        base_load[1]["chunk_size"],
        base_load[1]["vlm_gradient_checkpointing"],
        base_load[1]["ae_gradient_checkpointing"],
        base_load[1]["ae_gradient_checkpointing_layers"],
    ) == (2, True, True, 1)
    assert base_config.norm_stats_path is None

    checkpoint_config = PreTrainedConfig.from_pretrained(checkpoint_path)
    checkpoint_config.pretrained_path = checkpoint_path
    DM05Policy.from_pretrained(
        checkpoint_path,
        config=checkpoint_config,
        dataset_meta=object(),
    )
    checkpoint_load = _FakeCoreModel.from_pretrained_calls[-1]

    assert checkpoint_load[0] == str(checkpoint_path)
    assert isinstance(checkpoint_load[1]["config"], _FakeCoreConfig)
    assert checkpoint_config.pretrained_path is None
    assert checkpoint_config.pretrained_name_or_path == str(checkpoint_path)
    assert checkpoint_config.processor_name_or_path == str(checkpoint_path)
    assert checkpoint_config.norm_stats_path is None


def test_dm05_batch_conversion_absolute_and_relative():
    batch = {
        OBS_STATE: torch.tensor([[1.0, 2.0, 3.0]]),
        ACTION: torch.tensor([[4.0, 5.0, 6.0]]),
        "observation.images.front": torch.zeros(1, 3, 2, 2),
    }
    absolute = _convert(_dm05_config(action_mode="absolute"), batch)

    assert absolute["actions"].shape == (1, 2, 32)
    assert torch.equal(absolute["actions"][0, 0, :3], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(absolute["action_dim_mask"][0, :5], torch.tensor([True, True, True, False, False]))
    assert absolute["states"].shape == (1, 32)

    config = _dm05_config(action_mode="relative")
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    config.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    relative = _convert(
        config,
        {
            OBS_STATE: torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
            ACTION: torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]]),
            "observation.images.front": torch.zeros(1, 3, 2, 2),
        },
    )

    assert torch.equal(
        relative["actions"][0, 0, :7],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 14.0]),
    )


def test_dm05_collate_keeps_zero_pad_token_when_eos_differs():
    batch = {
        OBS_STATE: torch.zeros(2, 3),
        ACTION: torch.zeros(2, 3),
        "observation.images.front": torch.zeros(2, 3, 2, 2),
        "task": ["long", "short"],
    }

    out = _convert(_dm05_config(), batch, tokenization_cls=_PadZeroEosOneTokenizer)

    assert torch.equal(out["input_ids"], torch.tensor([[5, 6], [7, 0]]))
    assert torch.equal(out["labels"], torch.tensor([[5, 6], [7, -100]]))
