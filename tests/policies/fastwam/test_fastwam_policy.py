#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json

import pytest
import torch
from safetensors.torch import save_model
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import FastWAMConfig, get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.fastwam import modeling_fastwam
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy, resolve_wan_component_paths
from lerobot.policies.fastwam.processor_fastwam import FastWAMActionToggleProcessorStep
from lerobot.policies.fastwam.wan_components import (
    WAN_DIT_PATTERN,
    WAN_T5_CHECKPOINT,
    WAN_T5_TOKENIZER,
    WAN_VAE_CHECKPOINT,
    resolve_wan_checkpoint_paths,
)
from lerobot.utils.constants import ACTION, OBS_STATE


class FakeFastWAMCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(2, 2)

    def training_loss(self, sample):
        assert sample["video"].ndim == 5
        assert sample["context"].ndim == 3
        return sample[ACTION].sum() * 0.0 + torch.tensor(1.0), {"loss_action": 1.0}

    def infer_action(self, **kwargs):
        return {"action": torch.ones(1, kwargs["action_horizon"], 3)}


def test_fastwam_is_registered_and_publicly_exported():
    cfg = make_policy_config(
        "fastwam",
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        base_model_id=None,
    )

    assert isinstance(cfg, FastWAMConfig)
    assert cfg.type == "fastwam"
    assert get_policy_class("fastwam") is FastWAMPolicy


def test_config_validates_features_model_ids_and_saved_auto_route(tmp_path):
    cfg = FastWAMConfig()
    cfg.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())

    assert saved["pretrained_path"] is None
    assert cfg.image_features["observation.images.image"].type == FeatureType.VISUAL
    assert cfg.action_feature.shape == (7,)
    assert cfg.robot_state_feature.shape == (8,)
    with pytest.raises(ValueError, match="image feature"):
        FastWAMConfig(input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))})
    with pytest.raises(ValueError, match="tokenizer_model_id"):
        FastWAMConfig(tokenizer_model_id="somebody/other-tokenizer")


def test_preprocessor_normalizes_images_and_postprocessor_toggles_actions(tmp_path):
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        image_size=(2, 2),
        device="cpu",
        toggle_action_dimensions=[-1],
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        base_model_id=None,
    )
    dataset_stats = {
        "observation.images.image": {
            "mean": torch.full((3, 1, 1), 0.2),
            "std": torch.full((3, 1, 1), 0.1),
        },
        OBS_STATE: {
            "mean": torch.tensor([1.0, 3.0]),
            "std": torch.tensor([2.0, 4.0]),
        },
        ACTION: {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
    processed = preprocessor(
        {
            "observation.images.image": torch.tensor(
                [
                    [[0.0, 0.5], [1.0, 0.5]],
                    [[0.0, 0.5], [1.0, 0.5]],
                    [[0.0, 0.5], [1.0, 0.5]],
                ]
            ),
            OBS_STATE: torch.tensor([3.0, 7.0]),
        }
    )
    preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    _, loaded_postprocessor = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))

    expected_image = torch.tensor(
        [[[[-1.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]]]]
    )
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"
    assert torch.allclose(processed["observation.images.image"], expected_image)
    assert torch.allclose(processed[OBS_STATE], torch.tensor([[1.0, 1.0]]))
    assert torch.equal(dataset_stats["observation.images.image"]["mean"], torch.full((3, 1, 1), 0.2))
    assert any(isinstance(step, FastWAMActionToggleProcessorStep) for step in loaded_postprocessor.steps)
    assert torch.equal(
        loaded_postprocessor(torch.tensor([[0.25, 0.5, 1.0]])), torch.tensor([[0.25, 0.5, -1.0]])
    )


def test_policy_forward_and_predict_action_adapt_lerobot_batches(monkeypatch):
    captured = []

    class CapturingCore(FakeFastWAMCore):
        def infer_action(self, **kwargs):
            captured.append(
                {
                    "image_shape": tuple(kwargs["input_image"].shape),
                    "proprio_shape": tuple(kwargs["proprio"].shape),
                    "prompt": kwargs["prompt"],
                }
            )
            return {"action": torch.full((1, kwargs["action_horizon"], 3), float(len(captured)))}

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: CapturingCore())
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        image_size=(16, 16),
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        base_model_id=None,
    )
    with pytest.warns(RuntimeWarning, match="does not load pretrained FastWAM weights"):
        policy = FastWAMPolicy(cfg)

    output = policy.forward(
        {
            "observation.images.image": torch.zeros(1, 3, 16, 16),
            OBS_STATE: torch.zeros(1, 2),
            ACTION: torch.zeros(1, 4, 3),
            "context": torch.zeros(1, 5, 4096),
            "context_mask": torch.ones(1, 5, dtype=torch.bool),
        }
    )
    action = policy.predict_action_chunk(
        {
            "observation.images.image": torch.stack(
                [
                    torch.zeros(3, 16, 16),
                    torch.ones(3, 16, 16),
                ]
            ),
            OBS_STATE: torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "task": ["task 0", "task 1"],
        }
    )

    assert output["loss"].item() == 1.0
    assert output["loss_action"].item() == 1.0
    assert action.shape == (2, 4, 3)
    assert action[:, 0, 0].tolist() == [1.0, 2.0]
    assert [item["image_shape"] for item in captured] == [(1, 3, 16, 16), (1, 3, 16, 16)]
    assert [item["proprio_shape"] for item in captured] == [(1, 2), (1, 2)]
    assert [item["prompt"] for item in captured] == [
        cfg.prompt_template.format(task="task 0"),
        cfg.prompt_template.format(task="task 1"),
    ]


def test_from_pretrained_loads_weights_without_initializing_wan_backbone(monkeypatch, tmp_path):
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2, base_model_id=None)
    cfg.save_pretrained(tmp_path)
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: FakeFastWAMCore())
    reference_policy = FastWAMPolicy(cfg, _suppress_base_init_warning=True)
    save_model(reference_policy, str(tmp_path / "model.safetensors"))

    def fail_if_wan_pretrained_is_loaded(*args, **kwargs):
        raise AssertionError("from_pretrained must not initialize or download Wan2.2 backbone components")

    monkeypatch.setattr(
        "lerobot.policies.fastwam.modular_fastwam.FastWAM.from_wan22_pretrained",
        fail_if_wan_pretrained_is_loaded,
    )
    monkeypatch.setattr(
        modeling_fastwam,
        "_build_core_model_from_architecture",
        lambda config: FakeFastWAMCore(),
        raising=False,
    )
    loaded_components_from = []
    monkeypatch.setattr(
        FastWAMPolicy,
        "load_wan_components_from_pretrained",
        lambda self, path: loaded_components_from.append(path),
    )

    policy = FastWAMPolicy.from_pretrained(tmp_path, strict=False)

    assert isinstance(policy.model, FakeFastWAMCore)
    assert loaded_components_from == [tmp_path]


def test_save_pretrained_copies_required_wan_sidecars(monkeypatch, tmp_path):
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2, base_model_id=None)
    source = tmp_path / "source"
    tokenizer = source / WAN_T5_TOKENIZER
    tokenizer.mkdir(parents=True)
    vae = source / WAN_VAE_CHECKPOINT
    text_encoder = source / WAN_T5_CHECKPOINT
    tokenizer_file = tokenizer / "tokenizer.json"
    vae.write_bytes(b"vae")
    text_encoder.write_bytes(b"text")
    tokenizer_file.write_text("{}")
    core = FakeFastWAMCore()
    core.model_paths = {
        "vae": str(vae),
        "text_encoder": str(text_encoder),
        "tokenizer": str(tokenizer),
    }
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: core)
    policy = FastWAMPolicy(cfg, _suppress_base_init_warning=True)

    save_dir = tmp_path / "saved"
    policy.save_pretrained(save_dir)

    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / WAN_VAE_CHECKPOINT).read_bytes() == b"vae"
    assert (save_dir / WAN_T5_CHECKPOINT).read_bytes() == b"text"
    assert (save_dir / WAN_T5_TOKENIZER / "tokenizer.json").read_text() == "{}"


def test_wan_component_resolution_uses_fixed_safetensors_layout(tmp_path):
    tokenizer = tmp_path / WAN_T5_TOKENIZER
    tokenizer.mkdir(parents=True)
    (tmp_path / WAN_VAE_CHECKPOINT).touch()
    (tmp_path / WAN_T5_CHECKPOINT).touch()
    (tmp_path / "diffusion_pytorch_model-00001-of-00001.safetensors").touch()
    (tokenizer / "tokenizer.json").touch()

    paths = resolve_wan_checkpoint_paths(tmp_path)
    sidecar_paths = resolve_wan_component_paths(tmp_path)

    assert paths.dit == [tmp_path / "diffusion_pytorch_model-00001-of-00001.safetensors"]
    assert paths.vae == tmp_path / WAN_VAE_CHECKPOINT
    assert paths.text_encoder == tmp_path / WAN_T5_CHECKPOINT
    assert paths.tokenizer == tmp_path / WAN_T5_TOKENIZER
    assert sidecar_paths.dit == []
    assert WAN_DIT_PATTERN == "diffusion_pytorch_model*.safetensors"

    (tmp_path / WAN_T5_CHECKPOINT).unlink()
    with pytest.raises(FileNotFoundError, match="text encoder"):
        resolve_wan_checkpoint_paths(tmp_path)


def test_pretrained_config_round_trips_fastwam_features(tmp_path):
    cfg = FastWAMConfig(action_dim=7, proprio_dim=8, image_size=(224, 448), base_model_id=None)
    cfg.save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.type == "fastwam"
    assert loaded.image_features["observation.images.image"].type == FeatureType.VISUAL
    assert loaded.action_feature.shape == (7,)
    assert loaded.robot_state_feature.shape == (8,)
