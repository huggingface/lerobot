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
from safetensors import safe_open
from torch import nn

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import FastWAMConfig, get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.processor_fastwam import FastWAMActionToggleProcessorStep
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
        num_video_frames=5,
        action_video_freq_ratio=1,
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
    assert FastWAMConfig(tokenizer_model_id="somebody/other-tokenizer").tokenizer_model_id == (
        "somebody/other-tokenizer"
    )


def test_preprocessor_passes_images_through_and_postprocessor_toggles_actions(tmp_path):
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
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

    # VISUAL normalization is IDENTITY
    expected_image = torch.tensor(
        [[[[0.0, 0.5], [1.0, 0.5]], [[0.0, 0.5], [1.0, 0.5]], [[0.0, 0.5], [1.0, 0.5]]]]
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
        num_video_frames=5,
        action_video_freq_ratio=1,
        image_size=(16, 16),
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        base_model_id=None,
    )
    policy = FastWAMPolicy(cfg)

    loss, metrics = policy.forward(
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

    assert loss.item() == 1.0
    assert metrics["loss_action"] == 1.0
    assert action.shape == (2, 4, 3)
    assert action[:, 0, 0].tolist() == [1.0, 2.0]
    assert [item["image_shape"] for item in captured] == [(1, 3, 16, 16), (1, 3, 16, 16)]
    assert [item["proprio_shape"] for item in captured] == [(1, 2), (1, 2)]
    assert [item["prompt"] for item in captured] == [
        cfg.prompt_template.format(task="task 0"),
        cfg.prompt_template.format(task="task 1"),
    ]


class CoreWithFrozenComponents(FakeFastWAMCore):
    """Fake core mirroring the real one: frozen VAE / text encoder held as
    *unregistered* attributes (via `object.__setattr__`) so they are excluded from
    `state_dict()` and the saved checkpoint, but still moved by the `_apply` override."""

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "vae", nn.Linear(2, 2))
        object.__setattr__(self, "text_encoder", nn.Linear(2, 2))
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.vae._apply(fn)
        self.text_encoder._apply(fn)
        return self


def test_from_pretrained_uses_base_loader_and_skips_wan_backbone(monkeypatch, tmp_path):
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
        base_model_id=None,
    )

    def build_core(self, config):
        core = CoreWithFrozenComponents()
        with torch.no_grad():
            core.dit.weight.fill_(0.5)
        return core

    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", build_core)

    reference = FastWAMPolicy(cfg)
    with torch.no_grad():
        reference.model.dit.weight.fill_(1.25)  # a distinctive, trained-looking weight
    reference.save_pretrained(tmp_path)

    # Building from Wan2.2 must never happen on a checkpoint load.
    def fail_if_wan_pretrained_is_loaded(*args, **kwargs):
        raise AssertionError("from_pretrained must not initialize or download the Wan2.2 backbone")

    monkeypatch.setattr(
        "lerobot.policies.fastwam.wan.modular.FastWAM.from_wan22_pretrained",
        fail_if_wan_pretrained_is_loaded,
    )

    policy = FastWAMPolicy.from_pretrained(tmp_path)

    assert isinstance(policy.model, CoreWithFrozenComponents)
    # The bundled checkpoint weights overwrote the freshly built (0.5) DiT weights.
    assert torch.allclose(policy.model.dit.weight, torch.full_like(policy.model.dit.weight, 1.25))


def test_save_pretrained_excludes_frozen_components(monkeypatch, tmp_path):
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
        base_model_id=None,
    )
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: CoreWithFrozenComponents())
    policy = FastWAMPolicy(cfg)

    save_dir = tmp_path / "saved"
    policy.save_pretrained(save_dir)

    assert (save_dir / "model.safetensors").is_file()
    # No Wan sidecar files either: the frozen backbone comes from the diffusers repo.
    assert not (save_dir / "Wan2.2_VAE.safetensors").exists()
    assert not (save_dir / "google").exists()

    with safe_open(save_dir / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())
    # Lean checkpoint: only the trainable DiT is saved; the frozen VAE / UMT5 text
    # encoder are excluded (loaded from the diffusers/transformers repos at init).
    assert any(key.startswith("model.dit.") for key in keys)
    assert not any(key.startswith("model.vae.") for key in keys)
    assert not any(key.startswith("model.text_encoder.") for key in keys)


def test_frozen_components_excluded_from_params_but_follow_device_moves(monkeypatch):
    cfg = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
        base_model_id=None,
    )
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: CoreWithFrozenComponents())
    policy = FastWAMPolicy(cfg)

    # Unregistered: excluded from state_dict and from the optimizer's parameter set.
    sd = policy.state_dict()
    assert not any(k.startswith("model.vae.") or k.startswith("model.text_encoder.") for k in sd)
    param_names = [n for n, _ in policy.named_parameters()]
    assert not any("vae" in n or "text_encoder" in n for n in param_names)

    # ...but the `_apply` override still carries them through `.to()` (dtype stands in
    # for device on a CPU box), so they never strand off the rest of the model.
    policy.to(torch.float64)
    assert policy.model.dit.weight.dtype == torch.float64  # registered
    assert policy.model.vae.weight.dtype == torch.float64  # unregistered, moved via _apply
    assert policy.model.text_encoder.weight.dtype == torch.float64


def test_pretrained_config_round_trips_fastwam_features(tmp_path):
    cfg = FastWAMConfig(action_dim=7, proprio_dim=8, image_size=(224, 448), base_model_id=None)
    cfg.save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.type == "fastwam"
    assert loaded.image_features["observation.images.image"].type == FeatureType.VISUAL
    assert loaded.action_feature.shape == (7,)
    assert loaded.robot_state_feature.shape == (8,)


def test_vae_adapter_empty_build_encode_decode_shapes():
    """Offline glue check of the diffusers-backed VAE adapter (random weights).

    Validates the encode/decode contract — 48 latent channels, 16x spatial / 4x
    temporal compression, list-or-batch input, scaling round-trip — without any
    weight download. (Numerical fidelity vs the original Wan VAE is a separate,
    GPU + real-weights verification step.)
    """
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    from lerobot.policies.fastwam.wan import WanVideoVAE38

    # Production always loads a real pretrained VAE from the diffusers repo; here we
    # build the same architecture with random weights and dummy standardization stats
    # to exercise the adapter's shape/scaling contract offline (fidelity is checked
    # separately, with real weights, on GPU).
    arch = {
        "base_dim": 160,
        "decoder_base_dim": 256,
        "z_dim": 48,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temporal_downsample": [False, True, True],
        "dropout": 0.0,
        "is_residual": True,
        "in_channels": 12,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 4,
        "clip_output": False,
        "latents_mean": [0.0] * 48,
        "latents_std": [1.0] * 48,
    }
    raw = AutoencoderKLWan.from_config(arch)
    vae = WanVideoVAE38(dtype=torch.float32, device="cpu", pretrained=raw)
    assert vae.z_dim == 48
    assert vae.upsampling_factor == 16
    assert vae.temporal_downsample_factor == 4

    video = torch.rand(1, 3, 5, 32, 32) * 2 - 1  # [B,C,T,H,W] in [-1,1]
    latents = vae.encode(video)
    assert latents.shape == (1, 48, 2, 2, 2)  # T'=(5-1)//4+1, H'=W'=32//16

    decoded = vae.decode(latents)
    assert decoded.shape[0] == 1 and decoded.shape[1] == 3 and decoded.shape[-2:] == (32, 32)
    assert decoded.min() >= -1.0 and decoded.max() <= 1.0

    # list input is accepted and equals the batched path
    assert torch.equal(vae.encode([video[0]]), latents)
