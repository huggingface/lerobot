#!/usr/bin/env python

# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""CPU tests of the `cae_fastwam` policy with tiny offline components (no Wan2.2 / UMT5 weights)."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="cae_fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="cae_fastwam requires the `fastwam` extra (diffusers)")

import lerobot.policies as lerobot_policies
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import CAEFastWAMConfig, get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.cae_fastwam.configuration_cae_fastwam import CAE_FASTWAM_BASE_MODEL_ID
from lerobot.policies.cae_fastwam.modeling_cae_fastwam import (
    CAEActionDecoder,
    CAEActionEncoder,
    CAEFastWAMPolicy,
)
from lerobot.policies.cae_fastwam.processor_cae_fastwam import make_cae_fastwam_pre_post_processors
from lerobot.policies.fastwam import modeling_fastwam
from lerobot.policies.fastwam.processor_fastwam import (
    FastWAMActionToggleProcessorStep,
    make_fastwam_pre_post_processors,
)
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

SUITES = ["suite_a", "suite_b"]


class ShapeVAE(nn.Module):
    z_dim = 4
    upsampling_factor = 2
    temporal_downsample_factor = 4

    def __init__(self):
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def encode(self, video, **kwargs):
        if isinstance(video, torch.Tensor):
            # Batched training videos [B, 3, T, H, W] -> float32 latents [B, 4, (T - 1) // 4 + 1, H / 2, W / 2],
            # like LeRobot's Wan VAE adapter (float32 output).
            self.seen.append(video.clone())
            return shape_latents(video)
        self.seen.extend(item.clone() for item in video)
        # A deterministic function of the pixels, so the image path reaches the experts.
        return [item.mean(dim=0, keepdim=True)[..., ::2, ::2].repeat(4, 1, 1, 1).float() for item in video]


def shape_latents(video: torch.Tensor) -> torch.Tensor:
    x = video.float()
    return torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)[:, :, ::4, ::2, ::2]


@pytest.fixture
def offline_components(monkeypatch):
    monkeypatch.setattr(modeling_fastwam, "load_pretrained_wan_vae", lambda **kw: ShapeVAE())
    monkeypatch.setattr(modeling_fastwam, "build_wan_tokenizer", lambda **kw: None)


def tiny_config(**overrides) -> CAEFastWAMConfig:
    common = {
        "hidden_dim": 12,
        "ffn_dim": 16,
        "freq_dim": 8,
        "text_dim": 8,
        "num_heads": 2,
        "attn_head_dim": 6,
        "num_layers": 2,
        "eps": 1e-6,
    }
    fields = {
        "action_dim": 3,
        "proprio_dim": 2,
        "action_horizon": 8,
        "n_action_steps": 4,
        "action_latent_tokens": 2,
        "action_latent_dim": 5,
        "action_decoder_hidden_dim": 8,
        "action_encoder_hidden_dim": 8,
        "task_suites": list(SUITES),
        "task_suite": "suite_b",
        "task_suite_stats": suite_stats(),
        "num_video_frames": 5,
        "action_video_freq_ratio": 1,
        "image_size": (16, 32),
        "context_len": 4,
        "load_text_encoder": False,
        "device": "cpu",
        "num_inference_steps": 2,
        "video_dit_config": dict(
            common,
            patch_size=[1, 2, 2],
            in_dim=4,
            out_dim=4,
            seperated_timestep=True,
            action_conditioned=False,
            video_attention_mask_mode="first_frame_causal",
        ),
        "action_dit_config": dict(common, action_dim=5, hidden_dim=4, ffn_dim=8),
        "input_features": {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    }
    fields.update(overrides)
    return CAEFastWAMConfig(**fields)


def suite_stats():
    return {
        "suite_a": {
            OBS_STATE: {"min": [-1.0, 0.0], "max": [1.0, 2.0]},
            ACTION: {"min": [-1.0, -2.0, 0.0], "max": [1.0, 2.0, 1.0]},
        },
        "suite_b": {
            OBS_STATE: {"min": [0.0, 0.5], "max": [4.0, 1.5]},
            ACTION: {"min": [-0.5, -0.25, 0.0], "max": [0.5, 0.75, 1.0]},
        },
    }


def inference_batch():
    return {
        "observation.images.image": torch.rand(1, 3, 16, 16),
        "observation.images.image2": torch.rand(1, 3, 16, 16),
        OBS_STATE: torch.rand(1, 2),
        "context": torch.randn(1, 4, 8),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def test_cae_fastwam_is_registered_and_publicly_exported():
    cfg = make_policy_config("cae_fastwam")
    assert isinstance(cfg, CAEFastWAMConfig)
    assert cfg.type == "cae_fastwam"
    assert "CAEFastWAMConfig" in lerobot_policies.__all__
    assert get_policy_class("cae_fastwam") is CAEFastWAMPolicy
    assert CAEFastWAMPolicy.name == "cae_fastwam"


def test_default_config_matches_the_released_checkpoint():
    cfg = CAEFastWAMConfig()
    assert cfg.action_dit_config["hidden_dim"] == 32
    assert cfg.action_dit_config["ffn_dim"] == 256
    assert cfg.action_dit_config["action_dim"] == cfg.action_latent_dim == 16
    assert (cfg.action_latent_tokens, cfg.action_horizon, cfg.action_dim) == (8, 32, 7)
    assert cfg.task_suites == ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    assert not cfg.fp32_attention
    assert cfg.task_suite_stats is None  # the checkpoint's config carries them
    assert cfg.toggle_action_dimensions == [-1]  # FastWAM's LIBERO gripper toggle
    assert cfg.input_features == {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    assert cfg.n_action_steps == 10
    assert cfg.action_encoder_hidden_dim == 128 and cfg.action_encoder_variational
    assert cfg.loss == {"lambda_video": 1.0, "lambda_action": 1.0} and not cfg.freeze_video_expert
    # Like FastWAM's `fastwam_base`, a type-only config initializes from the converted checkpoint ...
    assert cfg.pretrained_path == Path(CAE_FASTWAM_BASE_MODEL_ID)
    # ... unless its architecture differs (or `base_model_id` is unset).
    assert tiny_config().pretrained_path is None
    assert CAEFastWAMConfig(action_latent_tokens=4).pretrained_path is None
    assert CAEFastWAMConfig(base_model_id=None).pretrained_path is None


def test_config_validation_and_suite_resolution():
    with pytest.raises(ValueError, match="action_latent_dim"):
        tiny_config(action_latent_dim=7)
    with pytest.raises(ValueError, match="multiple of"):
        tiny_config(action_latent_tokens=3)
    with pytest.raises(ValueError, match="Unknown `task_suite`"):
        tiny_config(task_suite="libero_spatial")
    with pytest.raises(ValueError, match="relative"):
        tiny_config(use_relative_actions=True)
    with pytest.raises(ValueError, match="task_suite_stats"):
        tiny_config(task_suite_stats={"suite_a": suite_stats()["suite_a"]})
    cfg = tiny_config(task_suite=None)
    with pytest.raises(ValueError, match="task_suite"):
        cfg.suite_id()
    assert cfg.suite_id("suite_b") == 1
    assert tiny_config(task_suites=["only"], task_suite=None, task_suite_stats=None).suite_id() == 0


def test_config_round_trips_through_pretrained_config(tmp_path):
    cfg = tiny_config()
    cfg._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, CAEFastWAMConfig)
    for name in (
        "action_latent_tokens",
        "action_latent_dim",
        "task_suites",
        "task_suite",
        "task_suite_stats",
        "action_dit_config",
    ):
        assert getattr(loaded, name) == getattr(cfg, name)
    assert json.loads((tmp_path / "config.json").read_text())["type"] == "cae_fastwam"


def test_action_decoder_shapes_and_suite_conditioning():
    torch.manual_seed(0)
    decoder = CAEActionDecoder(
        action_dim=7, action_horizon=32, num_tokens=8, latent_dim=16, hidden_dim=128, num_suites=4
    )
    assert [type(m).__name__ for m in decoder.up][:3] == ["ConvTranspose1d", "GroupNorm", "SiLU"]
    assert decoder.up[0].stride == (4,) and decoder.up[0].kernel_size == (8,)
    assert decoder.up[0].padding == (2,)
    nn.init.normal_(decoder.suite_embed.weight)
    latents = torch.randn(2, 8, 16)
    out0 = decoder(latents, torch.tensor([0, 0]))
    out3 = decoder(latents, torch.tensor([3, 3]))
    assert out0.shape == (2, 32, 7)
    assert not torch.equal(out0, out3)
    with pytest.raises(ValueError, match="suite"):
        decoder(latents, None)
    with pytest.raises(ValueError, match="divisible"):
        CAEActionDecoder(
            action_dim=7, action_horizon=30, num_tokens=8, latent_dim=16, hidden_dim=8, num_suites=0
        )
    # Non-power-of-two ratios still decode to the full horizon.
    odd = CAEActionDecoder(
        action_dim=2, action_horizon=12, num_tokens=2, latent_dim=3, hidden_dim=8, num_suites=0
    )
    assert odd(torch.randn(1, 2, 3)).shape == (1, 12, 2)


# Output of the reference decoder (faster-action-models `fastwam/models/action_ae.py` `ActionDecoder`,
# Apache-2.0) for the seeded state dict and latents below, generated once on CPU (torch 2.13, float32);
# `CAEActionDecoder` matched it bitwise. Shape (2, 16, 3): suite 0 then suite 1.
_REFERENCE_DECODER_PARAM_ABS_SUM = 1534.68310546875
_REFERENCE_DECODER_OUTPUT = [
    [-1.90839148e00, -1.83430898e00, 3.23879862e00],
    [-9.35650289e-01, -1.79616475e00, 2.73460102e00],
    [-3.08170915e-01, -2.08229899e00, 3.19498968e00],
    [-1.19268584e00, -1.76155150e00, 2.50637388e00],
    [-1.41109347e-01, -2.04972911e00, 2.19530988e00],
    [-5.97501695e-01, -1.47777736e00, 4.00555801e00],
    [1.35098386e00, -2.69314671e00, 3.76284075e00],
    [-9.48170960e-01, -1.33142412e00, 3.81526089e00],
    [1.78750753e-01, -1.61557639e00, 2.53896856e00],
    [2.28212655e-01, -1.59377420e00, 3.14849043e00],
    [-6.80843234e-01, -1.82480788e00, 4.04810858e00],
    [-2.06070364e-01, -1.30236614e00, 2.17075801e00],
    [-2.76486933e-01, -2.18410897e00, 1.76856077e00],
    [-4.11633790e-01, -1.19885612e00, 2.43469334e00],
    [-1.10544443e-01, -1.39655244e00, 3.15578079e00],
    [-7.50144720e-02, -1.70174932e00, 2.93172336e00],
    [-8.08032453e-01, -1.89695704e00, 2.66465449e00],
    [-4.47260261e-01, -1.95297873e00, 2.43777418e00],
    [-6.59754217e-01, -1.56661415e00, 2.53046036e00],
    [-5.43493330e-01, -1.93925345e00, 2.46799302e00],
    [-1.11321902e00, -1.44803536e00, 3.26550055e00],
    [-1.54538274e-01, -2.13225818e00, 2.30999827e00],
    [-1.21566176e00, -2.45470262e00, 3.05113769e00],
    [-1.08508515e00, -2.08179331e00, 2.98569298e00],
    [-1.03242218e-01, -2.51654840e00, 3.79780912e00],
    [-4.83924389e-01, -1.66696417e00, 2.59422016e00],
    [-1.33794379e00, -2.44865012e00, 3.08853507e00],
    [-6.81371093e-01, -1.63181949e00, 2.65900993e00],
    [-6.27481759e-01, -1.74427438e00, 3.14086437e00],
    [-6.90664232e-01, -1.38068986e00, 2.07691312e00],
    [-9.47414339e-01, -2.21128297e00, 2.57005548e00],
    [-1.32948303e00, -1.64265287e00, 3.61819100e00],
]


def test_action_decoder_matches_reference_values():
    decoder = CAEActionDecoder(
        action_dim=3, action_horizon=16, num_tokens=4, latent_dim=5, hidden_dim=16, num_suites=2
    )
    generator = torch.Generator().manual_seed(1234)
    state = {
        key: torch.randn(value.shape, generator=generator) * 0.5
        for key, value in sorted(decoder.state_dict().items())
    }
    # Guards against a change of the RNG stream rather than of the decoder.
    assert float(sum(v.abs().sum() for v in state.values())) == pytest.approx(
        _REFERENCE_DECODER_PARAM_ABS_SUM, rel=1e-6
    )
    decoder.load_state_dict(state)
    latents = torch.randn((2, 4, 5), generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        out = decoder(latents, torch.tensor([0, 1]))
    expected = torch.tensor(_REFERENCE_DECODER_OUTPUT, dtype=torch.float32).reshape(2, 16, 3)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-5)


def test_processors_are_fastwams_with_the_selected_suite_statistics():
    cfg = tiny_config()
    pre, post = make_cae_fastwam_pre_post_processors(
        cfg, dataset_stats={OBS_STATE: {"min": [9.0, 9.0], "max": [9.5, 9.5]}}
    )
    ref_pre, ref_post = make_fastwam_pre_post_processors(cfg, suite_stats()["suite_b"])
    assert [type(s) for s in pre.steps] == [type(s) for s in ref_pre.steps]
    assert [type(s) for s in post.steps] == [type(s) for s in ref_post.steps]
    assert FastWAMActionToggleProcessorStep in [type(s) for s in post.steps]
    obs = {
        "observation.images.image": torch.rand(3, 20, 20),
        "observation.images.image2": torch.rand(3, 20, 20),
        OBS_STATE: torch.tensor([1.0, 0.75]),
    }
    batch = pre(obs)
    # No resize step: the model resizes each camera, as in FastWAM.
    assert batch["observation.images.image"].shape == (1, 3, 20, 20)
    # suite_b state ranges [0, 4] and [0.5, 1.5]: 2 * (x - min) / (max - min) - 1.
    torch.testing.assert_close(batch[OBS_STATE], torch.tensor([[-0.5, -0.5]]), rtol=0, atol=0)
    for pipeline, ref in ((pre, ref_pre), (post, ref_post)):
        for step, ref_step in zip(pipeline.steps, ref.steps, strict=True):
            if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                assert step.state_dict().keys() == ref_step.state_dict().keys()
                for key, value in step.state_dict().items():
                    torch.testing.assert_close(value, ref_step.state_dict()[key], rtol=0, atol=0)
    # Without a selected suite (several suites, no `task_suite`), the dataset statistics apply, as in FastWAM.
    pre_none, _ = make_cae_fastwam_pre_post_processors(
        tiny_config(task_suite=None), dataset_stats={OBS_STATE: {"min": [0.0, 0.0], "max": [2.0, 2.0]}}
    )
    torch.testing.assert_close(pre_none(obs)[OBS_STATE], torch.tensor([[0.0, -0.25]]), rtol=0, atol=0)


def test_processors_save_load_and_follow_the_policy_task_suite(tmp_path):
    cfg = tiny_config()
    pre, post = make_cae_fastwam_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    obs = {
        "observation.images.image": torch.rand(3, 20, 20),
        "observation.images.image2": torch.rand(3, 20, 20),
        OBS_STATE: torch.tensor([1.0, 0.5]),
    }
    pre_a, post_a = make_pre_post_processors(tiny_config(task_suite="suite_a"), pretrained_path=str(tmp_path))
    # suite_a state ranges [-1, 1] and [0, 2]: [1.0, 0.5] -> [1.0, -0.5].
    torch.testing.assert_close(pre_a(obs)[OBS_STATE], torch.tensor([[1.0, -0.5]]), rtol=0, atol=0)
    # suite_a action ranges [-1, 1], [-2, 2], [0, 1]; normalized -1 on the gripper is 0 (close) in the dataset
    # convention -> +1 (close) in LIBERO.
    out = post_a(torch.tensor([[0.5, -0.5, -1.0]]))
    torch.testing.assert_close(out, torch.tensor([[0.5, -1.0, 1.0]]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_policy_predicts_decoded_chunks_and_round_trips(offline_components, tmp_path, dtype):
    # bfloat16 on the CPU runs under CPU autocast, as in FastWAM's tests.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        torch.manual_seed(0)
        policy = CAEFastWAMPolicy(tiny_config(dtype=dtype)).eval()
        assert policy.model.action_decoder.input_proj.weight.dtype == torch.float32
        assert {p.dtype for p in policy.model.mot.parameters()} == {dtype}
        batch = inference_batch()
        noise = torch.randn(1, 2, 5)
        chunk = policy.predict_action_chunk(batch, noise=noise)
        assert chunk.shape == (1, 8, 3) and chunk.dtype == torch.float32
        assert torch.isfinite(chunk).all()
        # The decoded chunk is the action decoder applied to the denoised latents with the configured suite.
        out = policy.model.infer_action(
            prompt=None,
            input_image=torch.cat(
                [batch["observation.images.image"], batch["observation.images.image2"]], dim=-1
            ),
            action_horizon=8,
            proprio=batch[OBS_STATE],
            context=batch["context"],
            context_mask=batch["context_mask"],
            num_inference_steps=2,
            seed=42,
            noise=noise,
        )
        assert out["action_latents"].shape == (2, 5)
        torch.testing.assert_close(out["action"].unsqueeze(0), chunk, rtol=0, atol=0)
        expected = (
            policy.model.action_decoder(out["action_latents"].unsqueeze(0), torch.tensor([1]))
            .to(dtype)
            .float()
        )
        torch.testing.assert_close(chunk, expected, rtol=0, atol=0)
        # Changing the suite changes the decoded chunk (suite embedding).
        nn.init.normal_(policy.model.action_decoder.suite_embed.weight)
        policy.config.task_suite = "suite_a"
        assert not torch.equal(policy.predict_action_chunk(batch, noise=noise), chunk)
        policy.config.task_suite = "suite_b"
        chunk = policy.predict_action_chunk(batch, noise=noise)
        # select_action serves n_action_steps actions from one chunk.
        policy.reset()
        first = policy.select_action(batch, noise=noise)
        torch.testing.assert_close(first, chunk[:, 0], rtol=0, atol=0)
        assert len(policy._action_queue) == policy.config.n_action_steps - 1
        # Save and reload: same weights (decoder included, kept in float32), same chunk.
        policy.save_pretrained(tmp_path)
        loaded = CAEFastWAMPolicy.from_pretrained(tmp_path, strict=True).eval()
        assert isinstance(loaded.config, CAEFastWAMConfig)
        assert loaded.model.action_decoder.input_proj.weight.dtype == torch.float32
        torch.testing.assert_close(loaded.predict_action_chunk(batch, noise=noise), chunk, rtol=0, atol=0)
        assert loaded.model.action_encoder.input_proj.weight.dtype == torch.float32
        torch.testing.assert_close(
            loaded.model.action_encoder.bottleneck.weight,
            policy.model.action_encoder.bottleneck.weight,
            rtol=0,
            atol=0,
        )


def test_joint_video_inference_is_rejected(offline_components):
    policy = CAEFastWAMPolicy(tiny_config()).eval()
    with pytest.raises(NotImplementedError, match="infer_action"):
        policy.model.infer(prompt=None, input_image=torch.zeros(1, 3, 16, 32), num_frames=5)


# ---------------------------------------------------------------------------------------------------------
# Training (CAEFastWAMPolicy.forward): the original CAE-FastWAM objective.
# ---------------------------------------------------------------------------------------------------------


def seeded_state(module: nn.Module, scale: float = 0.5, seed: int = 1234) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        k: torch.randn(v.shape, generator=generator) * scale for k, v in sorted(module.state_dict().items())
    }


def seeded_policy(dtype=torch.float32, **overrides) -> CAEFastWAMPolicy:
    policy = CAEFastWAMPolicy(tiny_config(dtype=dtype, **overrides))
    policy.load_state_dict(seeded_state(policy, scale=0.2), strict=True)
    return policy.train()


def training_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """A preprocessed LeRobot training batch for `tiny_config`: 5 frames of two 16x16 cameras, an 8-step chunk
    (the last 3 steps of sample 1 past the episode end), per-frame state, precomputed text context."""
    generator = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (batch_size, 5, 3, 16, 32), generator=generator).float() / 255.0
    image_is_pad = torch.zeros(batch_size, 5, dtype=torch.bool)
    image_is_pad[-1, -1] = True
    action_is_pad = torch.zeros(batch_size, 8, dtype=torch.bool)
    action_is_pad[-1, -3:] = True
    return {
        "observation.images.image": frames[..., :16],
        "observation.images.image2": frames[..., 16:],
        "observation.images.image_is_pad": image_is_pad.clone(),
        "observation.images.image2_is_pad": image_is_pad.clone(),
        OBS_STATE: torch.rand(batch_size, 5, 2, generator=generator) * 2 - 1,
        ACTION: torch.rand(batch_size, 8, 3, generator=generator) * 2 - 1,
        "action_is_pad": action_is_pad,
        "context": torch.randn(batch_size, 4, 8, generator=generator),
        "context_mask": torch.ones(batch_size, 4, dtype=torch.bool),
    }


# Output of the reference encoder (faster-action-models `FastWAM/src/fastwam/models/action_ae.py` `ActionEncoder`,
# variational, suite-conditioned) for the seeded state dict, actions and posterior noise of the test below,
# generated once on CPU (torch 2.13, float32); `CAEActionEncoder` matched it bitwise. Shape (2, 4, 5).
_REFERENCE_ENCODER_PARAM_ABS_SUM = 1492.014404296875
_REFERENCE_ENCODER_Z = [
    [
        [-0.7856535, -0.26623422, -4.1491013, 0.26371264, 4.82255],
        [0.5109068, 0.008014932, -9.081219, -0.64961374, -5.0746827],
        [-0.5161705, 0.19162938, -0.24192381, 0.83144826, -2.6394813],
        [0.53565097, 0.6971637, 1.4349396, 0.43805876, -5.8461866],
    ],
    [
        [-0.4832631, -0.64714813, 0.39292616, -0.7584307, -0.37171578],
        [0.061041743, 0.19970462, -6.309304, -0.501715, -2.79772],
        [-1.279663, 0.104412794, -3.430715, 4.9260836, -1.8105285],
        [0.1709027, 2.2330909, 4.1038637, -0.39430737, -7.5609713],
    ],
]
_REFERENCE_ENCODER_MU = [
    [
        [-1.4786118, -0.43212792, -1.3688054, 1.2958817, -2.4231317],
        [0.6801879, 0.067861706, -1.7722995, -0.7726133, -5.003718],
        [0.3147728, 0.45138818, -2.242964, 1.548145, -2.0158973],
        [0.9881874, -0.40327182, -1.6739764, 0.6383927, -2.8879857],
    ],
    [
        [-0.23555237, -0.22316948, 0.069018304, -0.7120153, -3.4602191],
        [0.48444155, 0.09468368, -1.6939843, 0.70231587, -2.764577],
        [-1.3701425, 0.21497008, -1.898046, 3.6764984, -1.6033762],
        [0.04029187, 1.6199286, -1.8237588, 1.5469998, -3.860786],
    ],
]


def test_action_encoder_matches_reference_values():
    encoder = CAEActionEncoder(
        action_dim=3,
        action_horizon=16,
        num_tokens=4,
        latent_dim=5,
        hidden_dim=16,
        num_suites=2,
        variational=True,
    )
    assert [type(m).__name__ for m in encoder.down][:3] == ["Conv1d", "GroupNorm", "SiLU"]
    assert encoder.down[0].kernel_size == (7,) and encoder.down[0].stride == (4,)
    assert encoder.down[0].padding == (3,)
    assert encoder.bottleneck.out_features == 10  # mean and log-variance
    state = seeded_state(encoder)
    assert float(sum(v.abs().sum() for v in state.values())) == pytest.approx(
        _REFERENCE_ENCODER_PARAM_ABS_SUM, rel=1e-6
    )
    encoder.load_state_dict(state)
    actions = torch.randn((2, 16, 3), generator=torch.Generator().manual_seed(99))
    torch.manual_seed(5)
    with torch.no_grad():
        out = encoder(actions, torch.tensor([0, 1]))
    torch.testing.assert_close(out["mu"], torch.tensor(_REFERENCE_ENCODER_MU), rtol=0, atol=1e-5)
    torch.testing.assert_close(out["z"], torch.tensor(_REFERENCE_ENCODER_Z), rtol=0, atol=1e-4)
    # The latent is a posterior sample drawn from the global generator: z = mu + exp(log_var / 2) * eps.
    torch.manual_seed(5)
    eps = torch.randn_like(out["mu"])
    torch.testing.assert_close(out["z"], out["mu"] + (0.5 * out["log_var"]).exp() * eps, rtol=0, atol=0)
    with pytest.raises(ValueError, match="suite"):
        encoder(actions, None)
    with pytest.raises(ValueError, match="16-step"):
        encoder(actions[:, :8], torch.tensor([0, 1]))
    deterministic = CAEActionEncoder(
        action_dim=3,
        action_horizon=16,
        num_tokens=4,
        latent_dim=5,
        hidden_dim=16,
        num_suites=0,
        variational=False,
    )
    assert set(deterministic(actions)) == {"z"} and deterministic(actions)["z"].shape == (2, 4, 5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_inputs_are_the_encoded_chunk_with_pooled_padding(offline_components, dtype):
    # bfloat16 on the CPU runs under CPU autocast, as in FastWAM's tests.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        policy = seeded_policy(dtype=dtype)
        batch = training_batch()
        captured = {}
        sample_targets = policy.model._sample_training_targets

        def capture(inputs):
            captured.update(inputs)
            return sample_targets(inputs)

        policy.model._sample_training_targets = capture
        torch.manual_seed(3)
        policy.forward(batch)
        # The action targets are the frozen encoder's posterior sample of the (model-dtype) normalized chunk, with
        # the configured suite; the encoder is the first consumer of the global generator, as in the original.
        torch.manual_seed(3)
        expected = policy.model.action_encoder(batch[ACTION].to(dtype).float(), torch.tensor([1, 1]))["z"].to(
            dtype
        )
        torch.testing.assert_close(captured["action"], expected, rtol=0, atol=0)
        # 8 steps -> 2 tokens of 4 steps; a token is padding if any of its steps is.
        assert captured["action_is_pad"].tolist() == [[False, False], [False, True]]
        # The video side is FastWAM's: frames in the model dtype mapped to [-1, 1], no `image_is_pad` derived.
        frames = torch.cat([batch["observation.images.image"], batch["observation.images.image2"]], dim=-1)
        pixels = frames.permute(0, 2, 1, 3, 4).to(dtype) * 2.0 - 1.0
        torch.testing.assert_close(policy.model.vae.seen[-1], pixels, rtol=0, atol=0)
        assert captured["image_is_pad"] is None


def test_training_loss_is_deterministic_and_supports_per_sample_reduction(offline_components):
    policy = seeded_policy()
    torch.manual_seed(42)
    loss, logs = policy.forward(training_batch())
    assert torch.isfinite(loss) and set(logs) == {"loss_video", "loss_action"}
    torch.manual_seed(42)
    again, _ = policy.forward(training_batch())
    torch.testing.assert_close(again, loss, rtol=0, atol=0)
    torch.manual_seed(42)
    per_sample, _ = policy.forward(training_batch(), reduction="none")
    assert per_sample.shape == (2,)
    torch.testing.assert_close(per_sample.mean(), loss, rtol=1e-6, atol=1e-7)


def test_training_updates_only_the_mixture_of_transformers_and_proprio_encoder(offline_components):
    policy = seeded_policy()
    torch.manual_seed(42)
    loss, _ = policy.forward(training_batch())
    loss.backward()
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert {name.split(".")[1] for name in trainable} == {"mot", "proprio_encoder"}
    assert {name for name, p in policy.named_parameters() if p.grad is not None} == trainable
    assert {id(p) for p in policy.get_optim_params()} == {
        id(p) for p in policy.parameters() if p.requires_grad
    }
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    for module in (policy.model.action_encoder, policy.model.action_decoder):
        assert not module.training  # kept in eval mode by policy.train()
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    optimizer = policy.config.get_optimizer_preset()
    assert optimizer.betas == (0.9, 0.95) and optimizer.grad_clip_norm == 1.0 and optimizer.lr == 1e-4


def test_fine_tuning_keeps_the_checkpoint_suite_statistics(tmp_path):
    """`lerobot-train` sends (un)normalizer overrides with the dataset statistics; the selected suite's
    statistics win (the action VAE was trained with them), also for a config without statistics
    (`--policy.type=cae_fastwam`), which takes the pretrained checkpoint's."""
    cfg = tiny_config()
    cfg._save_pretrained(tmp_path)
    pre, post = make_cae_fastwam_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    dataset_stats = {OBS_STATE: {"min": torch.zeros(2), "max": torch.ones(2)}}
    pre2, post2 = make_pre_post_processors(
        cfg,
        pretrained_path=str(tmp_path),
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {"stats": dataset_stats},
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={"unnormalizer_processor": {"stats": dataset_stats}},
    )
    obs = {
        "observation.images.image": torch.rand(3, 16, 16),
        "observation.images.image2": torch.rand(3, 16, 16),
        OBS_STATE: torch.tensor([1.0, 0.5]),
    }
    torch.testing.assert_close(pre2(obs)[OBS_STATE], pre(obs)[OBS_STATE], rtol=0, atol=0)
    action = torch.tensor([[0.25, -0.5, 1.0]])
    torch.testing.assert_close(post2(action), post(action), rtol=0, atol=0)
    type_only = tiny_config(task_suite_stats=None)
    pre3, _ = make_pre_post_processors(
        type_only,
        pretrained_path=str(tmp_path),
        preprocessor_overrides={"normalizer_processor": {"stats": dataset_stats}},
    )
    assert type_only.task_suite_stats == suite_stats()
    torch.testing.assert_close(pre3(obs)[OBS_STATE], pre(obs)[OBS_STATE], rtol=0, atol=0)


def test_training_needs_a_task_suite(offline_components):
    policy = seeded_policy(task_suite=None)
    with pytest.raises(ValueError, match="task_suite"):
        policy.forward(training_batch())


def test_training_through_the_saved_preprocessor_uses_the_configured_suite(offline_components, tmp_path):
    """The `lerobot-train` path: the saved preprocessor normalizes with `task_suite`'s statistics and the
    encoder uses the same suite."""
    cfg = tiny_config()
    pre, post = make_cae_fastwam_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre2, _ = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    raw = training_batch()
    raw[ACTION] = raw[ACTION] * 0.25 + 0.25  # inside suite_b's action range
    processed = pre2(raw)
    # suite_b statistics: action dim 0 has min -0.5 / max 0.5, so x -> 2 x.
    torch.testing.assert_close(processed[ACTION][..., 0], 2.0 * raw[ACTION][..., 0], rtol=1e-6, atol=1e-6)
    # The tiny model takes a precomputed text context instead of `task`.
    processed.update(context=raw["context"], context_mask=raw["context_mask"])
    policy = seeded_policy()
    torch.manual_seed(42)
    loss, _ = policy.forward(processed)
    assert torch.isfinite(loss)
