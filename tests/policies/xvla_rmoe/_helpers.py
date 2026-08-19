#!/usr/bin/env python

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

"""Shared tiny-model fixtures for the `xvla_rmoe` test suite.

Unlike `smolvla_rmoe` (whose tiny-policy tests need CUDA and a Hub config fetch, since its
VLM is a real pretrained architecture family), X-VLA's Florence-2 config is fully
self-contained -- `Florence2Config(vision_config=..., text_config=...)` builds a real (tiny)
model from plain dicts, no network access or GPU required. So every test in this suite can
instantiate a real, tiny, from-scratch `XVLARMoEPolicy` and run genuine forward/backward on
CPU: nothing here is a mock.

The vision config is deliberately not the smallest possible: DaViT downsamples by 16x
overall, and a 1x1 output feature map hits a degenerate edge case in Florence-2's learned 2D
position embedding (produces NaN). 64x64 images give a 4x4 feature map, avoiding that.
"""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xvla_rmoe.configuration_xvla_rmoe import XVLARMoEConfig
from lerobot.policies.xvla_rmoe.modeling_xvla_rmoe import XVLARMoEPolicy
from lerobot.policies.xvla_rmoe.moe_soft_transformer import MoEFFN
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_STATE

# A bare `grad.abs().sum() > 0` is not a safe test: pure floating-point roundoff noise around
# a degenerate zero-gradient point can itself be nonzero. See `expert_symmetry_breaking_std`
# in configuration_xvla_rmoe.py for why that degeneracy exists (bit-identical experts + a
# zero-init router is a permutation-symmetry fixed point) and how it's avoided. Gradient-flow
# assertions use this floor, comfortably above that noise band, mirroring `smolvla_rmoe`'s own
# `_GRAD_NOISE_FLOOR`.
GRAD_NOISE_FLOOR = 1e-6

IMAGE_SIZE = 64
STATE_DIM = 14
ACTION_DIM = 14

VISION_CONFIG = {
    "dim_embed": [32, 32, 32, 32],
    "num_heads": [4, 4, 4, 4],
    "num_groups": [4, 4, 4, 4],
    "depths": [1, 1, 1, 1],
    "patch_size": [7, 3, 3, 3],
    "patch_stride": [4, 2, 2, 2],
    "patch_padding": [3, 1, 1, 1],
    "patch_prenorm": [False, True, True, True],
    "window_size": 4,
    "projection_dim": 16,
}

TEXT_CONFIG = {
    "vocab_size": 64,
    "d_model": 16,
    "encoder_layers": 1,
    "encoder_ffn_dim": 32,
    "encoder_attention_heads": 2,
    "decoder_layers": 1,
    "decoder_ffn_dim": 32,
    "decoder_attention_heads": 2,
    "max_position_embeddings": 64,
}


def make_tiny_config(**overrides) -> XVLARMoEConfig:
    kwargs = {
        "florence_config": {
            "vision_config": VISION_CONFIG,
            "text_config": TEXT_CONFIG,
            "projection_dim": 16,
            "pad_token_id": 1,
        },
        "hidden_size": 32,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "num_domains": 2,
        "len_soft_prompts": 2,
        "dim_time": 8,
        "max_len_seq": 128,
        "chunk_size": 6,
        "n_action_steps": 6,
        "num_denoising_steps": 6,
        "action_mode": "joint",
        "max_state_dim": STATE_DIM,
        "use_proprio": True,
        "num_moe_experts": 3,
        "num_moe_layers": 2,
        "routing_hidden_dim": 8,
        "routing_timestep_dim": 8,
        "chunk_pos_emb_dim": 4,
        "recurrent_unroll_steps": 3,
        "recurrent_training_probability": 0.25,
        "tokenizer_max_length": 6,
        "resize_imgs_with_padding": (IMAGE_SIZE, IMAGE_SIZE),
        "num_image_views": 1,
        "device": "cpu",
    }
    kwargs.update(overrides)
    cfg = XVLARMoEConfig(**kwargs)
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return cfg


def construct_until_finite(build_fn, max_attempts: int = 8):
    """Call `build_fn()` (expected to return a freshly constructed X-VLA/X-VLA-RMoE policy),
    retrying if its vision tower lands in a numerically unstable region.

    This is a documented, pre-existing CPU-determinism limitation of `nn.Conv2d` (see
    https://pytorch.org/docs/stable/notes/randomness.html): the algorithm it picks on CPU is
    not guaranteed reproducible even under `torch.manual_seed` + single-threading, and
    Florence-2's DaViT vision tower (`self.vlm._encode_image`, wholly unmodified upstream
    code -- present verbatim in both `xvla` and `xvla_rmoe`) leans on it. At the extreme
    dimensions a fast unit-test config needs (32 channels vs. the real ~256-2048), a small
    fraction of otherwise-identical constructions land on a numerically unstable branch and
    produce NaN from the very first vision forward pass, before any policy-specific code runs
    at all. Retrying (each attempt draws different weights: the global RNG stream has already
    moved on from the previous attempt, no explicit reseeding needed) until a construction
    whose vision tower is well-behaved on a fixed probe image is found lets tests exercise the
    actual logic under test instead of intermittently tripping over this unrelated, upstream
    fragility. Every `xvla_rmoe` test-suite policy construction goes through this.
    """
    probe_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    policy = None
    for _attempt in range(max_attempts):
        policy = build_fn()
        with torch.no_grad():
            target_dtype = policy.model._get_target_dtype()
            probe_out = policy.model.vlm.get_image_features(probe_image.to(dtype=target_dtype)).pooler_output
            if not torch.isfinite(probe_out).all():
                continue
            # The vision tower alone being finite doesn't guarantee the full pipeline is: probe
            # a real single-t forward pass (the cheapest full path) too, in eval mode so it
            # can't accidentally consume `_train_call_counter` / pick the recurrent branch.
            was_training = policy.training
            policy.eval()
            probe_batch = dict(make_tiny_batch(policy))
            loss, _log = policy.forward(dict(probe_batch))
            if not torch.isfinite(loss).all():
                policy.train(was_training)
                continue
            # Also probe the multi-step denoising loop (`predict_action_chunk`): the risk
            # surface a single forward pass above doesn't cover (several Euler-like iterations
            # can compound instability that one pass alone would not show).
            actions = policy.predict_action_chunk(probe_batch)
            policy.train(was_training)
        if torch.isfinite(actions).all():
            policy.reset()  # undo the probe's queue mutations before handing the policy back
            return policy
    return policy  # exhausted retries: let the caller's own assertions surface it


def make_tiny_policy(**config_overrides) -> XVLARMoEPolicy:
    cfg = make_tiny_config(**config_overrides)
    return construct_until_finite(lambda: XVLARMoEPolicy(cfg))


def make_tiny_batch(policy: XVLARMoEPolicy, bsize: int = 2, pad_last_n: int = 0) -> dict[str, torch.Tensor]:
    cfg = policy.config
    action_is_pad = torch.zeros(bsize, cfg.chunk_size, dtype=torch.bool)
    if pad_last_n:
        action_is_pad[:, -pad_last_n:] = True
    return {
        OBS_STATE: torch.randn(bsize, STATE_DIM),
        f"{OBS_IMAGES}.cam": torch.rand(bsize, 3, IMAGE_SIZE, IMAGE_SIZE),
        OBS_LANGUAGE_TOKENS: torch.randint(0, TEXT_CONFIG["vocab_size"], (bsize, cfg.tokenizer_max_length)),
        ACTION: torch.randn(bsize, cfg.chunk_size, ACTION_DIM),
        "action_is_pad": action_is_pad,
        "domain_id": torch.zeros(bsize, dtype=torch.long),
    }


def perturb_router_weights(policy: XVLARMoEPolicy, std: float = 1.0) -> None:
    """MoEFFN routers are zero-initialized by design (uniform routing == original FFN output
    at step 0). At that exact point `d(logits)/d(routing_state) = router.weight = 0`, so no
    gradient can reach the GRU yet -- mirrors `smolvla_rmoe`'s own `_perturb_router_weights`
    and the reasoning documented there. Tests that check gradient flow into the GRU perturb
    the router first."""
    for module in policy.model.transformer.blocks:
        if isinstance(module.mlp, MoEFFN):
            torch.nn.init.normal_(module.mlp.router.weight, std=std)


def disable_dropout(policy: XVLARMoEPolicy) -> None:
    """Zero every `nn.Dropout` probability in the model, without touching `.training` /
    `self.training` (needed by `XVLARMoEPolicy._should_use_recurrent_step`, which gates on it).

    X-VLA's own `TransformerBlock` hardcodes non-zero dropout (`attn_drop=0.1`, `Mlp(...,
    drop=0.1)`), unlike `smolvla_rmoe`'s (dropout-free) gated-MLP expert. In train() mode this
    means every `MoEFFN` expert -- even bit-identical deep copies at
    `expert_symmetry_breaking_std=0` -- independently samples its own dropout mask and so
    stops being a bit-identical *function*, even though its *weights* still are. That is a
    real, separate source of router/GRU gradient, unrelated to (and would otherwise
    confound) what `expert_symmetry_breaking_std` is specifically responsible for. Gradient-
    floor and symmetry-trap tests call this after construction so the only thing that can
    make experts diverge is the mechanism actually under test.
    """
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
