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

"""A small random praxis checkpoint config for CPU tests, plus fake frozen parts.

The config has the layout of the single-arm checkpoint (chunk-causal backbone,
trunk-rope expert, chunked MoE dispatch, blank caption) with small dims; it is
the ``tiny_config`` of the praxis repository plus the ``video``, ``chunking``
and ``normalization`` sections the LeRobot adapter reads. The fake autoencoder
and text encoder are deterministic functions of their inputs, so two code paths
that feed them the same bytes get the same tensors.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.praxis.configuration_praxis import PRAXIS_ACTION_DIM, PraxisConfig
from lerobot.policies.praxis.praxis_core.backbone import TEXT, VIDEO
from lerobot.policies.praxis.praxis_core.checkpoint import EXPERT_TRUNK_ROPE, LAYOUT_STAMP_CHUNK_CAUSAL
from lerobot.policies.praxis.praxis_core.model import PraxisModel
from lerobot.utils.constants import ACTION, OBS_STATE

TEXT_DIM = 32
LATENT_CHANNELS = 192
TILE = (32, 64)  # one tile (h, w); the canvas is two tiles side by side -> latent grid 1 x 4

if TYPE_CHECKING:
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy


def tiny_praxis_config() -> dict:
    """A praxis ``config.json`` with small dims. The dims are multiples of 8, so the grouped GEMM runs on the CPU."""
    attention = {"num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "flex_block_size": 128}
    moe = {"num_routed_experts": 4, "top_k_routed": 2, "expert_hidden": 32, "routed_scaling_factor": 2.5}
    moe["dispatch_chunk_size"] = 40
    projections = {TEXT: {"c_in": TEXT_DIM}, VIDEO: {"c_in": LATENT_CHANNELS}}
    backbone = {
        "layout": "chunk_causal",
        "dim": 64,
        "timestep_dim": 16,
        "num_registers": 4,
        "modulation_dtype": "mixed",
        "dense_ffn_hidden": 96,
        "attention": attention,
        "rope": {
            "effective_dim": 8,
            "base": 26_000_000,
            "efficient_cache_max": 10**6,
            "ref_placement": "group",
        },
        "moe": moe,
        "encoder": {
            "depth": 3,
            "dense_blocks": [0],
            "tap_blocks": [0, 2],
            "input_projections": projections,
            "positive_zero_residual_add": True,
        },
        "decoder": {
            "depth": 2,
            "dense_blocks": [0],
            "modulation_dim": 64,
            "unpatch_factor": [1, 2, 2],
            "input_projections": {VIDEO: {"c_in": 48}},
            "positive_zero_residual_add": True,
        },
    }
    expert = {
        "dim": 64,
        "num_layers": 4,
        "num_heads": 1,
        "head_dim": 64,
        "trunk_dim": 64,
        "action_patch_dim": 40,
        "proprio_patch_dim": 10,
        "num_future_per_chunk": 2,
        "max_chunks": 16,
        "max_action_tokens": 64,
        "tap_norm": True,
        "clean_only_conditioning": False,
        "rope_cross": False,
        "rope_self": False,
        "rope_base": 100.0,
        "absolute_positions": True,
        "action_pos_table": False,
        "future_chunk_offset": "zero",
        "rope": dict(EXPERT_TRUNK_ROPE),
        "global_order": ["register", "anchor", "text"],
        "attention": "sdpa",
        "head_plus_zero": True,
        "layout_stamp": LAYOUT_STAMP_CHUNK_CAUSAL,
    }
    q01 = [-0.15, -0.2, -0.2, 0.68, -0.58, -0.39, -0.57, 0.69, -0.4, 0.0]
    q99 = [0.19, 0.2, 0.2, 1.0, 0.58, 0.45, 0.59, 1.0, 0.4, 1.0]
    return {
        "format_version": 2,
        "model_type": "praxis",
        "embodiment": "single_arm",
        "torch_dtype": "bfloat16",
        "backbone": backbone,
        "action_expert": expert,
        "action_space": {"dim": 10},
        "chunking": {"n_chunks": 4, "latents_per_chunk": 6, "action_chunk": 24, "grid_hz": 15.0},
        "video": {"canvas": "tiles", "tile_height": TILE[0], "tile_width": TILE[1], "aspect_tol": 0.03},
        "text_encoder": {"uncond": "blank_caption", "blank_caption": " "},
        "normalization": {
            "action_q01": q01,
            "action_q99": q99,
            "q99_normed_dims": [0, 1, 2, 9],
            "proprio_mean": [0.52, 0.0, 0.32, 0.74, -0.05, 0.09, -0.06, -0.76, 0.01, 0.39],
            "proprio_std": [0.12, 0.18, 0.17, 0.33, 0.48, 0.32, 0.49, 0.34, 0.25, 0.41],
        },
        "pose_frame": "flange",
    }


def tiny_lerobot_config(image_shape: tuple[int, int, int] = (3, 36, 64), **overrides) -> PraxisConfig:
    """A ``PraxisConfig`` on the CPU with the tiny praxis config and the single-arm features."""
    kwargs = {
        "text_encoder_id": "",  # no tokenizer step: the fake text encoder takes the task string
        "model_config": tiny_praxis_config(),
        "input_features": {
            "observation.images.exterior": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(PRAXIS_ACTION_DIM,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(PRAXIS_ACTION_DIM,))},
        "device": "cpu",
        "dtype": torch.bfloat16,
        "attention_backend": "eager",
    }
    kwargs.update(overrides)
    return PraxisConfig(**kwargs)


def randomize_(model: PraxisModel, seed: int = 0) -> PraxisModel:
    """Give ``model`` random bf16 weights. The routing biases stay float32."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator, dtype=torch.float32) * 0.05)
        for name, buf in model.named_buffers():
            if name.endswith("expert_bias"):
                buf.copy_(torch.randn(buf.shape, generator=generator) * 0.01)
            elif name.endswith(("freqs", "phases")):
                buf.copy_(torch.rand(buf.shape, generator=generator) * 2 - 1)
    return model


class FakeAutoencoder:
    """A deterministic stand-in for the Wan2.2 VAE: average pools frames to the latent grid.

    ``encode`` maps ``(1, 3, T, H, W)`` frames in [0, 255] to a bf16 policy latent
    ``(1, 192, 1 + (T - 1) // 4, H // 32, W // 32)``: the first frame alone, then
    every 4 frames, pooled in space and repeated over the channels with a fixed
    per-channel scale, so different inputs give different latents.
    """

    def __init__(self) -> None:
        self.scale = torch.linspace(0.5, 1.5, LATENT_CHANNELS).view(1, -1, 1, 1, 1)
        # the training objective reads the device of the real autoencoder's buffers
        self.latent_mean = torch.zeros(LATENT_CHANNELS)

    def to(self, device) -> FakeAutoencoder:
        return self

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.float() / 255.0
        _, _, t, h, w = x.shape
        groups = [x[:, :, :1]] + [x[:, :, 1 + 4 * i : 5 + 4 * i] for i in range((t - 1) // 4)]
        pooled = torch.stack([g.mean(dim=2) for g in groups], dim=2)  # (1, 3, L, H, W)
        pooled = torch.nn.functional.avg_pool2d(pooled.flatten(1, 2), 32).view(1, 3, -1, h // 32, w // 32)
        latent = pooled.repeat(1, LATENT_CHANNELS // 3, 1, 1, 1) * self.scale
        return (latent * 2 - 1).to(torch.bfloat16)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = latent.shape
        frames = 1 + 4 * (t - 1)
        return torch.zeros(frames, 3, h * 32, w * 32, dtype=torch.uint8)


class FakeTextEncoder:
    """A deterministic stand-in for Qwen3.5-2B: tokens from the hash of the prompt."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def encode_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token ids stand for the prompt's words: the embedding of ``" ".join(ids)``."""
        return self.encode([" ".join(str(int(i)) for i in token_ids.reshape(-1).tolist())])[0]

    def encode(self, prompts: list[str]) -> list[torch.Tensor]:
        out = []
        for prompt in prompts:
            text = prompt if prompt.strip() else " "
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            g = torch.Generator().manual_seed(seed)
            n_tokens = 1 if text == " " else 2 + len(text.split()) % 3
            out.append(torch.randn(1, n_tokens, TEXT_DIM, generator=g).to(torch.bfloat16))
        return out


def fake_observation(step: int, seed: int = 0, image_shape: tuple[int, int, int] = (3, 36, 64)) -> dict:
    """One LeRobot observation batch of one environment: random cameras, a slowly moving pose."""
    rng = np.random.default_rng(seed * 100_003 + step)
    exterior = torch.from_numpy(rng.integers(0, 256, size=image_shape, dtype=np.uint8)).float() / 255
    wrist = torch.from_numpy(rng.integers(0, 256, size=image_shape, dtype=np.uint8)).float() / 255
    state = np.array([0.5 + 0.001 * step, 0.0, 0.4 - 0.0005 * step, np.pi, 0.1, 0.2, 0.0], dtype=np.float32)
    return {
        "observation.images.exterior": exterior[None],
        "observation.images.wrist": wrist[None],
        OBS_STATE: torch.from_numpy(state)[None],
        "task": ["Pick up the red cube."],
    }


def make_tiny_policy(seed: int = 0, **overrides) -> PraxisPolicy:
    """A random tiny policy on the CPU with the fake autoencoder and text encoder."""
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy

    config = tiny_lerobot_config(**overrides)
    policy = PraxisPolicy(config)
    randomize_(policy.model, seed)
    policy.vae = FakeAutoencoder()
    policy.text_encoder = FakeTextEncoder()
    return policy.eval()


BIMANUAL_IMAGE_SHAPE = (3, 40, 80)  # fits the tile (32, 64) through the YAM fit rule: no resize, a 32 px trim
BIMANUAL_CAMERA_KEYS = (
    "observation.images.top_left",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


def tiny_bimanual_praxis_config() -> dict:
    """The tiny config as a bimanual (YAM-like) checkpoint: three views, two arms, 20-D actions."""
    config = tiny_praxis_config()
    config["embodiment"] = "bimanual"
    config["pose_frame"] = "rig"
    config["action_space"] = {"dim": 20}
    config["action_expert"]["action_patch_dim"] = 80
    config["action_expert"]["proprio_patch_dim"] = 20
    config["video"] = {
        "canvas": "tiles",
        "views": ["top_left", "wrist_left", "wrist_right"],
        "tile_height": TILE[0],
        "tile_width": TILE[1],
        "height": TILE[0],
        "width": 3 * TILE[1],
        "aspect_tol": 0.03,
        "antialias": True,
    }
    norm = config["normalization"]
    config["normalization"] = {
        "action_q01": norm["action_q01"] * 2,
        "action_q99": norm["action_q99"] * 2,
        "q99_normed_dims": [0, 1, 2, 9, 10, 11, 12, 19],
        "proprio_mean": norm["proprio_mean"] * 2,
        "proprio_std": norm["proprio_std"] * 2,
    }
    return config


def tiny_bimanual_lerobot_config(
    image_shape: tuple[int, int, int] = BIMANUAL_IMAGE_SHAPE, **overrides
) -> PraxisConfig:
    """A bimanual ``PraxisConfig`` on the CPU: three cameras, the 14-value state and action."""
    kwargs = {
        "text_encoder_id": "",
        "model_config": tiny_bimanual_praxis_config(),
        "input_features": {
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape) for key in BIMANUAL_CAMERA_KEYS
            },
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2 * PRAXIS_ACTION_DIM,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2 * PRAXIS_ACTION_DIM,))},
        "device": "cpu",
        "dtype": torch.bfloat16,
        "attention_backend": "eager",
        "t_shift": 1.0,
        "cfg_video_scale": 1.0,
    }
    kwargs.update(overrides)
    return PraxisConfig(**kwargs)


def fake_bimanual_observation(
    step: int, seed: int = 0, image_shape: tuple[int, int, int] = BIMANUAL_IMAGE_SHAPE
) -> dict:
    """One LeRobot observation of the bimanual rig: three random views, two slowly moving arms."""
    rng = np.random.default_rng(seed * 100_003 + step)
    obs: dict = {}
    for key in BIMANUAL_CAMERA_KEYS:
        obs[key] = (torch.from_numpy(rng.integers(0, 256, size=image_shape, dtype=np.uint8)).float() / 255)[
            None
        ]
    left = np.array(
        [0.4 + 0.001 * step, 0.3, 0.4 - 0.0005 * step, np.pi - 0.1, 0.05, 0.2, 0.1], dtype=np.float32
    )
    right = np.array(
        [0.4 + 0.0008 * step, -0.3, 0.4 + 0.0004 * step, np.pi - 0.05, -0.05, -0.2, 0.9], dtype=np.float32
    )
    obs[OBS_STATE] = torch.from_numpy(np.concatenate([left, right]))[None]
    obs["task"] = ["Pick up the pen from the center of the table, then place the pen in the bag"]
    return obs


def make_tiny_bimanual_policy(seed: int = 0, **overrides) -> PraxisPolicy:
    """A random tiny bimanual policy on the CPU with the fake autoencoder and text encoder."""
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy

    policy = PraxisPolicy(tiny_bimanual_lerobot_config(**overrides))
    randomize_(policy.model, seed)
    policy.vae = FakeAutoencoder()
    policy.text_encoder = FakeTextEncoder()
    return policy.eval()
