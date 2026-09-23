# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Dependency-light configurations and fake encoders shared by FLUX3 tests."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.flux3 import Flux3Config
from lerobot.policies.flux3.f3.text_encoder import TEXT_PAD_MULTIPLE
from lerobot.utils.constants import ACTION, OBS_STATE

TINY_DIT = {
    "hidden_size": 64,
    "num_heads": 2,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [8, 8, 8, 8],
    "context_in_dim": 32,
}
DROID_CAMS = [
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
]

# Explicit reference settings keep frame/DROID coverage independent of PEFT class defaults.
FRAME_REFERENCE_SETTINGS = {
    "conditioning": "frame",
    "condition_on_past_actions": False,
    "action_representation": "absolute",
    "text_fixed_length": None,
    "video_position_fps": None,
    "history_snapshots": 1,
    "separate_timesteps": False,
    "conditioning_noise_max": 0.0,
    "loss_reduction": "joint_tokens",
    "gradient_checkpointing": False,
    "n_obs_steps": 1,
    "n_action_steps": 32,
    "fps": 15.0,
    "gripper_flip_dims": [-1],
    "action_loss_weight": 50.0,
    "augment": True,
    "optimizer_lr": 1.92e-4,
    "optimizer_betas": (0.9, 0.99),
    "optimizer_weight_decay": 0.05,
    "optimizer_grad_clip_norm": 0.0,
    "scheduler_freeze_backbone_steps": 1000,
    "scheduler_warmup_steps": 2000,
    "scheduler_warmup_steps_heads": 1000,
}


class MockTextEncoder(nn.Module):
    """Deterministic pseudo-random context per caption, without pretrained weights."""

    def __init__(self, context_in_dim: int):
        super().__init__()
        self.context_in_dim = context_in_dim

    @torch.inference_mode()
    def forward_bucketed(self, text: str, *, fixed_length: int | None = None) -> torch.Tensor:
        # Different captions produce distinct contexts so tests exercise classifier-free guidance.
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(1, fixed_length or TEXT_PAD_MULTIPLE, self.context_in_dim, generator=gen).to(
            torch.bfloat16
        )


class FakeVideoVAE:
    """Shape-faithful stand-in for the Video VAE: /32 spatial, /4 temporal, 96 channels, deterministic."""

    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        self.proj = torch.randn(96, 3, generator=gen) * 0.1
        self.module = nn.Identity()  # what the policy moves along with `_apply`

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video.shape
        assert (t - 1) % 4 == 0, t
        x = video.float()[:, :, ::4]  # (B, 3, t', H, W), t' = 1 + (t - 1) // 4
        tp = x.shape[2]
        x = F.avg_pool2d(x.transpose(1, 2).reshape(-1, c, h, w), 32)  # (B*t', 3, H/32, W/32)
        x = torch.einsum("nchw,oc->nohw", x, self.proj)
        return x.reshape(b, tp, 96, h // 32, w // 32).transpose(1, 2).to(torch.bfloat16)


def droid_features():
    inputs = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)) for k in DROID_CAMS}
    inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return inputs, outputs


def droid_config(**overrides) -> Flux3Config:
    inputs, outputs = droid_features()
    kwargs = {
        **FRAME_REFERENCE_SETTINGS,
        "input_features": inputs,
        "output_features": outputs,
        "camera_keys": DROID_CAMS,
        "camera_layout": "droid",
        "canvas_hw": (544, 736),
        "fps": 15.0,
        "dit_config": TINY_DIT,
        "dtype": "float32",
        "video_vae_id": None,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return Flux3Config(**kwargs)


def single_config(**overrides) -> Flux3Config:
    inputs = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    kwargs = {
        **FRAME_REFERENCE_SETTINGS,
        "input_features": inputs,
        "output_features": outputs,
        "camera_layout": "single",
        "canvas_hw": (64, 96),
        "gripper_flip_dims": [],
        "dit_config": TINY_DIT,
        "dtype": "float32",
        "video_vae_id": None,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return Flux3Config(**kwargs)


class TaskVideoVAE(FakeVideoVAE):
    def encode_task(self, video):
        # Native VAE floor temporal reduction; unlike DROID, no externally padded tail.
        return self.encode(video[:, :, : 1 + 4 * ((video.shape[2] - 1) // 4)])


def task_config(**overrides):
    # Exercise native PEFT defaults with a tiny model and one small camera.
    kwargs = {
        "input_features": {
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "camera_layout": "single",
        "canvas_hw": (64, 96),
        "dit_config": TINY_DIT,
        "dtype": "float32",
        "video_vae_id": None,
        "device": "cpu",
        "delta_absolute_dims": [-1],
        "action_channel_weights": [1, 1, 1, 1, 1, 2],
        "caption_dropout": 0,
        "normalization_stats": {name: {"q01": [-2.0] * 6, "q99": [2.0] * 6} for name in ("action", "state")},
    }
    kwargs.update(overrides)
    return Flux3Config(**kwargs)


def rich_task_config(**overrides):
    """Explicit multi-timestep coverage, independent of the single-timestep PEFT defaults."""
    return task_config(
        **{"n_obs_steps": 8, "history_snapshots": 2, "condition_on_past_actions": True, **overrides}
    )
