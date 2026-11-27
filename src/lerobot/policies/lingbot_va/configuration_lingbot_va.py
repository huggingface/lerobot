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

"""Configuration for the LingBot-VA policy.

LingBot-VA is an autoregressive video-action world-model policy built on the Wan2.2
video-diffusion stack. It interleaves prediction of future video latents and robot
actions in a single dual-stream transformer. See ``docs/source/lingbot_va.mdx`` and the
upstream repository (https://github.com/Robbyant/lingbot-va).

Defaults below match the upstream LIBERO configuration (``wan_va/configs/va_libero_cfg.py``)
and the ``transformer/config.json`` of the released checkpoints.
"""

from dataclasses import dataclass, field

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.sampling import FrameSampler, TrainingSampler
from lerobot.configs.types import Deltas, FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_PREFIX, PREDICTION_PREFIX


@PreTrainedConfig.register_subclass("lingbot_va")
@dataclass
class LingBotVAConfig(PreTrainedConfig):
    """Configuration for the native LingBot-VA policy integration in LeRobot."""

    # Wan transformer architecture
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    # "flex" = training only (needs recent torch); inference uses "torch" SDPA or "flashattn".
    attn_mode: str = "torch"

    # Frozen sub-models (VAE + UMT5 text encoder + tokenizer)
    # ~20 GB of frozen weights, NOT bundled in the checkpoint; lazily pulled from this HF repo /
    # local dir (must hold diffusers-style ``vae/``, ``text_encoder/``, ``tokenizer/`` sub-folders).
    wan_pretrained_path: str = "robbyant/lingbot-va-base"
    # transformer / VAE / text-encoder dtype
    dtype: torch.dtype | None = torch.bfloat16
    # Frozen UMT5-XXL encoder device; "cpu" frees ~11 GB VRAM (it runs once per episode).
    text_encoder_device: str = "cpu"

    # Observation cameras (order matters: latents are concatenated on width; LIBERO defaults)
    obs_cam_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    # Undo the LIBERO env processor's extra horizontal flip to match the model's training orientation.
    image_hflip: bool = False
    # Camera latent layout: "width_concat" (cameras concatenated on width; LIBERO) or
    # "robotwin_tshape" (full-res head + half-res wrists in a "T"; RoboTwin).
    camera_layout: str = "width_concat"

    # Inference hyperparameters (LIBERO defaults)
    n_obs_steps: int = 1
    height: int = 128
    width: int = 128
    action_per_frame: int = 4
    frame_chunk_size: int = 4
    attn_window: int = 30
    num_inference_steps: int = 20
    video_exec_step: int = -1
    action_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    max_sequence_length: int = 512  # UMT5 prompt length

    # Subset of the 30-d action space used by the benchmark (LIBERO = 7-DoF). The action
    # (un)normalization quantiles live in the checkpoint's ``policy_postprocessor.json``, not here.
    used_action_channel_ids: list[int] = field(default_factory=lambda: list(range(7)))

    # Removed: predicted video comes back with the chunk (``return_predicted_video``). Declared only so
    # configs written before still parse (draccus rejects keys the dataclass no longer declares);
    # setting it raises.
    save_predicted_video: bool = False
    # Opt-in: VAE-decode each chunk's predicted video and return it per camera as
    # ``prediction.images.<camera>``, its frames spread over the chunk's actions, shown live by
    # ``--display_data``. Costs one VAE decode per chunk.
    return_predicted_video: bool = False

    # Normalization: IDENTITY here; images are scaled + VAE-encoded and actions are
    # quantile-(un)normalized inside the policy / dedicated processor steps.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Optimizer / scheduler (training; AdamW + warmup-constant per upstream train.py)
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.save_predicted_video:
            raise ValueError(
                "`save_predicted_video` was removed: use `--policy.return_predicted_video=true` with "
                "`--display_data=true` to show the predicted video next to each camera."
            )
        if self.attn_mode not in ("torch", "flashattn", "flex"):
            raise ValueError(f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}")
        if self.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError(
                f"Unsupported dtype={self.dtype!r}. Expected torch.bfloat16, torch.float16 or torch.float32."
            )
        if self.action_per_frame % self.frame_stride:
            raise ValueError("action_per_frame must be a multiple of max(1, action_per_frame // 4)")

    @property
    def chunk_size(self) -> int:
        """Number of single-step actions produced per autoregressive chunk."""
        return self.frame_chunk_size * self.action_per_frame

    @property
    def frame_stride(self) -> int:
        """Actions between two camera frames: the Wan2.2 VAE (temporal downsample 4) encodes
        ``frame_chunk_size`` latent frames from the frames of a chunk."""
        return max(1, self.action_per_frame // 4)

    @property
    def n_action_steps(self) -> int:
        """Number of actions executed before refilling (the whole chunk)."""
        return self.chunk_size

    def validate_features(self) -> None:
        if self.input_features is None or self.output_features is None:
            raise ValueError(
                "LingBot-VA requires `input_features` and `output_features` to be resolved before validation."
            )
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "LingBot-VA requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.used_action_channel_ids),)
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        # Upstream uses a linear warmup followed by a constant LR (warmup_constant_lambda).
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def training_sampler(self) -> TrainingSampler:
        # Future frames are video targets (see ``frame_stride``).
        video = range(0, self.frame_chunk_size * 4 * self.frame_stride, self.frame_stride)
        observation = dict.fromkeys((FeatureType.VISUAL, FeatureType.STATE, FeatureType.ENV), video)
        return FrameSampler(Deltas(observation=observation, action=range(self.chunk_size)))

    @property
    def inference_deltas(self) -> Deltas:
        # The frames observed while the previous chunk executed, the last one current
        # (see ``LingBotVAPolicy.predict_action_chunk``).
        window = range(self.frame_stride - self.chunk_size, 1, self.frame_stride)
        return Deltas(observation=dict.fromkeys(self.obs_cam_keys, window))

    @property
    def prediction_features(self) -> dict[str, PolicyFeature]:
        """The imagined clip per camera with ``return_predicted_video``, at the VAE's resolution."""
        if not self.return_predicted_video:
            return {}
        h, w = self.height, self.width
        shapes = [(3, h, w)] * len(self.obs_cam_keys)
        if self.camera_layout == "robotwin_tshape":  # the head, then two half-size wrists
            shapes = [(3, h, w), (3, h // 2, w // 2), (3, h // 2, w // 2)]
        return {
            PREDICTION_PREFIX + key.removeprefix(OBS_PREFIX): PolicyFeature(
                type=FeatureType.VISUAL, shape=shape
            )
            for key, shape in zip(self.obs_cam_keys, shapes, strict=True)
        }
