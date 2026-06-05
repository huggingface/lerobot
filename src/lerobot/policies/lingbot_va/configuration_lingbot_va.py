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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION

# Upstream LIBERO action-normalization quantiles (single 7-DoF arm + gripper).
# Verbatim from wan_va/configs/va_libero_cfg.py (channels 0-6 of a 30-dim action space).
LIBERO_ACTION_Q01 = [
    -0.6589285731315613,
    -0.84375,
    -0.9375,
    -0.12107142806053162,
    -0.15964286029338837,
    -0.26571428775787354,
    -1.0,
]
LIBERO_ACTION_Q99 = [
    0.8999999761581421,
    0.8544642925262451,
    0.9375,
    0.17142857611179352,
    0.1842857152223587,
    0.34392857551574707,
    1.0,
]


@PreTrainedConfig.register_subclass("lingbot_va")
@dataclass
class LingBotVAConfig(PreTrainedConfig):
    """Configuration for the native LingBot-VA policy integration in LeRobot."""

    # ── Wan transformer architecture (from transformer/config.json) ──
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
    # "flex" is supported for training only and needs a recent torch build. Inference uses
    # "torch" SDPA (always available) or, optionally, "flashattn".
    attn_mode: str = "torch"

    # ── Frozen sub-models (VAE + UMT5 text encoder + tokenizer) ──
    # These heavy frozen weights (~20 GB) are NOT bundled into the LeRobot safetensors
    # checkpoint (only the trainable ~5B transformer is). They are lazily pulled from this
    # HF repo / local directory at policy-init time. The directory must contain the
    # diffusers-style ``vae/``, ``text_encoder/`` and ``tokenizer/`` sub-folders.
    wan_pretrained_path: str = "robbyant/lingbot-va-posttrain-libero-long"
    # dtype used for the transformer / VAE / text-encoder weights at inference.
    dtype: str = "bfloat16"  # one of "bfloat16", "float16", "float32"

    # ── Observation cameras (order matters: latents are concatenated on width) ──
    # Defaults match the LIBERO env feature keys (agentview -> image, eye-in-hand -> image2).
    obs_cam_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )

    # ── Inference hyperparameters (LIBERO defaults) ──
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

    # Subset of the 30-d action space actually used by the benchmark (LIBERO = 7-DoF).
    used_action_channel_ids: list[int] = field(default_factory=lambda: list(range(7)))
    # Fixed quantiles for action (un)normalization on the *used* channels.
    action_q01: list[float] = field(default_factory=lambda: list(LIBERO_ACTION_Q01))
    action_q99: list[float] = field(default_factory=lambda: list(LIBERO_ACTION_Q99))

    # Opt-in: VAE-decode the predicted video latents and stash them on
    # ``self.last_predicted_frames`` so eval/train can save predicted-video MP4s.
    save_predicted_video: bool = False

    # ── Normalization (handled internally / via custom steps, hence IDENTITY here) ──
    # Images are scaled to [-1, 1] and VAE-encoded inside the policy; actions are
    # quantile-(un)normalized by dedicated processor steps using the fixed quantiles above.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # ── Optimizer / scheduler (training; AdamW + warmup-constant per upstream train.py) ──
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.attn_mode not in ("torch", "flashattn", "flex"):
            raise ValueError(f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}")
        if len(self.action_q01) != len(self.used_action_channel_ids) or len(self.action_q99) != len(
            self.used_action_channel_ids
        ):
            raise ValueError(
                "action_q01 / action_q99 must each have one entry per used_action_channel_ids "
                f"({len(self.used_action_channel_ids)}); got {len(self.action_q01)} / {len(self.action_q99)}."
            )

    @property
    def chunk_size(self) -> int:
        """Number of single-step actions produced per autoregressive chunk."""
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self) -> int:
        """Number of actions executed before refilling (the whole chunk)."""
        return self.chunk_size

    def validate_features(self) -> None:
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
        from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig

        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
