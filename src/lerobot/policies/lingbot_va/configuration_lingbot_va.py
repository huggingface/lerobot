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
from lerobot.optim.schedulers import (
    ConstantWithWarmupSchedulerConfig,
    CosineAnnealingWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)
from lerobot.utils.constants import ACTION

# Sentinel action delta used to fetch the episode's first action as the anchor.
# ``DatasetReader._get_query_indices`` clamps every query index into
# ``[dataset_from_index, dataset_to_index - 1]``, so any delta more negative than the longest episode
# resolves to that episode's first frame (and sets ``action_is_pad[0]``). It rides on the ACTION key
# on purpose: ``observation_delta_indices`` is applied to *every* ``observation.*`` feature, cameras
# included, so an observation-side anchor would decode one extra video frame per camera per sample.
# ACTION is not a video key, so this costs one extra parquet row lookup and nothing else.
EPISODE_ANCHOR_DELTA = -(1 << 20)


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
    dtype: str = "bfloat16"  # transformer / VAE / text-encoder dtype: "bfloat16", "float16", "float32"
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

    # Action anchoring. "none" = absolute actions (upstream LIBERO). "episode" = every action in an
    # episode is expressed relative to a single anchor captured at the episode's first frame, which
    # is what upstream's relative mode does (``get_relative_pose`` anchors on the segment's first
    # action; the RoboTwin client captures ``inint_eef_pose`` once per episode and adds it back to
    # every chunk). A per-episode anchor is *anchor-stable*: the fed-back action tokens in the KV
    # cache and the tokens the model is about to emit share one reference, so the append-only cache
    # stays coherent. The old per-chunk relative mode was not, which is what made the stream reset
    # to zero displacement at every chunk boundary.
    action_anchor: str = "none"
    # Joint names to keep absolute (never anchored). Empty list = all dims anchored.
    action_anchor_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy (used to build the exclude mask).
    action_feature_names: list[str] | None = None

    # Number of latent frames per *training* sample. None -> frame_chunk_size (one inference chunk
    # per sample, the historical behaviour). Upstream trains on whole ``action_config`` segments —
    # tens of latent frames — which is what makes the block-causal mask's random chunk_size /
    # window_size meaningful and what actually trains the autoregressive action-history
    # conditioning the KV cache relies on at inference. With frame_chunk_size=2 a sample collapses
    # into a single causal block for any sampled block size >= 2, so ~75% of steps train no
    # cross-frame action conditioning at all. This knob is training-only: the inference chunk stays
    # frame_chunk_size.
    train_latent_frames: int | None = None

    # Opt-in: VAE-decode predicted video latents to ``self.last_predicted_frames`` for saving MP4s.
    save_predicted_video: bool = False

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
    # Scheduler after warmup. "constant_with_warmup" (upstream default: warmup then flat peak LR)
    # or "cosine_annealing_with_warmup" (warmup then cosine anneal peak->0 over the remaining steps).
    # Cosine tightens the loss tail and often nudges final loss down; it does NOT reduce the
    # flow-matching estimator's step-to-step noise (that's metric variance, LR-independent).
    scheduler_type: str = "constant_with_warmup"
    # Probability of corrupting the action stream's conditioning (clean/context) tokens with
    # flow-matching noise during training, mirroring the video stream's noisy_cond_prob=0.5.
    # Upstream train.py hardcodes 0.0 for actions (never corrupted) with no exposed knob; this is
    # an experimental deviation to make the model more tolerant of imperfect action history
    # (e.g. clamp-induced drift between predicted and executed actions during rollout).
    action_noisy_cond_prob: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.attn_mode not in ("torch", "flashattn", "flex"):
            raise ValueError(f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}")
        if self.action_anchor not in ("none", "episode"):
            raise ValueError(f"action_anchor must be one of 'none', 'episode'; got {self.action_anchor!r}")
        if self.train_latent_frames is not None and self.train_latent_frames < self.frame_chunk_size:
            raise ValueError(
                f"train_latent_frames ({self.train_latent_frames}) must be >= frame_chunk_size "
                f"({self.frame_chunk_size}): a training sample cannot be shorter than one inference chunk."
            )

    @property
    def train_frames(self) -> int:
        """Latent frames per training sample (``frame_chunk_size`` unless overridden)."""
        return self.train_latent_frames or self.frame_chunk_size

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
        # Default (upstream): linear warmup then constant LR (warmup_constant_lambda).
        # Optionally cosine-anneal peak->0 over the remaining steps via scheduler_type.
        if self.scheduler_type == "cosine_annealing_with_warmup":
            return CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        """Observation frame deltas for the training clip, sized to what the VAE actually reads.

        ``diffusers``' ``AutoencoderKLWan._encode`` runs ``iter_ = 1 + (n - 1) // 4`` passes over
        ``x[:, :, :1]`` then ``x[:, :, 1 + 4*(i-1) : 1 + 4*i]``, so it only ever consumes the first
        ``4 * (iter_ - 1) + 1`` frames of an ``n``-frame clip. Asking for ``frame_chunk_size * 4``
        frames (the previous formula) therefore decoded 3 frames per sample that never reached the
        encoder: at ``frame_chunk_size=2`` the deltas were ``[0, 4, ..., 28]`` and only
        ``[0, 4, 8, 12, 16]`` were used -- verified by ablation, scrambling the tail left the latents
        bit-identical.

        Requesting exactly ``4 * (train_frames - 1) + 1`` frames yields the same ``train_frames``
        latent frames with every loaded frame used, and drops the wasted video decode. The stride is
        unchanged, so the frames that do reach the model are the same ones.
        """
        temporal_downsample = 4
        stride = max(1, self.action_per_frame // temporal_downsample)
        num_frames = temporal_downsample * (self.train_frames - 1) + 1
        return [i * stride for i in range(num_frames)]

    @property
    def action_delta_indices(self) -> list[int]:
        """Action deltas for one training clip, aligned to upstream's retrospective convention.

        Latent frame ``j`` ends at ``t = j * action_per_frame`` (the Wan VAE folds each group of 4
        loaded frames into one latent frame), and upstream's dataset puts in action frame ``j`` the
        ``action_per_frame`` actions that were executed to get *from* latent frame ``j - 1`` *to*
        latent frame ``j`` -- it left-pads one action frame before trimming to
        ``latent_frames * action_per_frame`` (``lerobot_latent_dataset._action_post_process``).
        Action frame 0 is therefore pre-clip history, which is exactly why the sampling loop pins it
        (``action_cond``) and drops it on the first chunk.

        So the deltas start at ``-action_per_frame``, not 0. The count is unchanged; the window is
        shifted back by one action frame. Asking for ``range(train_frames * action_per_frame)``
        instead -- as this did before -- puts every action one frame early relative to the video,
        which the sampling loop then reads as a full ``action_per_frame`` phase lead.

        Frame 0 gets the *real* previous actions rather than upstream's hard zeros; at an episode's
        start the dataset clamps and flags them via ``action_is_pad``, which reproduces upstream's
        zero frame exactly where it belongs.
        """
        apf = self.action_per_frame
        deltas = list(range(-apf, (self.train_frames - 1) * apf))
        if self.action_anchor == "episode":
            # Prepended, so the anchor is ``batch[ACTION][:, 0]``; the anchoring processor step
            # consumes and strips it before the normalizer sees it.
            deltas = [EPISODE_ANCHOR_DELTA] + deltas
        return deltas

    @property
    def reward_delta_indices(self) -> None:
        return None
