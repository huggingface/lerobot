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
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("lingbot_va")
@dataclass
class LingBotVAConfig(PreTrainedConfig):
    """Configuration for the native LingBot-VA policy integration in LeRobot.

    Defaults match the upstream LIBERO configuration (`wan_va/configs/va_libero_cfg.py`) and the
    `transformer/config.json` of the released checkpoints.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1): Number of environment steps of observation to
            pass to the policy (the current step plus this many additional steps looking back).
        input_features (`dict[str, lerobot.configs.types.PolicyFeature] | None`, *optional*): Mapping from input feature name to its `PolicyFeature` (type and shape). Populated automatically from the dataset when not explicitly provided.
        output_features (`dict[str, lerobot.configs.types.PolicyFeature] | None`, *optional*): Mapping from output feature name to its `PolicyFeature` (type and shape). Populated automatically from the dataset when not explicitly provided.
        device (`str | None`, *optional*): Device the policy runs on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`. If unset or unavailable, auto-selected on construction.
        use_amp (`bool`, *optional*, defaults to `False`): Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`): Whether this policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`): Whether to push the trained policy to the Hugging Face Hub after training.
        repo_id (`str | None`, *optional*): Hugging Face Hub repository id to push the policy to, when `push_to_hub` is enabled.
        private (`bool | None`, *optional*): Whether to create/push the Hub repository as private.
        tags (`list[str] | None`, *optional*): Tags to attach to the policy's Hub model card.
        license (`str | None`, *optional*): License identifier to add to the policy's Hub model card.
        pretrained_path (`pathlib.Path | None`, *optional*): Path or Hub repo id of pretrained weights to initialize the policy from. If `None`, the policy is initialized from scratch.
        pretrained_revision (`str | None`, *optional*): Hub revision (branch, tag, or commit hash) pinning the pretrained model version.
        patch_size (`tuple[int, int, int]`, *optional*, defaults to `(1, 2, 2)`): Wan transformer's
            spatiotemporal patch size (time, height, width).
        num_attention_heads (`int`, *optional*, defaults to 24): Number of attention heads in the Wan
            transformer.
        attention_head_dim (`int`, *optional*, defaults to 128): Dimension per attention head.
        in_channels (`int`, *optional*, defaults to 48): Number of input channels to the transformer
            (VAE latent channels).
        out_channels (`int`, *optional*, defaults to 48): Number of output channels from the
            transformer.
        action_dim (`int`, *optional*, defaults to 30): Dimension of the action stream fed to and
            predicted by the transformer.
        text_dim (`int`, *optional*, defaults to 4096): Dimension of the UMT5 text embeddings.
        freq_dim (`int`, *optional*, defaults to 256): Dimension of the sinusoidal timestep embedding.
        ffn_dim (`int`, *optional*, defaults to 14336): Hidden dimension of the transformer's
            feed-forward blocks.
        num_layers (`int`, *optional*, defaults to 30): Number of transformer layers.
        cross_attn_norm (`bool`, *optional*, defaults to `True`): Whether to normalize the
            cross-attention inputs.
        eps (`float`, *optional*, defaults to 1e-06): Epsilon used in the transformer's normalization
            layers.
        rope_max_seq_len (`int`, *optional*, defaults to 1024): Maximum sequence length for the
            transformer's rotary position embeddings.
        attn_mode (`str`, *optional*, defaults to `"torch"`): Attention backend. `"torch"` (SDPA) or
            `"flashattn"` for inference; `"flex"` for training only, and only on a recent torch.
        wan_pretrained_path (`str`, *optional*, defaults to `"robbyant/lingbot-va-base"`): Hub id or
            local directory holding the frozen VAE, UMT5 text encoder, and tokenizer sub-folders
            (diffusers layout, ~20 GB). Lazily loaded and not bundled in the checkpoint.
        dtype (`str`, *optional*, defaults to `"bfloat16"`): Transformer/VAE/text-encoder dtype:
            `"bfloat16"`, `"float16"`, or `"float32"`.
        text_encoder_device (`str`, *optional*, defaults to `"cpu"`): Device for the frozen UMT5-XXL
            text encoder, which runs once per episode. `"cpu"` frees ~11 GB of VRAM.
        obs_cam_keys (`list[str]`, *optional*): Observation camera keys, in concatenation order (order
            matters: latents are concatenated on width). Defaults to the LIBERO camera keys.
        image_hflip (`bool`, *optional*, defaults to `False`): Whether to undo the LIBERO env
            processor's extra horizontal flip, to match the model's training orientation.
        camera_layout (`str`, *optional*, defaults to `"width_concat"`): Camera latent layout:
            `"width_concat"` (cameras concatenated on width; LIBERO) or `"robotwin_tshape"` (full-res
            head plus half-res wrists in a "T"; RoboTwin).
        height (`int`, *optional*, defaults to 128): Observation image height fed to the VAE.
        width (`int`, *optional*, defaults to 128): Observation image width fed to the VAE.
        action_per_frame (`int`, *optional*, defaults to 4): Number of single-step actions decoded per
            predicted video frame.
        frame_chunk_size (`int`, *optional*, defaults to 4): Number of video frames predicted per
            autoregressive chunk.
        attn_window (`int`, *optional*, defaults to 30): Attention window size, in frames, for the
            causal streaming KV cache.
        num_inference_steps (`int`, *optional*, defaults to 20): Number of denoising steps for the
            video-latent flow-matching scheduler.
        video_exec_step (`int`, *optional*, defaults to -1): Which decoded video frame index to treat
            as "executed" for KV-cache feedback. `-1` uses the last frame.
        action_num_inference_steps (`int`, *optional*, defaults to 50): Number of denoising steps for
            the action flow-matching scheduler.
        guidance_scale (`float`, *optional*, defaults to 5.0): Classifier-free guidance scale for the
            video-latent stream.
        action_guidance_scale (`float`, *optional*, defaults to 1.0): Classifier-free guidance scale
            for the action stream.
        snr_shift (`float`, *optional*, defaults to 5.0): Flow-matching noise-schedule shift for the
            video-latent stream.
        action_snr_shift (`float`, *optional*, defaults to 0.05): Flow-matching noise-schedule shift
            for the action stream.
        max_sequence_length (`int`, *optional*, defaults to 512): Maximum UMT5 prompt length.
        used_action_channel_ids (`list[int]`, *optional*): Subset of the 30-d action space used by the
            benchmark; defaults to the first 7 channels (LIBERO's 7-DoF action). The action
            (un)normalization quantiles live in the checkpoint's `policy_postprocessor.json`, not here.
        save_predicted_video (`bool`, *optional*, defaults to `False`): Whether to VAE-decode predicted
            video latents into `self.last_predicted_frames`, opt-in for saving MP4s.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*): Per-feature-type
            normalization mode. Always `IDENTITY`: images are scaled and VAE-encoded, and actions are
            quantile-(un)normalized, inside the policy or a dedicated processor step.
        optimizer_lr (`float`, *optional*, defaults to 1e-05): AdamW learning rate.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`): AdamW betas.
        optimizer_eps (`float`, *optional*, defaults to 1e-08): AdamW epsilon.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.0001): AdamW weight decay.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 1.0): Gradient clipping norm.
        scheduler_warmup_steps (`int`, *optional*, defaults to 1000): Number of linear-warmup steps
            before the constant learning-rate phase.
    """

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

    def __post_init__(self):
        """Validate `attn_mode`.

        Raises:
            ValueError: If `attn_mode` is not one of `"torch"`, `"flashattn"`, or `"flex"`.
        """
        super().__post_init__()
        if self.attn_mode not in ("torch", "flashattn", "flex"):
            raise ValueError(f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}")

    @property
    def chunk_size(self) -> int:
        """Number of single-step actions produced per autoregressive chunk."""
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self) -> int:
        """Number of actions executed before refilling (the whole chunk)."""
        return self.chunk_size

    def validate_features(self) -> None:
        """Validate and set up input/output features for LingBot-VA.

        Raises:
            ValueError: If no visual input feature is present in `input_features`.
        """
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
        """Return the AdamW optimizer configuration built from the `optimizer_*` fields."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Return the linear-warmup-then-constant scheduler configuration, matching upstream's `warmup_constant_lambda`."""
        # Upstream uses a linear warmup followed by a constant LR (warmup_constant_lambda).
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        """Return the keyframe-sampling indices used to build the observed-frame history."""
        temporal_downsample = 4
        stride = max(1, self.action_per_frame // temporal_downsample)
        return list(range(0, self.frame_chunk_size * temporal_downsample * stride, stride))

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for LingBot-VA)."""
        return None
