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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    PreTrainedConfig,
)
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE

WAN22_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_DIFFUSERS_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FASTWAM_BASE_MODEL_ID = "lerobot/fastwam_base"
WAN_T5_TOKENIZER_ID = "google/umt5-xxl"


_FASTWAM_VIDEO_BASE_COMPAT_KEYS = (
    "patch_size",
    "in_dim",
    "hidden_dim",
    "ffn_dim",
    "freq_dim",
    "text_dim",
    "out_dim",
    "num_heads",
    "attn_head_dim",
    "num_layers",
)

_FASTWAM_ACTION_BASE_COMPAT_KEYS = (
    "hidden_dim",
    "ffn_dim",
    "num_heads",
    "attn_head_dim",
    "num_layers",
    "text_dim",
    "freq_dim",
)


def default_video_dit_config(action_dim: int) -> dict[str, Any]:
    """Return the default kwargs dict for the video-generation DiT backbone, sized for `action_dim`."""
    return {
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
        "fp32_attention": True,
    }


def default_action_dit_config(action_dim: int) -> dict[str, Any]:
    """Return the default kwargs dict for the action-generation DiT backbone, sized for `action_dim`."""
    return {
        "action_dim": action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": False,
        "fp32_attention": True,
    }


def _coerce_enum(enum_cls: type, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError) as exc:
        member = getattr(enum_cls, str(value), None)
        if member is None:
            raise ValueError(f"Cannot coerce {value!r} into {enum_cls.__name__}.") from exc
        return member


def _coerce_policy_features(features: dict[str, Any] | None) -> dict[str, PolicyFeature] | None:
    if features is None:
        return None
    coerced = {}
    for name, feature in features.items():
        if isinstance(feature, PolicyFeature):
            coerced[name] = feature
            continue
        coerced[name] = PolicyFeature(
            type=_coerce_enum(FeatureType, feature["type"]),
            shape=tuple(feature["shape"]),
        )
    return coerced


def _is_local_model_id(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() or value.startswith(("./", "../", "~")) or path.exists()


def _validate_wan_model_id(value: str, field_name: str) -> str:
    if value == WAN22_MODEL_ID or _is_local_model_id(value):
        return value
    raise ValueError(f"`{field_name}` must be `{WAN22_MODEL_ID}` or an explicit local path, got `{value}`.")


def is_fastwam_base_compatible_config(config: FastWAMConfig) -> bool:
    """Return whether `fastwam_base` partial weights can initialize this config."""
    default_video_config = default_video_dit_config(config.action_dim)
    default_action_config = default_action_dit_config(config.action_dim)
    return all(
        config.video_dit_config.get(key) == default_video_config.get(key)
        for key in _FASTWAM_VIDEO_BASE_COMPAT_KEYS
    ) and all(
        config.action_dit_config.get(key) == default_action_config.get(key)
        for key in _FASTWAM_ACTION_BASE_COMPAT_KEYS
    )


@PreTrainedConfig.register_subclass("fastwam")
@dataclass
class FastWAMConfig(PreTrainedConfig):
    """Configuration for the FastWAM LeRobot policy.

    FastWAM adapts the Wan2.2 video-diffusion backbone into a robot policy: a video expert and an action
    expert are jointly trained (or fine-tuned) as a Mixture-of-Transformers, sharing attention over a
    predicted future video and the corresponding action chunk.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy.
        input_features (`dict[str, PolicyFeature]`, *optional*):
            Input feature specification, keyed by feature name. `__post_init__` builds a synthetic
            single-image default at `image_size` when left unset; `set_dataset_feature_metadata` later
            replaces it with the dataset's real per-camera keys.
        output_features (`dict[str, PolicyFeature]`, *optional*):
            Output feature specification, keyed by feature name. `__post_init__` builds a default `action`
            feature of shape `(action_dim,)` when left unset.
        device (`str`, *optional*):
            Torch device to run the policy on, e.g. `"cuda"` or `"cpu"`. Auto-selected when unset or
            unavailable.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub.
        repo_id (`str`, *optional*):
            Hub repository id to push the policy to.
        private (`bool`, *optional*):
            Whether the pushed Hub repository is private.
        tags (`list[str]`, *optional*):
            Tags to attach to the policy on the Hub.
        license (`str`, *optional*):
            License identifier for the policy on the Hub.
        pretrained_path (`Path`, *optional*):
            Repo id or local directory of pretrained weights saved with `save_pretrained`. Auto-populated
            from `base_model_id` when the DiT configs are `fastwam_base`-compatible; otherwise left unset
            to initialize from scratch.
        pretrained_revision (`str`, *optional*):
            Hub revision to pin when loading `pretrained_path`.
        action_dim (`int`, *optional*, defaults to 7):
            Number of scalar action channels per timestep.
        proprio_dim (`int`, *optional*, defaults to 8):
            Number of proprioception channels used as an extra text-context token. `None` disables proprio
            conditioning.
        action_horizon (`int`, *optional*, defaults to 32):
            Number of actions predicted by one policy call.
        n_action_steps (`int`, *optional*, defaults to 32):
            Number of actions from a predicted chunk that are actually executed before re-querying the
            policy. Must not exceed `action_horizon`.
        num_video_frames (`int`, *optional*, defaults to 33):
            Raw video sampling window, in dataset frames. The model actually operates on
            `model_video_frames` frames after subsampling by `action_video_freq_ratio`.
        action_video_freq_ratio (`int`, *optional*, defaults to 4):
            Actions are sampled at this multiple of the video frame rate. Video frames are taken every
            `action_video_freq_ratio`-th raw frame, so the model sees `(num_video_frames - 1) // ratio + 1`
            frames spanning the same time window as `action_horizon` actions.
        image_size (`tuple[int, int]`, *optional*, defaults to `(224, 448)`):
            Concatenated image size as `(height, width)`, shared across every camera view.
        context_len (`int`, *optional*, defaults to 128):
            Maximum text embedding token length.
        model_id (`str`, *optional*, defaults to `"Wan-AI/Wan2.2-TI2V-5B"`):
            Hub id (or local path) of the Wan2.2 video-diffusion backbone.
        tokenizer_model_id (`str`, *optional*, defaults to `"google/umt5-xxl"`):
            Hub id of the UMT5 tokenizer.
        text_encoder_model_id (`str`, *optional*, defaults to `"Wan-AI/Wan2.2-TI2V-5B-Diffusers"`):
            Hub id of the frozen UMT5 text encoder and VAE used for text/video conditioning.
        base_model_id (`str`, *optional*, defaults to `"lerobot/fastwam_base"`):
            Hub id of the FastWAM base checkpoint used to auto-populate `pretrained_path` when the DiT
            configs are compatible with it. `None` disables this auto-loading.
        tokenizer_max_len (`int`, *optional*, defaults to 128):
            Maximum token length passed to the tokenizer.
        load_text_encoder (`bool`, *optional*, defaults to `True`):
            Whether to load the frozen UMT5 text encoder. Disable when the batch always supplies
            precomputed `context`/`context_mask`.
        mot_checkpoint_mixed_attn (`bool`, *optional*, defaults to `False`):
            Whether the Mixture-of-Transformers module checkpoints its mixed video/action attention.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the Wan backbone and action expert are built and run in.
        prompt_template (`str`, *optional*, defaults to `"A video recorded from a robot's point of view executing the following instruction: {task}"`):
            Template the raw `task` string is formatted into before text encoding.
        num_inference_steps (`int`, *optional*, defaults to 10):
            Number of denoising steps used at inference time.
        inference_seed (`int`, *optional*, defaults to 42):
            Random seed for the inference noise sampler. `None` samples fresh noise every call.
        rand_device (`str`, *optional*, defaults to `"cpu"`):
            Device the inference noise sampler draws from.
        text_cfg_scale (`float`, *optional*, defaults to 1.0):
            Classifier-free-guidance scale applied against `negative_prompt` at inference time.
        negative_prompt (`str`, *optional*, defaults to `""`):
            Negative prompt used for classifier-free guidance.
        sigma_shift (`float`, *optional*):
            Overrides the diffusion schedule's sigma shift at inference time. `None` uses the scheduler's
            own shift.
        tiled (`bool`, *optional*, defaults to `False`):
            Whether to run the Wan VAE in tiled mode to reduce memory use.
        fp32_attention (`bool`, *optional*, defaults to `True`):
            Whether the video and action DiT experts compute attention in fp32.
        use_gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable activation checkpointing in both DiT experts, trading compute for memory.
            Propagated into `video_dit_config` and `action_dit_config`.
        freeze_video_expert (`bool`, *optional*, defaults to `False`):
            Whether to freeze the ~5B Wan video expert so only the action expert and proprio encoder
            train, cutting the AdamW optimizer footprint substantially. Also set `loss.lambda_video=0` to
            skip the now-gradient-free video loss compute.
        toggle_action_dimensions (`list[int]`, *optional*):
            Action dimensions the postprocessor flips between two fixed values, for LIBERO-style toggle
            actions such as the gripper. Empty disables the toggle.
        video_scheduler (`dict[str, float | int]`, *optional*):
            Train/inference shift and step-count settings for the video diffusion scheduler.
        action_scheduler (`dict[str, float | int]`, *optional*):
            Train/inference shift and step-count settings for the action diffusion scheduler.
        loss (`dict[str, float]`, *optional*):
            Per-term loss weights, keyed by `"lambda_video"` and `"lambda_action"`.
        video_dit_config (`dict[str, Any]`, *optional*):
            Wan video expert architecture config. Built from `default_video_dit_config(action_dim)` when
            left unset.
        action_dit_config (`dict[str, Any]`, *optional*):
            Action expert architecture config. Built from `default_action_dit_config(action_dim)` when
            left unset.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps each `FeatureType` to the `NormalizationMode` used to normalize/unnormalize it.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Learning rate used to build the default `AdamWConfig` optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.01):
            Weight decay for the default optimizer preset.
    """

    n_obs_steps: int = 1
    action_dim: int = 7
    proprio_dim: int | None = 8
    action_horizon: int = 32
    n_action_steps: int = 32
    num_video_frames: int = 33
    action_video_freq_ratio: int = 4
    image_size: tuple[int, int] = (224, 448)
    context_len: int = 128
    model_id: str = WAN22_MODEL_ID
    tokenizer_model_id: str = WAN_T5_TOKENIZER_ID
    text_encoder_model_id: str = WAN22_DIFFUSERS_MODEL_ID
    base_model_id: str | None = FASTWAM_BASE_MODEL_ID
    tokenizer_max_len: int = 128
    load_text_encoder: bool = True
    mot_checkpoint_mixed_attn: bool = False
    torch_dtype: str = "bfloat16"
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    num_inference_steps: int = 10
    inference_seed: int | None = 42
    rand_device: str = "cpu"
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    sigma_shift: float | None = None
    tiled: bool = False
    fp32_attention: bool = True
    use_gradient_checkpointing: bool = False
    freeze_video_expert: bool = False
    toggle_action_dimensions: list[int] = field(default_factory=list)
    video_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    action_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    loss: dict[str, float] = field(default_factory=lambda: {"lambda_video": 1.0, "lambda_action": 1.0})
    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    optimizer_lr: float = 1.0e-4
    optimizer_weight_decay: float = 1.0e-2

    def __post_init__(self) -> None:
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the DiT/video backbone configuration."""
        super().__post_init__()
        self.image_size = tuple(self.image_size)
        self.model_id = _validate_wan_model_id(self.model_id, "model_id")
        self.input_features = _coerce_policy_features(self.input_features)
        self.output_features = _coerce_policy_features(self.output_features)
        self.toggle_action_dimensions = [int(dim) for dim in self.toggle_action_dimensions]
        self.video_dit_config = self.video_dit_config or default_video_dit_config(self.action_dim)
        self.action_dit_config = self.action_dit_config or default_action_dit_config(self.action_dim)
        self.video_dit_config["fp32_attention"] = bool(self.fp32_attention)
        self.action_dit_config["fp32_attention"] = bool(self.fp32_attention)
        self.video_dit_config["use_gradient_checkpointing"] = bool(self.use_gradient_checkpointing)
        self.action_dit_config["use_gradient_checkpointing"] = bool(self.use_gradient_checkpointing)
        if self.input_features is None:
            height, width = self.image_size
            self.input_features = {
                "observation.images.image": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, height, width),
                )
            }
            if self.proprio_dim is not None:
                self.input_features[OBS_STATE] = PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.proprio_dim,),
                )
        if self.output_features is None:
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}
        self.validate_features()
        if self.pretrained_path or self.use_peft or not self.base_model_id:
            return
        if not is_fastwam_base_compatible_config(self):
            return
        self.pretrained_path = Path(self.base_model_id)
        self._auto_pretrained_path = True

    def _save_pretrained(self, save_directory: Path) -> None:
        if not getattr(self, "_auto_pretrained_path", False):
            super()._save_pretrained(save_directory)
            return

        pretrained_path = self.pretrained_path
        self.pretrained_path = None
        try:
            super()._save_pretrained(save_directory)
        finally:
            self.pretrained_path = pretrained_path

    def get_optimizer_preset(self) -> AdamWConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return None

    def set_dataset_feature_metadata(self, dataset_features: dict[str, Any]) -> None:
        """Rebuild visual input features from the dataset's real camera keys.

        FastWAM's `__post_init__` installs a synthetic single-image default
        (`observation.images.image` at full `image_size` width). For datasets
        with one or more separately-named cameras (e.g. `observation.images.top`,
        `observation.images.wrist`), this hook — invoked by `make_policy` once the
        dataset metadata is known — replaces that default with the actual camera
        keys, each declared at the policy's native per-camera resolution
        (`image_size[0]` x `image_size[1] // num_cameras`). The accompanying
        resize step in `make_fastwam_pre_post_processors` resizes raw frames to
        match, so heterogeneous source resolutions (e.g. 480x640) are supported.
        """
        image_keys = sorted(
            key
            for key, feature in dataset_features.items()
            if key.startswith("observation.images.") and feature.get("dtype") in ("video", "image")
        )
        if not image_keys:
            return
        height, total_width = self.image_size
        per_cam_width = total_width // len(image_keys)
        new_inputs: dict[str, PolicyFeature] = {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, per_cam_width))
            for key in image_keys
        }
        if self.proprio_dim is not None and OBS_STATE in dataset_features:
            new_inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,))
        self.input_features = new_inputs
        self.validate_features()

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("`n_action_steps` cannot exceed `action_horizon`.")
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {self.action_video_freq_ratio}."
            )
        # Video frames are subsampled by action_video_freq_ratio; the resulting model frame
        # count must satisfy T % 4 == 1 for the VAE temporal tokenization (mirrors the
        # original FastWAM dataset asserts).
        if (self.num_video_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"`num_video_frames - 1` ({self.num_video_frames - 1}) must be divisible by "
                f"`action_video_freq_ratio` ({self.action_video_freq_ratio})."
            )
        if ((self.num_video_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                f"Subsampled video transitions ({(self.num_video_frames - 1) // self.action_video_freq_ratio}) "
                "must be divisible by 4 for VAE tokenization (i.e. model_video_frames % 4 == 1)."
            )
        if self.action_horizon % ((self.num_video_frames - 1) // self.action_video_freq_ratio) != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be divisible by the number of "
                f"video transitions ({(self.num_video_frames - 1) // self.action_video_freq_ratio})."
            )
        if not self.image_features:
            raise ValueError("FastWAM requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("FastWAM requires `action` in output_features.")
        action_shape = tuple(self.action_feature.shape)
        if action_shape != (self.action_dim,):
            raise ValueError(
                f"FastWAM action feature shape must be ({self.action_dim},), got {action_shape}."
            )
        if self.proprio_dim is not None:
            state_feature = self.robot_state_feature
            if state_feature is None:
                raise ValueError("FastWAM requires `observation.state` when `proprio_dim` is set.")
            state_shape = tuple(state_feature.shape)
            if state_shape != (self.proprio_dim,):
                raise ValueError(
                    f"FastWAM state feature shape must be ({self.proprio_dim},), got {state_shape}."
                )
        height, width = self.image_size
        image_width_sum = 0
        for name, feature in self.image_features.items():
            shape = tuple(feature.shape)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"FastWAM image feature `{name}` must have shape (3, H, W), got {shape}.")
            if shape[1] != height:
                raise ValueError(f"FastWAM image feature `{name}` height must be {height}, got {shape[1]}.")
            image_width_sum += shape[2]
        if image_width_sum != width:
            raise ValueError(f"FastWAM image feature widths must sum to {width}, got {image_width_sum}.")

    @property
    def model_video_frames(self) -> int:
        """Number of video frames the model actually operates on.

        Computed by subsampling the raw `num_video_frames` window by `action_video_freq_ratio` (e.g.
        33 -> 9).
        """
        return (self.num_video_frames - 1) // self.action_video_freq_ratio + 1

    @property
    def observation_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        # Load the video frames the model is supervised on: the future window subsampled by
        # action_video_freq_ratio (e.g. [0, 4, 8, ..., 32] -> 9 frames). Each video frame is
        # thus `action_video_freq_ratio` actions apart, while actions load at the full rate
        # (`action_delta_indices` = range(action_horizon)). Returning None would load only the
        # current frame, making the video target a static repeat (degenerate supervision).
        return list(range(0, self.num_video_frames, self.action_video_freq_ratio))

    @property
    def action_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
