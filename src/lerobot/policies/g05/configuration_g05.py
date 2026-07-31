# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import draccus
from draccus.parsers.decoding import decode_dataclass

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _default_layer_types() -> list[str]:
    return ["linear_attention" if index % 4 != 3 else "full_attention" for index in range(24)]


@PreTrainedConfig.register_subclass("g05")
@dataclass
class G05Config(PreTrainedConfig):
    """Serializable G0.5 architecture and data-boundary contract.

    Values that vary between published checkpoints are stored in the converted
    artifact. Runtime code never infers an embodiment from a checkpoint name.

    Args:
        chunk_size: Number of future actions predicted by one model forward. It
            is part of the trained action-expert shape and dataset sampling.
        n_action_steps: Number of predicted actions consumed by ``select_action``
            before the policy runs the model again. It defaults to ``chunk_size``
            and may be overridden at inference time without changing model weights.
        n_obs_steps: Number of observation frames supplied for each camera.
        camera_keys: Real LeRobot image feature keys expected by the checkpoint.
        dummy_camera_keys: Camera slots always filled with zero images.
        optional_camera_keys: Real camera slots filled with zero images when the
            hardware does not provide them.
        camera_order: Checkpoint camera order, including real and dummy cameras.
        internal_action_dim: Checkpoint action width before removing padded or
            embodiment-specific dimensions.
        internal_state_dim: Checkpoint state width after processor-side padding.
        action_indices: Mapping from physical action dimensions into checkpoint
            dimensions; empty means the leading dimensions.
        state_indices: Mapping from physical state dimensions into checkpoint
            dimensions; empty means the leading dimensions.
        action_normalization: Serialized per-horizon statistics for a post-trained
            checkpoint. Empty for a base model using LeRobot normalization.
        state_normalization: Serialized state statistics for a post-trained
            checkpoint. Empty for a base model using LeRobot normalization.
        normalization_strategy: ``lerobot`` for replaceable dataset statistics,
            or ``g05_stepwise`` for the fixed stats of post-trained checkpoints.
        relative_action_mask: Physical action dimensions represented relative to
            the current state. Unmarked dimensions, such as grippers, stay absolute.
        joint_signs: Optional physical-arm to checkpoint-frame sign transform.
        joint_offsets: Optional physical-arm to checkpoint-frame offset transform.
        embodiment: Prompt name serialized from the original checkpoint config.
        num_inference_steps: Euler steps used to solve the flow-matching action field.
        flow_sampling: Training-time flow-time distribution (``beta`` or ``uniform``).
        num_flow_samples: Noisy flow samples drawn per training example.
        flow_joint_training: Whether flow loss backpropagates through the VLM prefix.
        discrete_action: Enable the autoregressive action-token objective or inference path.
        continuous_action: Enable the flow-matching action objective or inference path.
        cot_prompt: Inference instruction placed immediately before the CoT boundary.
        predict_cot: Deployment switch. When enabled, generate CoT before the
            enabled action head; when disabled, use the direct-action prefix.
        dtype: Model autocast precision; checkpoint precision islands remain fp32.
        attn_implementation: Text and action-expert attention backend.
        vision_attn_implementation: Vision attention backend, configured separately
            because the released model used a different effective backend.
    """

    # Policy I/O horizons. Only chunk_size shapes the trained prediction target;
    # n_action_steps controls the inference action queue.
    chunk_size: int = 32
    n_action_steps: int | None = None
    n_obs_steps: int = 1

    # Image layout serialized per embodiment.
    image_size: tuple[int, int] = (256, 256)
    camera_keys: list[str] = field(default_factory=list)
    dummy_camera_keys: list[str] = field(default_factory=list)
    optional_camera_keys: list[str] = field(default_factory=list)
    camera_order: list[str] = field(default_factory=list)

    # Physical robot vectors are mapped into these checkpoint-width tensors.
    internal_action_dim: int = 27
    internal_state_dim: int = 27
    action_indices: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    action_normalization: list[dict] = field(default_factory=list)
    state_normalization: list[dict] = field(default_factory=list)
    normalization_strategy: str = "lerobot"
    relative_action_mask: list[bool] = field(default_factory=list)
    action_feature_names: list[str] = field(default_factory=list)
    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None
    embodiment: str = "unknown"

    # Tokenizer contract. Token IDs are serialized because converted artifacts
    # are fully offline and may include checkpoint-specific action tokens.
    vocab_size: int = 252189
    pad_token_id: int = 0
    eos_token_id: int = 248044
    image_token_id: int = 248056
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    state_token_id: int | None = None
    eov_token_id: int | None = None
    max_task_tokens: int = 200
    max_prompt_length: int = 1200

    # Qwen3.5 text backbone architecture.
    text_hidden_size: int = 2048
    text_intermediate_size: int = 6144
    text_num_layers: int = 24
    text_num_heads: int = 8
    text_num_kv_heads: int = 2
    text_head_dim: int = 256
    text_layer_types: list[str] = field(default_factory=_default_layer_types)
    rope_theta: float = 10_000_000.0
    mrope_section: tuple[int, int, int] = (11, 11, 10)

    # Qwen3.5 vision tower architecture.
    vision_depth: int = 24
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_heads: int = 16
    vision_patch_size: int = 16
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2

    # Bidirectional flow-matching action expert architecture.
    expert_hidden_size: int = 1024
    expert_intermediate_size: int = 4096
    expert_num_layers: int = 24
    expert_num_heads: int = 8
    expert_num_kv_heads: int = 2
    expert_head_dim: int = 256

    # Flow matching and optional discrete action / CoT objectives.
    num_inference_steps: int = 10
    flow_sig_min: float = 0.001
    flow_sampling: str = "beta"
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    num_flow_samples: int = 1
    flow_joint_training: bool = True
    fm_loss_weight: float = 1.0
    action_token_loss_weight: float = 0.0
    action_token_start_id: int | None = None
    action_token_end_id: int | None = None
    max_cot_tokens: int = 300
    max_action_tokens: int = 300
    ar_do_sample: bool = False
    ar_temperature: float = 0.7
    ar_top_k: int = 128
    ar_top_p: float = 0.95
    ar_repetition_penalty: float = 1.0
    ar_no_repeat_ngram_size: int = 0
    cot_prompt: str = ""
    predict_cot: bool = False
    discrete_action: bool = False
    continuous_action: bool = True

    # Runtime numerical backends. Flash dependencies remain optional and local.
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    vision_attn_implementation: str = "sdpa"

    # LeRobot training presets; these do not affect checkpoint architecture.
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1e-6

    # Conversion records these values rather than leaving runtime references to
    # Hydra configs or dataset-stat files.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )
    tokenizer_subdir: str = "processor"
    action_tokenizer_subdir: str = "action_tokenizer"

    def __post_init__(self) -> None:
        if self.n_action_steps is None:
            self.n_action_steps = self.chunk_size
        super().__post_init__()
        if self.chunk_size <= 0 or self.n_action_steps <= 0:
            raise ValueError("chunk_size and n_action_steps must be positive")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if self.internal_action_dim <= 0 or self.internal_state_dim <= 0:
            raise ValueError("internal state and action dimensions must be positive")
        if self.max_task_tokens <= 0 or self.max_prompt_length <= 0:
            raise ValueError("prompt token limits must be positive")
        if len(self.text_layer_types) != self.text_num_layers:
            raise ValueError("text_layer_types must contain one entry per text layer")
        if any(kind not in {"linear_attention", "full_attention"} for kind in self.text_layer_types):
            raise ValueError("unsupported Qwen3.5 layer type")
        if self.flow_sampling not in {"beta", "uniform"}:
            raise ValueError("flow_sampling must be 'beta' or 'uniform'")
        if self.num_flow_samples <= 0:
            raise ValueError("num_flow_samples must be positive")
        if self.max_cot_tokens <= 0 or self.max_action_tokens <= 0:
            raise ValueError("AR token limits must be positive")
        if self.ar_temperature < 0:
            raise ValueError("ar_temperature must be non-negative")
        if self.ar_top_k < 0 or not 0 < self.ar_top_p <= 1:
            raise ValueError("invalid AR top-k/top-p sampling configuration")
        if self.ar_repetition_penalty <= 0 or self.ar_no_repeat_ngram_size < 0:
            raise ValueError("invalid AR repetition configuration")
        if self.predict_cot and not self.cot_prompt.strip():
            raise ValueError("predict_cot=true requires the checkpoint's inference CoT prompt")
        if not self.discrete_action and not self.continuous_action:
            raise ValueError("at least one of discrete_action or continuous_action must be enabled")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        supported_attention = {"eager", "sdpa", "flash_attention_2", "flash_attention_4"}
        if self.attn_implementation not in supported_attention:
            raise ValueError("unsupported text/action attention implementation")
        if self.vision_attn_implementation not in supported_attention:
            raise ValueError("unsupported vision attention implementation")
        if self.normalization_strategy not in {"lerobot", "g05_stepwise"}:
            raise ValueError("normalization_strategy must be 'lerobot' or 'g05_stepwise'")
        if self.normalization_strategy == "g05_stepwise" and (
            not self.action_normalization or not self.state_normalization
        ):
            raise ValueError("g05_stepwise requires serialized state and action statistics")
        if (self.joint_signs is None) != (self.joint_offsets is None):
            raise ValueError("joint_signs and joint_offsets must be configured together")
        if self.joint_signs is not None:
            # The postprocessor inverts the frame transform as
            # ``signs * (action - offsets)``, which only undoes
            # ``signs * state + offsets`` when every sign is +1 or -1.
            if any(abs(sign) != 1 for sign in self.joint_signs):
                raise ValueError("joint_signs entries must be +1 or -1 to stay invertible")
            state = (self.input_features or {}).get(OBS_STATE)
            action = (self.output_features or {}).get(ACTION)
            physical_dims = {feature.shape[-1] for feature in (state, action) if feature is not None}
            if len(self.joint_signs) != len(self.joint_offsets) or any(
                len(self.joint_signs) > dimension for dimension in physical_dims
            ):
                raise ValueError("joint frame transform exceeds the physical state/action width")
        if (self.action_token_start_id is None) != (self.action_token_end_id is None):
            raise ValueError("action token range must define both start and end")
        action_dim = self.output_features.get(ACTION)
        if self.relative_action_mask and (
            action_dim is None or len(self.relative_action_mask) != action_dim.shape[-1]
        ):
            raise ValueError("relative_action_mask must match the physical action dimension")
        if self.action_feature_names and (
            action_dim is None or len(self.action_feature_names) != action_dim.shape[-1]
        ):
            raise ValueError("action_feature_names must match the physical action dimension")
        if len(set(self.action_feature_names)) != len(self.action_feature_names):
            raise ValueError("action_feature_names must be unique")
        duplicate_cameras = set(self.camera_keys) & set(self.dummy_camera_keys)
        if duplicate_cameras:
            raise ValueError(f"cameras cannot be both real and dummy: {sorted(duplicate_cameras)}")
        unknown_optional_cameras = set(self.optional_camera_keys) - set(self.camera_keys)
        if unknown_optional_cameras:
            raise ValueError(
                f"optional cameras must also be real camera keys: {sorted(unknown_optional_cameras)}"
            )
        known_cameras = set(self.camera_keys) | set(self.dummy_camera_keys)
        if self.camera_order and (
            set(self.camera_order) != known_cameras or len(self.camera_order) != len(known_cameras)
        ):
            raise ValueError("camera_order must contain every real and dummy camera exactly once")

    def validate_features(self) -> None:
        state = self.input_features.get(OBS_STATE)
        action = self.output_features.get(ACTION)
        if state is None or state.type is not FeatureType.STATE:
            raise ValueError(f"G0.5 requires a {OBS_STATE!r} state feature")
        if action is None or action.type is not FeatureType.ACTION:
            raise ValueError(f"G0.5 requires an {ACTION!r} action feature")
        if state.shape[-1] > self.internal_state_dim:
            raise ValueError("physical state dimension exceeds the checkpoint state dimension")
        if action.shape[-1] > self.internal_action_dim:
            raise ValueError("physical action dimension exceeds the checkpoint action dimension")

        visual_keys = {
            key for key, feature in self.input_features.items() if feature.type is FeatureType.VISUAL
        }
        missing = set(self.camera_keys) - visual_keys
        if missing:
            raise ValueError(f"missing configured G0.5 cameras: {sorted(missing)}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@draccus.decode.register(G05Config)
def _decode_g05_config(raw_value: dict[str, Any], path: tuple[str, ...]) -> G05Config:
    config = dict(raw_value)
    config.pop("action_attend_cot", None)
    return decode_dataclass(G05Config, config, path)
