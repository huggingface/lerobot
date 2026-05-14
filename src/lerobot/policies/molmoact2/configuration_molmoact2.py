from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    AdamWConfig,
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

MOLMOACT2_DEFAULT_NUM_IMAGES = 2
MOLMOACT2_IMAGE_TOKENS_PER_IMAGE = 196
MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET = 80
MOLMOACT2_TASK_TOKEN_BUDGET = 32
MOLMOACT2_SEQUENCE_LENGTH_MARGIN = 32
MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE = 64
MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS = 4
MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP = 6
MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM = 0.95


@LRSchedulerConfig.register_subclass("molmoact2_cosine_decay_with_warmup")
@dataclass
class MolmoAct2CosineDecayWithWarmupSchedulerConfig(CosineDecayWithWarmupSchedulerConfig):
    """MolmoAct2-local cosine scheduler with optional decay-step auto-match.

    LeRobot's generic cosine scheduler keeps an explicit integer decay length.
    For MolmoAct2, leaving num_decay_steps unset means "decay across this run's
    training steps"; build() is the first point where num_training_steps is known.
    """

    num_decay_steps: int | None

    def build(self, optimizer, num_training_steps: int):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.peak_lr,
            decay_lr=self.decay_lr,
            num_warmup_steps=self.num_warmup_steps,
            num_decay_steps=num_training_steps if self.num_decay_steps is None else self.num_decay_steps,
        ).build(optimizer, num_training_steps=num_training_steps)


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def infer_molmoact2_max_sequence_length(
    *,
    num_images: int,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    include_discrete_action: bool,
) -> int:
    """Infer the padded text/image sequence cap from MolmoAct2's fixed token layout."""
    if num_images < 1:
        num_images = MOLMOACT2_DEFAULT_NUM_IMAGES
    if state_dim < 0:
        state_dim = 0
    if action_dim < 1:
        action_dim = 1
    if action_horizon < 1:
        action_horizon = 1

    image_tokens = num_images * MOLMOACT2_IMAGE_TOKENS_PER_IMAGE
    prompt_tokens = (
        MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET
        + MOLMOACT2_TASK_TOKEN_BUDGET
        + state_dim
        + MOLMOACT2_SEQUENCE_LENGTH_MARGIN
    )
    action_tokens = 0
    if include_discrete_action:
        action_tokens_per_step = max(
            MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP,
            math.ceil(action_dim * MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM),
        )
        action_tokens = MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS + action_horizon * action_tokens_per_step

    return _round_up(
        image_tokens + prompt_tokens + action_tokens,
        MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE,
    )


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """MolmoAct2 policy backed by the converted HF checkpoint implementation."""

    checkpoint_path: str = "allenai/MolmoAct2"
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False
    trust_remote_code: bool = True

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    action_mode: str = "both"
    inference_action_mode: str | None = None
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    discrete_generation_max_steps: int | None = None
    norm_tag: str | None = None

    setup_type: str = ""
    control_mode: str = ""
    image_keys: list[str] = field(default_factory=list)
    normalize_language: bool = True
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    normalize_gripper: bool = False
    num_state_tokens: int = 256
    # Leave unset for the default MolmoAct2 sequence budget inferred from the fixed
    # image/prompt/state/action token layout. Override only for unusual long prompts.
    max_sequence_length: int | None = None

    # Fixed by released MolmoAct2 checkpoints. We validate this at model load.
    expected_max_action_dim: int = 32

    # Flow-matching training knobs copied from the original MolmoAct2 training path.
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_inference_steps: int | None = None
    mask_action_dim_padding: bool = True
    enable_inference_cuda_graph: bool = True
    # MolmoAct2-local eval option. When enabled, stochastic continuous action
    # generation uses a rollout-local generator derived from eval_seed.
    per_episode_seed: bool = False
    eval_seed: int | None = None
    rtc_config: RTCConfig | None = None

    # Default is full finetuning with gradients from the action expert flowing into the VLM.
    enable_lora_vlm: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    enable_lora_action_expert: bool = False
    enable_knowledge_insulation: bool = False
    freeze_embedding: bool = True
    train_action_expert_only: bool = False
    gradient_checkpointing: bool = False

    model_dtype: str = "bfloat16"
    softmax_auxiliary_loss: bool = True
    softmax_auxiliary_loss_scale: float = 1e-4
    discrete_loss_token_weighting: str = "root_subsegments_root_tokens"

    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_connector_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int | None = None
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    dataset_feature_names: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode not in {"continuous", "discrete", "both"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}. "
                "Expected one of {'continuous', 'discrete', 'both'}."
            )
        if self.inference_action_mode not in {None, "continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {None, 'continuous', 'discrete'}."
            )
        if self.inference_action_mode == "continuous" and self.action_mode == "discrete":
            raise ValueError("MolmoAct2 action_mode='discrete' cannot run continuous inference.")
        if self.inference_action_mode == "discrete" and self.action_mode == "continuous":
            raise ValueError("MolmoAct2 action_mode='continuous' cannot run discrete inference.")
        if self.train_action_expert_only and self.action_mode != "continuous":
            raise ValueError("MolmoAct2 train_action_expert_only requires action_mode='continuous'.")
        if self.train_action_expert_only and self.enable_lora_vlm:
            raise ValueError("MolmoAct2 train_action_expert_only is incompatible with enable_lora_vlm.")
        if self.enable_lora_action_expert and not self.enable_lora_vlm:
            raise ValueError("MolmoAct2 enable_lora_action_expert requires enable_lora_vlm.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.expected_max_action_dim != 32:
            raise ValueError("MolmoAct2 released checkpoints use expected_max_action_dim=32.")
        if self.model_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                f"Unsupported model_dtype={self.model_dtype!r}. Expected 'float32', 'bfloat16', or 'float16'."
            )
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}.")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}.")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}.")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                f"Unsupported lora_bias={self.lora_bias!r}. Expected one of 'none', 'all', or 'lora_only'."
            )
        if self.discrete_loss_token_weighting not in {
            "none",
            "token",
            "root_tokens",
            "root_subsegments",
            "root_subsegments_root_tokens",
        }:
            raise ValueError(
                f"Unsupported discrete_loss_token_weighting={self.discrete_loss_token_weighting!r}."
            )
        if self.discrete_generation_max_steps is not None and self.discrete_generation_max_steps < 1:
            raise ValueError(
                f"discrete_generation_max_steps must be >= 1 or None, got {self.discrete_generation_max_steps}."
            )
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError(f"max_sequence_length must be >= 1 or None, got {self.max_sequence_length}.")

    def inferred_max_sequence_length(
        self,
        *,
        num_images: int | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        action_horizon: int | None = None,
        include_discrete_action: bool | None = None,
    ) -> int:
        if self.max_sequence_length is not None:
            return int(self.max_sequence_length)

        if num_images is None:
            num_images = len(self.image_keys) or len(self.image_features) or MOLMOACT2_DEFAULT_NUM_IMAGES
        if state_dim is None:
            state_feature = self.robot_state_feature
            state_dim = int(state_feature.shape[0]) if state_feature is not None else 0
        if action_dim is None:
            action_feature = self.action_feature
            action_dim = (
                int(action_feature.shape[0]) if action_feature is not None else self.expected_max_action_dim
            )
        if action_horizon is None:
            action_horizon = self.chunk_size
        if include_discrete_action is None:
            include_discrete_action = self.action_mode in {"discrete", "both"}

        return infer_molmoact2_max_sequence_length(
            num_images=int(num_images),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            action_horizon=int(action_horizon),
            include_discrete_action=bool(include_discrete_action),
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return MolmoAct2CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        self.dataset_feature_names = {}
        for key in (ACTION, OBS_STATE):
            feature = features.get(key) if isinstance(features, dict) else None
            if isinstance(feature, dict) and feature.get("names") is not None:
                self.dataset_feature_names[key] = feature["names"]

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(0,))
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(0,))
