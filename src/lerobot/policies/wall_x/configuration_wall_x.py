# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("wall_x")
@dataclass
class WallXConfig(PreTrainedConfig):
    """Configuration class for the Wall-X policy.

    Wall-X is based on Qwen2.5-VL with action prediction capabilities using flow matching. It supports
    cross-embodiment robotic control through unified action representations, and multi-modal learning
    with vision, language, and action data.

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
        chunk_size (`int`, *optional*, defaults to 32): The size of the action prediction chunk
            (`action_horizon` in Wall-X terminology).
        n_action_steps (`int`, *optional*, defaults to 32): The number of actions from a predicted
            chunk that are actually queued for execution. Must not exceed `chunk_size`.
        max_action_dim (`int`, *optional*, defaults to 20): Maximum action dimension Wall-X supports;
            shorter actions are zero-padded.
        max_state_dim (`int`, *optional*, defaults to 20): Maximum proprioceptive-state dimension
            Wall-X supports; shorter states are zero-padded.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*): Per-feature-type
            normalization mode; defaults to `IDENTITY` for vision and `MEAN_STD` for state/action.
        pretrained_name_or_path (`str`, *optional*, defaults to `"x-square-robot/wall-oss-flow"`): Hub id
            or local path of the pretrained Wall-X model to load.
        action_tokenizer_path (`str | None`, *optional*, defaults to `"lerobot/fast-action-tokenizer"`): Hub
            id of the FAST action tokenizer, used only when `prediction_mode="fast"`. Forced to `None` in
            `__post_init__` when `prediction_mode` is `"diffusion"`.
        prediction_mode (`str`, *optional*, defaults to `"diffusion"`): Action prediction mode:
            `"diffusion"` (flow matching) or `"fast"` (discrete FAST tokens).
        attn_implementation (`str`, *optional*, defaults to `"eager"`): Attention backend for the
            language/action-token model. Only `"eager"` is currently supported, since Wall-X's
            bidirectional action-token islands require an explicit attention mask.
        vision_attn_implementation (`str`, *optional*, defaults to `"auto"`): Attention backend for
            vision, independent from the text action-token mask: `"auto"` (packed variable-length
            attention when supported, otherwise per-chunk SDPA), `"sdpa"`, or `"varlen"`.
        optimizer_lr (`float`, *optional*, defaults to 2e-05): AdamW learning rate.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`): AdamW betas.
        optimizer_eps (`float`, *optional*, defaults to 1e-08): AdamW epsilon.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.01): AdamW weight decay.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 1.0): Gradient clipping norm.
        scheduler_warmup_steps (`int`, *optional*, defaults to 1000): Number of warmup steps for the
            cosine-decay-with-warmup scheduler.
        scheduler_decay_steps (`int`, *optional*, defaults to 100000): Number of decay steps for the
            scheduler.
        scheduler_decay_lr (`float`, *optional*, defaults to 1e-06): Final learning rate at the end of
            the decay schedule.
    """

    # ==================== Input / Output Structure ====================
    n_obs_steps: int = 1
    chunk_size: int = 32  # action_horizon in wall-x
    n_action_steps: int = 32

    # Action dimension - wall-x uses 20
    max_action_dim: int = 20
    max_state_dim: int = 20  # For proprioception

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==================== Action Prediction ====================
    # Pretrained model paths
    pretrained_name_or_path: str = "x-square-robot/wall-oss-flow"

    # Tokenizer settings
    action_tokenizer_path: str | None = "lerobot/fast-action-tokenizer"

    # Action prediction mode: "diffusion" or "fast"
    prediction_mode: str = "diffusion"

    # Wall-X's bidirectional action-token islands currently require eager attention.
    attn_implementation: str = "eager"

    # Vision attention is independent from the text action-token mask. ``auto`` uses
    # PyTorch's packed variable-length attention when the runtime supports it and
    # otherwise falls back to the native per-chunk SDPA implementation.
    vision_attn_implementation: str = "auto"

    # ==================== Optimizer Presets ====================
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        """Validate cross-field constraints and derive `use_fast_tokenizer` from `prediction_mode`.

        Raises:
            ValueError: If `n_action_steps` exceeds `chunk_size`, if `prediction_mode` is not
                `"diffusion"` or `"fast"`, if `attn_implementation` is not `"eager"`, or if
                `vision_attn_implementation` is not one of `"auto"`, `"sdpa"`, or `"varlen"`.
        """
        super().__post_init__()

        # Input validation
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.prediction_mode not in ["diffusion", "fast"]:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

        if self.attn_implementation != "eager":
            raise ValueError(
                "Wall-X currently supports only attn_implementation='eager' because its "
                "bidirectional action-token islands require an explicit attention mask."
            )

        if self.vision_attn_implementation not in {"auto", "sdpa", "varlen"}:
            raise ValueError(
                "vision_attn_implementation must be one of 'auto', 'sdpa', or 'varlen', got "
                f"{self.vision_attn_implementation!r}"
            )

        # Assign use_fast_tokenizer based on prediction_mode
        if self.prediction_mode == "fast":
            self.use_fast_tokenizer = True
        elif self.prediction_mode == "diffusion":
            self.use_fast_tokenizer = False
            self.action_tokenizer_path = None  # disable action tokenizer for diffusion mode
        else:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Wall-X policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
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

    def get_scheduler_preset(self):
        """Return the cosine-decay-with-warmup scheduler configuration built from the `scheduler_*` fields."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        """Return indices for delta observations (None for Wall-X)."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """Return indices for delta actions."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for Wall-X)."""
        return None
