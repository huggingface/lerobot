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


@PreTrainedConfig.register_subclass("lingbot_vla")
@dataclass
class LingbotVLAConfig(PreTrainedConfig):
    """
    Configuration class for the LingBot-VLA policy.

    LingBot-VLA is a Qwen2.5-VL based vision-language-action model that predicts
    action chunks via flow matching. It supports cross-embodiment control through a
    unified 75-dim state/action representation and an optional depth-distillation branch.
    """

    # ==================== Input / Output Structure ====================
    n_obs_steps: int = 1
    chunk_size: int = 50  # action_horizon in LingBot-VLA
    n_action_steps: int = 50

    # Unified cross-embodiment slots (padded to these dims).
    max_action_dim: int = 75
    max_state_dim: int = 75

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==================== Pretrained backbone ====================
    pretrained_name_or_path: str = "robbyant/lingbot-vla-4b"
    tokenizer_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer_max_length: int = 72

    # Image resize target (width, height) applied before Qwen2.5-VL patchification.
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Number of flow-matching denoising steps at inference.
    num_steps: int = 10

    # ==================== Optional depth distillation branch ====================
    # Only used by the LingBot-VLA-4B-Depth variant.
    use_depth: bool = False
    num_task_tokens: int = 8

    # ==================== Modeling internals (FlowMatching / dual-stream expert) ====================
    # Attention used inside the vendored dual-stream model, one of {"eager", "fa2", "flex"}.
    # "eager" is the safe default and required where flash-attn is unavailable.
    attention_implementation: str = "eager"
    use_cache: bool = True
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    # 0 keeps the Qwen2.5-VL vocab as-is (no resize).
    vocab_size: int = 0
    use_lm_head: bool = False
    loss_type: str = "fm"  # flow-matching MSE loss ("L1_fm" for L1)
    # Empty dict disables the depth-alignment heads (no-depth checkpoint).
    align_params: dict = field(default_factory=dict)

    # Adaptive layernorm settings for the action expert. The 4B checkpoint was
    # trained with adanorm_time=True, which replaces the expert RMSNorms with
    # AdaNorm modules (gamma/beta nn.Linear from the time embedding). The rest
    # match the LingBot training defaults (arguments.py).
    adanorm_time: bool = True
    split_gate_liner: bool = False
    nosplit_gate_liner: bool = False
    separate_time_proj: bool = False
    final_norm_adanorm: bool = False
    norm_qkv: bool = False

    # ==================== Optimizer / Scheduler Presets ====================
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 40000
    scheduler_decay_lr: float = 5e-5  # constant lr schedule (decay_lr == peak_lr)

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"attention_implementation must be one of 'eager', 'fa2', 'flex', "
                f"got {self.attention_implementation}"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "LingBot-VLA policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
