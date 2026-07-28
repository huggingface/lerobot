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

"""PI0.5 with hierarchical text generation and flow-matched actions."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

from ..pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi052")
@dataclass
class PI052Config(PI05Config):
    """PI0.5 with recipe-driven text and action supervision."""

    # Recipe / language stack ---------------------------------------------
    recipe_path: str | None = "recipes/subtask_mem.yaml"
    """Recipe path, or ``None`` for the plain PI0.5 prompt."""

    apply_chat_template: bool = False
    """Apply the tokenizer's chat template."""

    # Balance frequent recipe text supervision against the paper's α=10 flow weight.
    text_loss_weight: float = 1.0
    """Text cross-entropy weight; ``0`` disables it."""

    flow_loss_weight: float = 10.0
    """Flow-matching loss weight."""

    # Backbone training ---------------------------------------------------
    unfreeze_lm_head: bool = True
    """Train PaliGemma's language head."""

    # Optional context dropout improves tolerance to missing or stale language state.
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0

    # FAST adds discrete-action CE to the text and flow objectives from paper §III.B-C.
    enable_fast_action_loss: bool = True
    """Add FAST action-token cross-entropy."""

    action_tokenizer_name: str = "physical-intelligence/fast"
    """FAST tokenizer identifier."""

    max_action_tokens: int = 256
    """Maximum FAST tokens per action chunk."""

    fast_skip_tokens: int = 1152
    """Reserved vocabulary IDs skipped by FAST token mapping."""

    fast_action_loss_weight: float = 1.0
    """FAST action-token loss weight."""

    subtask_replan_steps: int = 0
    """Steps between subtask generations; non-positive replans every chunk."""

    joint_subtask_conditioning: bool = False
    """Condition actions on the task and generated subtask."""

    auto_fit_fast_tokenizer: bool = False
    """Fit and cache a dataset-specific FAST tokenizer."""

    fast_tokenizer_cache_dir: str = "~/.cache/lerobot/fast_tokenizers"
    """Cache directory for fitted FAST tokenizers."""

    fast_tokenizer_fit_samples: int = 1024
    """Action chunks sampled for tokenizer fitting."""

    fast_tokenizer_validation_samples: int = 256
    """Held-out chunks used for tokenizer validation."""

    fast_tokenizer_max_reconstruction_rmse: float = 0.10
    """Maximum validation reconstruction RMSE."""

    fast_tokenizer_max_dim_rmse: float = 0.20
    """Maximum per-dimension validation RMSE."""

    # Knowledge insulation detaches VLM K/V from action-loss gradients (paper §III.B).
    knowledge_insulation: bool = True
    """Detach VLM keys and values from action-loss gradients."""

    # Optional training backends. Defaults preserve the eager/SDPA path.
    use_flashrt_adarms: bool = False
    """Use FlashRT adaptive RMSNorm kernels."""

    use_compiled_text_ce: bool = False
    """Compile text and FAST cross-entropy."""

    use_compiled_vision: bool = False
    """Compile the SigLIP vision tower."""

    use_flex_attention: bool = False
    """Use FlexAttention for knowledge insulation."""

    use_manual_attention: bool = False
    """Use manual attention for profiled KI shapes."""

    manual_attention_scope: str = "all"
    """Manual-attention scope: ``all`` or ``action``."""

    # Scale language-head updates relative to the base optimizer schedule.
    lm_head_lr_scale: float = 1.0

    # Scale backbone and action-expert optimizer groups independently.
    backbone_lr_scale: float = 1.0
    action_expert_lr_scale: float = 1.0

    # Reuse each VLM prefix across independent denoising draws; 1 restores single-draw flow.
    flow_num_repeats: int = 5

    # PaLM-style z-loss stabilizes large-vocabulary CE; 0 disables it.
    text_ce_z_loss_weight: float = 1e-4

    use_flashrt_fp8_mlp: bool = False
    """Use calibrated FlashRT FP8 MLP kernels."""

    # Keep serialized PI052 AdamW options local because PI05Config lacks them.
    optimizer_foreach: bool | None = False
    optimizer_fused: bool | None = True

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            foreach=self.optimizer_foreach,
            fused=self.optimizer_fused,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.enable_fast_action_loss and not self.recipe_path:
            raise ValueError("PI052 FAST action loss requires recipe_path to build action supervision.")
        if self.text_loss_weight > 0 and self.unfreeze_lm_head:
            self.train_expert_only = False
        if self.flow_num_repeats < 1:
            raise ValueError(f"flow_num_repeats must be >= 1, got {self.flow_num_repeats}")
        if self.fast_tokenizer_validation_samples < 1:
            raise ValueError("fast_tokenizer_validation_samples must be >= 1")
        if self.fast_tokenizer_max_reconstruction_rmse <= 0 or self.fast_tokenizer_max_dim_rmse <= 0:
            raise ValueError("FAST tokenizer reconstruction thresholds must be positive")
        if self.manual_attention_scope not in {"all", "action"}:
            raise ValueError(
                f"manual_attention_scope must be 'all' or 'action', got {self.manual_attention_scope!r}"
            )
        if self.use_flex_attention and self.use_manual_attention:
            raise ValueError("use_flex_attention and use_manual_attention are mutually exclusive")
        if self.use_flex_attention and self.flow_num_repeats == 1:
            raise ValueError("use_flex_attention requires flow_num_repeats > 1")
        if not self.knowledge_insulation and (
            self.use_flex_attention or self.use_manual_attention or self.use_flashrt_adarms
        ):
            raise ValueError("KI attention and AdaRMS optimizations require knowledge_insulation=True")
