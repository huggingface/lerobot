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
    """PI0.5 configuration for recipe-driven text and action supervision."""

    # Recipe / language stack ---------------------------------------------
    recipe_path: str | None = "recipes/subtask_mem.yaml"
    """Recipe path relative to ``src/lerobot/configs/``, or ``None`` for the plain PI0.5 prompt."""

    apply_chat_template: bool = False
    """Whether to apply a tokenizer chat template.

    PaliGemma defaults to plain recipe-rendered prefixes because it is not chat-pretrained.
    """

    # Balance frequent recipe text supervision against the paper's α=10 flow weight.
    text_loss_weight: float = 1.0
    """LM-head cross-entropy weight; ``0`` disables text training."""

    flow_loss_weight: float = 10.0
    """Weight on action-expert flow matching relative to text supervision."""

    # Backbone training ---------------------------------------------------
    unfreeze_lm_head: bool = True
    """Keep PaliGemma's language head trainable for hierarchical inference."""

    # Optional context dropout improves tolerance to missing or stale language state.
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0

    # FAST adds discrete-action CE to the text and flow objectives from paper §III.B-C.
    enable_fast_action_loss: bool = True
    """Add FAST-tokenized action cross-entropy to text CE and flow matching."""

    action_tokenizer_name: str = "physical-intelligence/fast"
    """HF identifier for the FAST action tokenizer."""

    max_action_tokens: int = 256
    """Maximum number of FAST tokens per action chunk."""

    fast_skip_tokens: int = 128
    """Number of low-vocab tokens the FAST tokenizer skips to avoid
    collisions with PaliGemma's text vocabulary."""

    fast_action_loss_weight: float = 1.0
    """Weight on FAST action-token CE relative to continuous-flow supervision."""

    subtask_replan_steps: int = 0
    """Environment steps between subtask generations during evaluation.

    Non-positive values regenerate each action chunk while still refreshing the action prompt every chunk.
    """

    auto_fit_fast_tokenizer: bool = False
    """Fit and cache a dataset-specific FAST tokenizer before training.

    Disabled by default to avoid the extra dataset pass and use the universal tokenizer.
    """

    fast_tokenizer_cache_dir: str = "~/.cache/lerobot/fast_tokenizers"
    """Where fitted FAST tokenizers are stored. ``~`` expands."""

    fast_tokenizer_fit_samples: int = 1024
    """Number of action chunks sampled when fitting FAST."""

    # Knowledge insulation detaches VLM K/V from action-loss gradients (paper §III.B).
    knowledge_insulation: bool = True
    """Block action-loss gradients through VLM keys and values."""

    # Optional training backends. Defaults preserve the eager/SDPA path.
    use_flashrt_adarms: bool = False
    """Use FlashRT adaptive RMSNorm kernels when available."""

    use_compiled_text_ce: bool = False
    """Compile the materialized-logits text and FAST CE path."""

    use_compiled_vision: bool = False
    """Compile the SigLIP tower for no-grad flow and inference passes."""

    use_flex_attention: bool = False
    """Use FlexAttention for amortized KI, with SDPA fallback where unsupported."""

    use_manual_attention: bool = False
    """Use materialized-logits attention for explicitly profiled KI shapes."""

    manual_attention_scope: str = "all"
    """Apply manual attention to all KI queries or only action queries."""

    # Scale language-head updates relative to the base optimizer schedule.
    lm_head_lr_scale: float = 1.0

    # Scale backbone and action-expert optimizer groups independently.
    backbone_lr_scale: float = 1.0
    action_expert_lr_scale: float = 1.0

    # Reuse each VLM prefix across independent denoising draws; 1 restores single-draw flow.
    flow_num_repeats: int = 5

    # Training-time RTC (arXiv:2512.05964). Zero preserves standard flow matching.
    rtc_training_max_delay: int = 0
    """Largest clean action-prefix length sampled during training.

    A value greater than zero enables training-time action conditioning. Each
    flow draw samples a delay uniformly from ``[0, rtc_training_max_delay]``;
    the corresponding action prefix stays clean and is excluded from the loss.
    """

    # PaLM-style z-loss stabilizes large-vocabulary CE; 0 disables it.
    text_ce_z_loss_weight: float = 1e-4

    use_flashrt_fp8_mlp: bool = False
    """Enable calibrated FlashRT FP8 kernels for Gemma and SigLIP MLPs.

    Apply after loading with ``PI052Policy.apply_flashrt_fp8_mlp``; unavailable kernels keep BF16.
    """

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
        if self.text_loss_weight > 0 and self.unfreeze_lm_head:
            self.train_expert_only = False
        if self.flow_num_repeats < 1:
            raise ValueError(f"flow_num_repeats must be >= 1, got {self.flow_num_repeats}")
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )
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
