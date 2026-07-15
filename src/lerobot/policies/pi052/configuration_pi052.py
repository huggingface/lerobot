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

"""PI0.5 with its PaliGemma text head enabled for hierarchical language/action training.

The runtime generates high-level text and low-level flow-matched actions at separate rates.
"""

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
    """``TrainingRecipe`` YAML path (absolute or relative to
    ``src/lerobot/configs/``). ``None`` disables recipe rendering — unannotated
    datasets fall back to π0.5's plain ``Task: ... Action:`` prompt."""

    apply_chat_template: bool = False
    """PaliGemma is *not* chat-pretrained — its tokenizer doesn't ship a
    chat template, so we don't apply one. The recipe renderer's output
    is concatenated as a plain prefix + assistant suffix instead,
    mirroring how the π0.5 paper's high-level inference samples text
    auto-regressively after the prefix."""

    # Balance frequent recipe text supervision against the paper's α=10 flow weight.
    text_loss_weight: float = 1.0
    """Weight on the LM-head cross-entropy term. Set to ``0`` to disable
    text training entirely (reverts to flow-only / π0.5 behaviour)."""

    flow_loss_weight: float = 5.0
    """Weight on the action-expert flow-matching term. ``5.0`` — a milder
    flow:text split than the paper's α=10, since the flow-only
    ``low_level`` recipe already gives the action expert frequent
    gradient. Lower it further if the LM head still underfits."""

    # Backbone training ---------------------------------------------------
    unfreeze_lm_head: bool = True
    """Whether to keep the PaliGemma ``lm_head`` unfrozen for fine-tuning.
    The existing ``PI05Policy`` zeroes / freezes the head on load
    because it never reads from it. Must be ``True`` for π0.5-style
    hierarchical inference."""

    # Optional context dropout improves tolerance to missing or stale language state.
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0

    # FAST adds discrete-action CE to the text and flow objectives from paper §III.B-C.
    enable_fast_action_loss: bool = True
    """If True, tokenise actions with the FAST tokenizer and add a
    cross-entropy loss on the LM head. On by default to match the
    π0.5 paper's three-loss objective (text CE + FAST CE + flow MSE,
    §III.B-C Eq. 1). Set to False if you only want the
    post-training-style flow + text recipe."""

    action_tokenizer_name: str = "physical-intelligence/fast"
    """HF identifier for the FAST action tokenizer."""

    max_action_tokens: int = 256
    """Maximum number of FAST tokens per action chunk."""

    fast_skip_tokens: int = 128
    """Number of low-vocab tokens the FAST tokenizer skips to avoid
    collisions with PaliGemma's text vocabulary."""

    fast_action_loss_weight: float = 1.0
    """Weight on the FAST-action-token CE loss. Paper §III.C uses 1.0."""

    subtask_replan_steps: int = 0
    """Eval-only: regenerate the low-level subtask every this many env steps.
    ``<=0`` (default) regenerates on every action chunk (i.e. every
    ``n_action_steps`` steps). Set e.g. to 20 (≈1s at 20 fps) to hold the
    subtask across several action chunks, closer to training's subtask
    intervals; the action prompt is still rebuilt with the current state each
    chunk."""

    auto_fit_fast_tokenizer: bool = False
    """If True, the processor factory checks ``fast_tokenizer_cache_dir``
    for a previously-fitted tokenizer keyed on ``(dataset_repo_id,
    base_tokenizer_name, fit_samples)``. On cache miss, it loads
    ``action_tokenizer_name`` as a base, samples
    ``fast_tokenizer_fit_samples`` action chunks from the dataset, runs
    ``.fit()``, saves the result, and uses *that* fitted path as the
    actual tokenizer. Pertsch et al. 2025 (FAST paper [64], π0.5 §III.C)
    explicitly recommend per-dataset fitting for best compression.

    Off by default because the fit requires a separate pre-training
    pass over the dataset (~1-2 min on a medium dataset) and depends
    on the FAST tokenizer snapshot having a ``.fit()`` method. Opt in
    when you want paper-faithful compression; leave off to fall back
    on the universal ``physical-intelligence/fast`` codebook."""

    fast_tokenizer_cache_dir: str = "~/.cache/lerobot/fast_tokenizers"
    """Where fitted FAST tokenizers are stored. ``~`` expands."""

    fast_tokenizer_fit_samples: int = 1024
    """Number of action chunks to sample for the fit. The FAST paper uses
    a few thousand; 1024 is a reasonable default for medium datasets."""

    # Knowledge insulation detaches VLM K/V from action-loss gradients (paper §III.B).
    knowledge_insulation: bool = True
    """If True, route every transformer layer through the KI
    attention path that blocks action→VLM gradient flow on K/V."""

    # Boost sparse text-head updates while retaining PI0.5's optimizer schedule.
    lm_head_lr_scale: float = 5.0

    # Scale pretrained backbone and new action-expert groups independently; 1.0 preserves legacy behavior.
    backbone_lr_scale: float = 1.0
    action_expert_lr_scale: float = 1.0

    # Reuse each VLM prefix across independent denoising draws; 1 restores single-draw flow.
    flow_num_repeats: int = 5

    # PaLM-style z-loss stabilizes large-vocabulary CE; 0 disables it.
    text_ce_z_loss_weight: float = 1e-4

    # Liger patches are optional, process-global, and idempotent.
    use_flashrt_fp8_mlp: bool = False
    """Opt-in: swap every Gemma GeGLU MLP (action expert + prefix LM) and the
    SigLIP vision MLP to FlashRT fused FP8 kernels (Hugging Face Kernel Hub
    ``flashrt/*``). The swap needs a one-time activation calibration on a real
    observation, so it is applied explicitly via
    ``PI052Policy.apply_flashrt_fp8_mlp(batch)`` after loading (not at build).
    Degrades gracefully to BF16 if ``kernels`` / the FlashRT packages are
    missing. Default off keeps behaviour identical to the BF16 path."""

    use_flex_attention: bool = False
    """Accepted for checkpoint-config compatibility only — no-op in this branch.
    Newer training runs serialize ``use_flex_attention`` into ``config.json`` to
    select the FlexAttention kernel at train time; this branch's attention path
    is SDPA/eager, which is mathematically equivalent (same softmax attention,
    different kernel), so inference/eval results are unchanged. Retained so those
    checkpoints load instead of raising ``DecodingError: The fields
    use_flex_attention are not valid for PI052Config``."""

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
        # Override PI0.5's expert-only default when training text.
        if self.text_loss_weight > 0 and self.unfreeze_lm_head:
            self.train_expert_only = False
        if self.flow_num_repeats < 1:
            raise ValueError(f"flow_num_repeats must be >= 1, got {self.flow_num_repeats}")
