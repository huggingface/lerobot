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

"""π0.5 v2 (with text head) — reproduction of the π0.5 paper's
hierarchical inference recipe.

Same architecture as the existing ``PI05Policy`` (PaliGemma 2B VLM +
~300M Gemma action expert, joint training with FAST tokens during
pre-train and flow matching during post-train), but with the
PaliGemma ``lm_head`` re-enabled so the same model can be supervised
to predict both:

  * **subtask strings** at the high level (cross-entropy on the LM
    head), and
  * **action chunks** at the low level (flow matching on the
    action-expert tokens).

This is the dual-head co-training pattern from the paper:

    L = H(x, f_θ_text) + α * ‖ω - a - f_θ_action(a_τ, o, ℓ)‖²

with α = 10.0 per § IV.D of arxiv:2504.16054. The π0.5 model splits
inference into a text-prediction step followed by an action-prediction
step, which the multi-rate ``PI052Runtime`` (in
``lerobot.policies.pi052.inference``) drives at separate rates.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi052")
@dataclass
class PI052Config(PI05Config):
    """π0.5 with the PaliGemma LM head re-enabled for subtask prediction.

    Recipe-driven dual-head training: the flow head supervises actions,
    the LM head supervises subtask / plan / memory / VQA text. The
    flow:text loss split is the milder 5:1 (see ``flow_loss_weight``).
    """

    # Recipe / language stack ---------------------------------------------
    recipe_path: str | None = "recipes/subtasks_vqa.yaml"
    """Path (absolute or relative to ``src/lerobot/configs/``) to a
    ``TrainingRecipe`` YAML. Defaults to the canonical Hi-Robot blend
    shipped alongside this policy. Set to ``None`` to disable recipe
    rendering and fall back to π0.5's single-task ``Task: ... Action:``
    prompt path (unannotated datasets keep working that way)."""

    apply_chat_template: bool = False
    """PaliGemma is *not* chat-pretrained — its tokenizer doesn't ship a
    chat template, so we don't apply one. The recipe renderer's output
    is concatenated as a plain prefix + assistant suffix instead,
    mirroring how the π0.5 paper's high-level inference samples text
    auto-regressively after the prefix."""

    # Loss weights --------------------------------------------------------
    # Paper §IV.D uses α=10 between the flow and text terms, assuming
    # text is a rare auxiliary task. With the recipe stack the flow-only
    # `low_level` branch fires on a large share of samples, so α=10
    # swamps the LM head and collapses generation into degenerate
    # repetition. We use the milder 5:1 split here.
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

    # Per-component prompt dropout (Pi0.7 §V.E) ---------------------------
    # Randomly drop non-target context messages so the LM head learns
    # to handle missing /
    # stale plan / memory at inference. Defaults to 0.0 so behaviour
    # is identical until explicitly enabled.
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0

    # FAST discrete-action supervision — paper §III.B-C ------------------
    # When enabled, actions are *also* tokenised via the FAST tokenizer
    # ("physical-intelligence/fast") and supervised with cross-entropy
    # on the PaliGemma LM head — exactly as in the paper's pre-training
    # objective (Eq. 1 mixes FAST CE + flow MSE + subtask CE). The
    # ActionTokenizerProcessorStep is wired into the preprocessor
    # pipeline when this flag is set; the loss is computed in
    # PI052Policy.forward.
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

    # Knowledge insulation — paper §III.B --------------------------------
    # When enabled, gradients from the action expert's flow loss are
    # blocked from flowing back into the VLM's K/V projections. This
    # prevents the action loss from over-fitting the language backbone
    # to robot-specific features. Implemented in ``modeling_pi052`` as
    # a per-instance monkey-patch on ``paligemma_with_expert.forward``
    # that splits queries into VLM and action halves and ``.detach()``-s
    # the VLM K/V tensors used in the action-half's attention.
    knowledge_insulation: bool = False
    """If True, route every transformer layer through the KI
    attention path that blocks action→VLM gradient flow on K/V."""

    # Learning-rate defaults --------------------------------------------
    # pi052 inherits π0.5's openpi-validated optimizer config (peak LR
    # 2.5e-5, cosine→2.5e-6, 1k warmup, AdamW (0.9, 0.95), wd=0.01,
    # grad_clip=1.0). The only place pi052 needs to diverge from pi05
    # is the LM-head LR multiplier: pi05 has no text supervision so the
    # head doesn't get gradients; pi052 always has text supervision
    # (subtask / memory / VQA) via the recipe, and under KI the LM head
    # only sees gradients on ~30–45% of the batch (the text-CE mask
    # share of the recipe). Under aggressive cosine decay this is too
    # weak to keep the head pinned, so it drifts back toward PaliGemma's
    # pretrained ``<loc>`` first-token bias. 5x is the documented fix
    # (see ``PI05Config.lm_head_lr_scale`` docstring); the wiring is
    # already in ``PI05Policy.get_optim_params`` — it splits the LM head
    # + tied ``embed_tokens`` into their own param group while sharing
    # the same cosine lambda, so the 5x ratio is preserved across decay.
    lm_head_lr_scale: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Backbone needs gradients flowing through the text head when
        # we're training it. Override the π0.5 default
        # (``train_expert_only=True``) unless the user explicitly opts
        # out of text training via ``text_loss_weight=0``.
        if self.text_loss_weight > 0 and self.unfreeze_lm_head:
            self.train_expert_only = False
