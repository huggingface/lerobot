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
step, which mirrors what ``SmolVLA2Runtime`` already does on a
SmolVLM2 backbone.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi052")
@dataclass
class PI052Config(PI05Config):
    """π0.5 with the PaliGemma LM head re-enabled for subtask prediction.

    See ``SmolVLA2Config`` for the analogous SmolVLM2-backed dual-head
    config. Same recipe-driven training surface; the only differences
    are which backbone the policy uses (PaliGemma here vs SmolVLM2
    there) and the default loss-weight scale (paper §IV.D uses
    ``α=10`` between the two heads, which we encode as
    ``flow_loss_weight=10, text_loss_weight=1``).
    """

    # Recipe / language stack ---------------------------------------------
    recipe_path: str | None = "recipes/pi052_hirobot.yaml"
    """Path (absolute or relative to ``src/lerobot/configs/``) to a
    ``TrainingRecipe`` YAML. Defaults to the canonical Hi-Robot blend
    shipped alongside this policy. Set to ``None`` to disable recipe
    rendering and fall back to π0.5's single-task ``Task: ... Action:``
    prompt path (unannotated datasets keep working that way)."""

    apply_chat_template: bool = False
    """PaliGemma is *not* chat-pretrained — its tokenizer doesn't ship a
    chat template. So unlike SmolVLA2 we don't apply one. The recipe
    renderer's output is concatenated as a plain prefix + assistant
    suffix instead, mirroring how the π0.5 paper's high-level inference
    samples text auto-regressively after the prefix."""

    # Loss weights --------------------------------------------------------
    # Paper §IV.D: total = H(text) + α * MSE(flow), α = 10. We split
    # the same total into two configurable knobs so individual scaling
    # is recoverable.
    text_loss_weight: float = 1.0
    """Weight on the LM-head cross-entropy term. Set to ``0`` to disable
    text training entirely (reverts to flow-only / π0.5 behaviour)."""

    flow_loss_weight: float = 10.0
    """Weight on the action-expert flow-matching term. Default ``10.0``
    matches the paper's α."""

    # Backbone training ---------------------------------------------------
    unfreeze_lm_head: bool = True
    """Whether to keep the PaliGemma ``lm_head`` unfrozen for fine-tuning.
    The existing ``PI05Policy`` zeroes / freezes the head on load
    because it never reads from it. Must be ``True`` for π0.5-style
    hierarchical inference."""

    # Per-component prompt dropout (Pi0.7 §V.E) ---------------------------
    # Same regulariser surface as SmolVLA2: randomly drop non-target
    # context messages so the LM head learns to handle missing /
    # stale plan / memory at inference. Defaults to 0.0 so behaviour
    # is identical until explicitly enabled.
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Backbone needs gradients flowing through the text head when
        # we're training it. Override the π0.5 default
        # (``train_expert_only=True``) unless the user explicitly opts
        # out of text training via ``text_loss_weight=0``.
        if self.text_loss_weight > 0 and self.unfreeze_lm_head:
            self.train_expert_only = False
