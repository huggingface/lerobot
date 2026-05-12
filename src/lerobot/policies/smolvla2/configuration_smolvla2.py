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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla2")
@dataclass
class SmolVLA2Config(SmolVLAConfig):
    """SmolVLA2 — SmolVLA with the underlying SmolVLM language head re-enabled.

    SmolVLA strips the LM head from the SmolVLM backbone because it only
    needs flow-matching action prediction. SmolVLA2 keeps the LM head so the
    same model can train on:

      * **action-only sub-recipes** (e.g. ``low_level_execution``) — flow loss
        on the action expert, same as SmolVLA. ``predict_actions=True``.
      * **text-only sub-recipes** (e.g. ``memory_update`` / ``ask_vqa`` /
        ``user_interjection_response`` / ``high_level_subtask``) — cross-
        entropy loss on the LM head over the recipe's target message tokens.
        Skips the flow head entirely. ``predict_actions=False``.
      * **mixed sub-recipes** — both heads run, losses summed (weighted).

    The split is controlled by ``predict_actions = bool(targets_by_stream
    .get("low_level"))`` per the Pi0.5 convention in the steerable
    annotation plan (Section I.7), implemented inside the processor /
    forward path. Recipes drive it via ``stream`` + ``target`` metadata.

    Compared to ``SmolVLAConfig`` this adds:

    - ``recipe_path``: path to a ``TrainingRecipe`` YAML (loaded by the
      train script). When ``None``, SmolVLA2 falls back to the SmolVLA
      task-only path so unannotated datasets still work.
    - ``text_loss_weight`` / ``flow_loss_weight``: relative weights when
      both losses are active in a single sample.
    - ``unfreeze_lm_head``: must be ``True`` for the text head to learn —
      SmolVLA freezes ``lm_head`` to "avoid unused params issues" and we
      need to undo that for SmolVLA2.
    - ``train_expert_only=False`` by default, since the VLM body now also
      participates in text-target gradients.
    """

    # Recipe / language stack ---------------------------------------------
    recipe_path: str | None = "recipes/smolvla2_hirobot.yaml"
    """Path (absolute or relative to ``src/lerobot/configs/``) to a
    ``TrainingRecipe`` YAML. The default points at the canonical Hi Robot
    blend shipped alongside SmolVLA2. Set to ``None`` to disable recipe
    rendering and fall back to SmolVLA's single-task prompt path
    (unannotated datasets keep working that way)."""

    apply_chat_template: bool = True
    """Apply the SmolVLM tokenizer's chat template to the rendered messages
    before tokenizing. SmolVLM's backbone is chat-pretrained, so this
    matches its training distribution."""

    # Loss weights --------------------------------------------------------
    text_loss_weight: float = 1.0
    """Weight on the LM-head cross-entropy term. Set to ``0`` to disable
    text training entirely (reverts to flow-only / SmolVLA behaviour)."""

    flow_loss_weight: float = 1.0
    """Weight on the action-expert flow-matching term."""

    # Backbone training ---------------------------------------------------
    unfreeze_lm_head: bool = True
    """Whether to unfreeze the SmolVLM ``lm_head`` (and the immediately
    preceding norm + last text-model layer that SmolVLA freezes). Must be
    ``True`` for the text head to learn. Setting this to ``False``
    effectively reduces SmolVLA2 back to SmolVLA's flow-only training,
    which is occasionally useful for ablations."""

    # Per-component prompt dropout (Pi0.7 §V.E) ---------------------------
    # At training, randomly drop non-target context messages whose
    # content was substituted from the named recipe binding. Forces
    # the model to handle missing context — directly attacks the
    # memorisation collapse where a stale or missing plan/memory at
    # inference puts the prompt out-of-distribution and the LM head
    # falls back to dominant-mode fragments. All default to 0.0 so
    # behaviour is identical until explicitly enabled.
    plan_dropout_prob: float = 0.0
    """Drop messages whose content starts with ``Plan:`` or ``Previous plan``
    with this probability per sample."""
    memory_dropout_prob: float = 0.0
    """Drop messages whose content starts with ``Memory:`` or ``Previous memory``
    with this probability per sample."""
    subtask_dropout_prob: float = 0.0
    """Drop messages whose content starts with ``Current subtask`` or
    ``Completed subtask`` with this probability per sample."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Backbone needs gradients flowing through its text path when the
        # LM head is producing supervised text. Override the SmolVLA
        # default (`train_expert_only=True`) unless the user explicitly
        # opts out of text training via `text_loss_weight=0`.
        if self.text_loss_weight > 0 and self.unfreeze_lm_head:
            # The user can still flip this back via CLI; this only
            # changes the *default* when SmolVLA2 is actually training a
            # text head.
            self.train_expert_only = False
