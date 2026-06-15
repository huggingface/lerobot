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

"""SmolVLA-KI — SmolVLA with π0.5 / Knowledge-Insulation style training.

Extends :class:`SmolVLAConfig` to support a two-objective recipe on the
SmolVLM2-500M backbone:

* the original flow-matching action expert (cross-attending VLM features), and
* an autoregressive FAST action-token loss computed on the VLM's own ``lm_head``
  (injects spatial/action understanding into the backbone),

with a **stop-gradient** ("knowledge insulation") on the VLM features the expert
reads, so the freshly-initialised expert's gradients never corrupt the backbone.

Architecture (Option A, see ``my_docs/smolvla_fast_pretrain_experiment_plan.md``):
keep the **full** VLM (all layers) so its native ``lm_head`` stays in-distribution
for the FAST token loss, but let the action expert cross-attend only the **first
``expert_attend_layers``** layers (early/mid representations, à la SmolVLA's N=L/2).
At inference only those first layers + the expert run, so deploy cost is unchanged.

Experiment arms (set via flags):
* **A0** baseline: ``knowledge_insulation=False, enable_fast_action_loss=False``,
  ``train_expert_only=True`` — identical to stock SmolVLA (freeze VLM, train expert).
* **A1** hard-freeze-after-pretrain: load a FAST-pretrained VLM, ``train_expert_only=True``,
  ``enable_fast_action_loss=False``.
* **A2** KI co-train (recommended): ``knowledge_insulation=True, enable_fast_action_loss=True``,
  ``train_expert_only=False`` (VLM trains, but only from the FAST/text loss).
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_ki")
@dataclass
class SmolVLAKIConfig(SmolVLAConfig):
    # ── Architecture (Option A) ──────────────────────────────────────────────
    # Keep the full VLM stack (do NOT truncate to num_vlm_layers). Required so the
    # native lm_head stays in-distribution for the FAST autoregressive token loss.
    keep_full_vlm: bool = True
    # The action expert cross-attends only the first `expert_attend_layers` VLM
    # layers (early/mid features). VLM layers beyond this run standalone and only
    # feed the lm_head (FAST/text loss). Replaces the meaning of `num_expert_layers`
    # for this policy; must be <= total VLM layers.
    expert_attend_layers: int = 16

    # ── Knowledge insulation (stop-gradient) ─────────────────────────────────
    # Detach the VLM key/value states before the expert cross-attends them, so the
    # flow-matching loss cannot backprop into the VLM backbone (π0.5 KI, §III.B).
    knowledge_insulation: bool = True

    # ── FAST autoregressive action-token loss (Stage 2) ──────────────────────
    enable_fast_action_loss: bool = True
    fast_action_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0
    # FAST tokenizer (DCT + BPE). Either the universal PI tokenizer or a
    # dataset-specific fit (see auto_fit_fast_tokenizer).
    # lerobot's packaged universal FAST tokenizer (DCT+BPE) — loads cleanly with
    # the pinned transformers, unlike the `physical-intelligence/fast` repo whose
    # bundled tokenizer fails slow->fast conversion here. pi0_fast uses this one.
    action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    max_action_tokens: int = 256
    # Offset added to FAST token ids to avoid colliding with the VLM text vocab.
    fast_skip_tokens: int = 128
    # Off by default: the universal tokenizer closely matches a dataset-specific
    # fit (FAST paper) and the fit path re-loads `physical-intelligence/fast`.
    # Flip on once that base loads, to fit on the dataset's action distribution.
    auto_fit_fast_tokenizer: bool = False
    fast_tokenizer_cache_dir: str = "~/.cache/lerobot/fast_tokenizers"
    fast_tokenizer_fit_samples: int = 1024

    def __post_init__(self):
        super().__post_init__()
        if self.keep_full_vlm and not self.load_vlm_weights:
            # A full, in-distribution VLM (with a usable lm_head) only makes sense
            # when its pretrained weights are loaded. Guard the obvious misconfig.
            raise ValueError(
                "SmolVLAKIConfig: keep_full_vlm=True requires load_vlm_weights=True "
                "(the FAST token loss uses the VLM's pretrained lm_head)."
            )
        if self.expert_attend_layers <= 0:
            raise ValueError(
                f"expert_attend_layers must be > 0, got {self.expert_attend_layers}."
            )
        if self.knowledge_insulation and self.train_expert_only:
            # KI means the VLM IS trained (by the FAST/text loss) while insulated
            # from the expert's gradients — that is incompatible with freezing it.
            raise ValueError(
                "knowledge_insulation=True trains the VLM via the FAST/text loss, so "
                "train_expert_only must be False. For a hard-frozen VLM (arm A1) set "
                "knowledge_insulation=False and enable_fast_action_loss=False."
            )
