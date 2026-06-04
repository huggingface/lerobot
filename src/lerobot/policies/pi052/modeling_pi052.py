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

"""π0.5 v2 policy — dual-head training & hierarchical inference.

A thin subclass of :class:`PI05Policy` that:

* keeps the PaliGemma ``lm_head`` unfrozen during fine-tuning
  (``PI05Policy`` zeroes / freezes it because it never reads from
  the head; ``PI052Config.unfreeze_lm_head`` flips that),
* adds a ``text_loss`` term computed via cross-entropy on
  ``text_labels`` (built by ``PI052TextTokenizerStep``),
* adds :meth:`select_message` for AR text generation at inference
  (the high-level step in the π0.5 paper's two-stage inference loop),
* combines both losses in :meth:`forward` per Eq. (1) of the paper:

      L = H(x, f_θ_text) + α * ‖ω - a - f_θ_action(...)‖²

  with α controllable via ``config.flow_loss_weight``.

The multi-rate inference runtime in ``lerobot.policies.pi052.inference``
(driven by the ``lerobot-pi052-runtime`` CLI) sits on top of this:
``predict_action_chunk`` for the action expert and ``select_message``
for the LM head.
"""

from __future__ import annotations

import builtins
import logging
import types
from collections import deque
from pathlib import Path
from typing import Any, Unpack

import torch
from torch import Tensor
from torch.nn import functional as F

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi052 import PI052Config
from .pi05_backbone import ActionSelectKwargs, PI05Pytorch, pad_vector, resize_with_pad_torch

logger = logging.getLogger(__name__)


# FAST action-token vocab size (``lerobot/fast-action-tokenizer``). The
# tokenizer maps a FAST BPE id ``t`` to the PaliGemma vocab id
# ``vocab_size - 1 - fast_skip_tokens - t`` (see ``TokenizerProcessorStep``),
# so action tokens occupy the top ``_FAST_ACTION_VOCAB_SIZE`` ids below the
# ``fast_skip_tokens`` margin. The upper part collides with the reserved
# ``<loc>`` block; the lower part sits just under it and otherwise leaks into
# generated text as high-codepoint gibberish (the action-trained LM head puts
# heavy mass on these ids), so ``select_message`` masks it.
_FAST_ACTION_VOCAB_SIZE = 2048


_HF_KERNELS_ENABLED = False


def _enable_hf_kernels() -> None:
    """Patch PaliGemma / Gemma / Siglip layers with Liger fused kernels.

    Must run BEFORE ``PaliGemmaWithExpertModel`` is built — the patch
    replaces classes in ``transformers.models.{gemma,paligemma,siglip}``,
    so any model constructed after this picks up the fused forwards.
    Idempotent (process-global). ``cross_entropy`` / ``fused_linear_*``
    are deliberately skipped — pi052 uses ``F.cross_entropy`` directly
    and never traverses ``PaliGemmaForConditionalGeneration.forward``,
    so those Liger paths wouldn't fire without model-code changes.
    See bench job 22161421 in ``examples/benchmark/`` for the numbers.
    """
    global _HF_KERNELS_ENABLED
    if _HF_KERNELS_ENABLED:
        return
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_paligemma  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "PI052: liger-kernel is not installed; skipping fused Triton "
            "kernels (rope/geglu/layer_norm). Install with "
            "``pip install liger-kernel`` for a ~4.5%% step speedup."
        )
        return
    apply_liger_kernel_to_paligemma(
        rope=True,
        geglu=True,
        layer_norm=True,
        rms_norm=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    _HF_KERNELS_ENABLED = True
    logger.info("PI052: HF kernels (Liger) enabled — rope, geglu, layer_norm fused.")


# ----------------------------------------------------------------------
# Loss helpers (shared between fused and prefix-only paths)
# ----------------------------------------------------------------------


def _mask_per_sample(per_sample: Tensor, predict_actions_t: Tensor | None) -> Tensor:
    """Mean over samples where ``predict_actions_t`` is True, else over all."""
    if predict_actions_t is None:
        return per_sample.mean()
    mask = predict_actions_t.to(per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


def _shifted_lin_ce(
    hidden: Tensor,
    lm_head_weight: Tensor,
    labels: Tensor,
    z_loss_weight: float = 0.0,
) -> Tensor:
    """Liger-fused (hidden @ W.T → softmax → CE) on shifted labels.

    Replaces the explicit ``lm_head(hidden) → F.cross_entropy(...)``
    pair with Liger's ``LigerFusedLinearCrossEntropyLoss``: the full
    ``(B, T, V)`` logits tensor is never materialised — the kernel
    chunks over the (B*T) axis, computing matmul + logsumexp + CE
    in fused Triton blocks. On a 257k-vocab head this saves ~10 GB
    of activation memory per CE branch and ~30 % step time vs the
    eager ``F.cross_entropy`` path.

    Semantics:
      * Shift convention identical to the eager version — hidden at
        position ``t`` predicts label at ``t+1``; ``ignore_index=-100``.
      * No ``.any().item()`` sync — Liger returns 0.0 cleanly when
        every label is ignored.
      * ``z_loss_weight`` maps directly to Liger's ``lse_square_scale``
        (same ``z²·w`` formula on per-position logsumexp). Setting it
        to 0 disables the z-loss term at zero cost.
    """
    # Liger is imported lazily so the module still imports on machines
    # without liger-kernel — the call site only fires from the training
    # forward, which always pulls in the kernel.
    from liger_kernel.transformers.fused_linear_cross_entropy import (  # noqa: PLC0415
        LigerFusedLinearCrossEntropyLoss,
    )

    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().long()
    B, T_1, H = shift_hidden.shape
    flat_hidden = shift_hidden.reshape(B * T_1, H)
    flat_labels = shift_labels.reshape(B * T_1)
    # Match the dtype the eager path used: cast hidden to the lm_head's
    # weight dtype so bf16 weights see bf16 activations.
    flat_hidden = flat_hidden.to(lm_head_weight.dtype)
    loss_fn = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        lse_square_scale=float(z_loss_weight),
        reduction="mean",
    )
    return loss_fn(lm_head_weight, flat_hidden, flat_labels)


def _mark_target_span_causal(
    prefix_att_masks: Tensor, text_labels: Tensor, lang_start: int, lang_end: int
) -> Tensor:
    """Make the supervised text-target span causally masked.

    ``embed_prefix`` lays the PaliGemma prefix out as ``[images,
    language]`` with the language block flagged ``att=0`` — which
    ``make_att_2d_masks`` turns into one fully *bidirectional* block.
    A supervised target token's hidden state then attends to the very
    tokens it is trained to predict, so the text cross-entropy
    degenerates into a copy task (loss → ~0) and the LM head never
    learns causal next-token prediction. At inference ``select_message``
    decodes autoregressively (causally) and the head collapses to
    repeated/garbage tokens.

    Fix: set ``att=1`` on the language positions that are supervised
    targets (``text_labels != -100``). Under ``make_att_2d_masks``'s
    cumulative-block rule each target token then attends bidirectionally
    to images + the user prompt and causally to *earlier* targets only —
    genuine next-token prediction, matching inference. Non-target
    language (the user prompt, the flow-only ``low_level`` subtask) stays
    ``att=0`` / bidirectional. The action expert / FAST tokens are
    unaffected: they sit at a strictly higher cumsum and still attend to
    every prefix token.
    """
    att = prefix_att_masks.clone()
    n = min(text_labels.shape[1], lang_end - lang_start)
    if n <= 0:
        return att
    target = text_labels[:, :n] != -100  # (B, n) bool
    seg = att[:, lang_start : lang_start + n].bool()
    att[:, lang_start : lang_start + n] = seg | target
    return att


def _fast_lin_ce(
    hidden: Tensor,
    lm_head_weight: Tensor,
    action_tokens: Tensor,
    action_code_mask: Tensor,
    predict_actions_t: Tensor | None,
) -> Tensor:
    """Liger-fused FAST action-code CE with span masking + sample gating.

    Mirrors ``_shifted_lin_ce`` but with FAST-specific masking: only
    the discrete action-code positions (``action_code_mask``) are
    supervised, and samples whose recipe sets ``predict_actions=False``
    get all code positions masked. Masked positions are folded into
    Liger's ``ignore_index=-100`` so the kernel skips them without
    a CPU-side gather (which would synchronise + break CUDA graphs).
    """
    from liger_kernel.transformers.fused_linear_cross_entropy import (  # noqa: PLC0415
        LigerFusedLinearCrossEntropyLoss,
    )

    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_targets = action_tokens[:, 1:].contiguous().long()
    shift_valid = action_code_mask[:, 1:].contiguous().bool()
    if predict_actions_t is not None:
        sample_mask = predict_actions_t[:, None].expand_as(shift_valid)
        shift_valid = shift_valid & sample_mask
    # Fold the boolean mask into the target via ignore_index. No
    # ``.any().item()`` sync — Liger returns 0.0 when every position
    # is ignored, preserving graph capture for CUDA graphs.
    shift_targets = torch.where(
        shift_valid, shift_targets, torch.full_like(shift_targets, -100)
    )

    B, T_1, H = shift_hidden.shape
    flat_hidden = shift_hidden.reshape(B * T_1, H).to(lm_head_weight.dtype)
    flat_labels = shift_targets.reshape(B * T_1)

    loss_fn = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
    )
    return loss_fn(lm_head_weight, flat_hidden, flat_labels)


# ----------------------------------------------------------------------
# Knowledge insulation — ported from pi05_full (branch ``feat/add-pi05``)
# ----------------------------------------------------------------------
#
# Per-layer attention that splits the queries into VLM and action
# parts, computing attention for action queries with .detach()'d VLM
# K/V so the action loss's gradient cannot flow back into the VLM's K
# and V projections. Forward output is bit-equivalent to the standard
# layer; backward differs only on the path action_loss → VLM K/V.

def _compute_layer_ki(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    from transformers.models.gemma import modeling_gemma  # noqa: PLC0415

    # ``_gated_residual`` is a lerobot helper (adaRMSNorm gated residual),
    # not part of HF's ``modeling_gemma``. pi05's own layer code imports
    # it from ``pi_gemma`` — mirror that here.
    from ..pi_gemma import _gated_residual  # noqa: PLC0415

    models = [paligemma.model.language_model, gemma_expert.model]
    query_states, key_states, value_states, gates = [], [], [], []

    vlm_len = inputs_embeds[0].shape[1]

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(q)
        key_states.append(k)
        value_states.append(v)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling

    # Split queries / K / V at the VLM-vs-action boundary.
    Q_vlm = query_states[:, :, :vlm_len, :]
    Q_action = query_states[:, :, vlm_len:, :]
    K_vlm = key_states[:, :, :vlm_len, :]
    K_action = key_states[:, :, vlm_len:, :]
    V_vlm = value_states[:, :, :vlm_len, :]
    V_action = value_states[:, :, vlm_len:, :]

    # Detach VLM K/V *only* on the path the action queries use.
    K_vlm_det = K_vlm.detach()
    V_vlm_det = V_vlm.detach()
    K_for_vlm = key_states  # full (gradients flow)
    V_for_vlm = value_states
    K_for_action = torch.cat([K_vlm_det, K_action], dim=2)
    V_for_action = torch.cat([V_vlm_det, V_action], dim=2)

    mask_for_vlm = attention_mask[:, :, :vlm_len, :]
    mask_for_action = attention_mask[:, :, vlm_len:, :]
    # ``_prepare_attention_masks_4d`` always returns fp32 (0.0 / -inf
    # literals), but PaliGemma weights are bf16 when ``dtype=bfloat16``,
    # making q bf16. SDPA's ``scaled_dot_product_attention`` then raises
    # "invalid dtype for bias - should match query's dtype". Cast each
    # mask slice to the corresponding query dtype right before use.
    if mask_for_vlm.dtype != Q_vlm.dtype:
        mask_for_vlm = mask_for_vlm.to(dtype=Q_vlm.dtype)
    if mask_for_action.dtype != Q_action.dtype:
        mask_for_action = mask_for_action.to(dtype=Q_action.dtype)

    from .pi05_backbone import sdpa_attention_forward  # noqa: PLC0415

    att_vlm, _ = sdpa_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        Q_vlm, K_for_vlm, V_for_vlm, mask_for_vlm, scaling,
    )
    att_action, _ = sdpa_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        Q_action, K_for_action, V_for_action, mask_for_action, scaling,
    )
    att = torch.cat([att_vlm, att_action], dim=1)

    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att = att.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end = start + hidden_states.shape[1]
        if att.dtype != layer.self_attn.o_proj.weight.dtype:
            att = att.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att[:, start:end])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb.clone(), cond=adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first, out_emb, gate)
        outputs_embeds.append(out_emb)
        start = end
    return outputs_embeds


def _paligemma_forward_ki(
    self,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    adarms_cond=None,
):
    """Replacement ``PaliGemmaWithExpertModel.forward`` that routes the
    dual-expert layer pass through :func:`_compute_layer_ki`.

    Bound onto the model instance when ``config.knowledge_insulation``
    is True (see ``PI052Policy.__init__``). Single-expert branches
    (VLM-only or action-only) defer back to the original forward —
    KI only matters when actions and VLM tokens are forwarded together.
    """
    from .pi05_backbone import layernorm_forward  # noqa: PLC0415

    if adarms_cond is None:
        adarms_cond = [None, None]

    # Single-expert paths: defer to the original forward saved in
    # PI052Policy.__init__.
    if inputs_embeds[0] is None or inputs_embeds[1] is None:
        return self._pi052_orig_forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )

    models = [self.paligemma.model.language_model, self.gemma_expert.model]
    num_layers = self.paligemma.config.text_config.num_hidden_layers
    use_gc = (
        hasattr(self.gemma_expert.model, "gradient_checkpointing")
        and self.gemma_expert.model.gradient_checkpointing
        and self.training
    ) or (
        hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training
    )

    for layer_idx in range(num_layers):
        if use_gc:
            inputs_embeds = torch.utils.checkpoint.checkpoint(
                _compute_layer_ki,
                layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
                use_reentrant=False, preserve_rng_state=False,
                paligemma=self.paligemma, gemma_expert=self.gemma_expert,
            )
        else:
            inputs_embeds = _compute_layer_ki(
                layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
                paligemma=self.paligemma, gemma_expert=self.gemma_expert,
            )

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
        outputs_embeds.append(out_emb)
    return [outputs_embeds[0], outputs_embeds[1]], None


class PI052Policy(PreTrainedPolicy):
    """π0.5 with the PaliGemma LM head re-enabled.

    Self-contained: the PI0.5 backbone (PaliGemmaWithExpertModel / PI05Pytorch)
    is vendored in ``pi05_backbone.py`` and the PI05Policy wrapper logic is
    inlined directly here, so this policy does not depend on or inherit from
    ``lerobot.policies.pi05`` (which stays identical to ``main``).
    """

    config_class = PI052Config
    name = "pi052"

    def __init__(self, config: PI052Config, **kwargs: Any) -> None:
        # Patch ops BEFORE the backbone is built (the backbone constructed
        # below instantiates the Gemma/Siglip layers we want to swap).
        # Always-on — the patch is process-global / idempotent and degrades
        # gracefully if liger-kernel is missing.
        _enable_hf_kernels()

        # ---- inlined PI05Policy.__init__ ----------------------------------
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()
        # ---- end inlined PI05Policy.__init__ ------------------------------

        # ``PI05Policy.__init__`` zeroes the PaliGemma ``lm_head`` and
        # freezes a few terminal layers when ``train_expert_only`` is
        # the (default) True. We re-enable the head if the user
        # wants text supervision.
        if config.text_loss_weight > 0 and config.unfreeze_lm_head:
            self._unfreeze_lm_head()

        # Knowledge insulation: bind a custom ``forward`` on the
        # PaliGemmaWithExpertModel instance that uses
        # :func:`_compute_layer_ki` for the dual-expert layer pass.
        # The bind is per-instance, so this doesn't leak into stock
        # ``pi05`` policies that share the same class.
        if getattr(config, "knowledge_insulation", False):
            backbone = self.model.paligemma_with_expert
            backbone._pi052_orig_forward = backbone.forward
            backbone.forward = types.MethodType(_paligemma_forward_ki, backbone)
            logger.info(
                "PI052: knowledge insulation enabled — action→VLM K/V "
                "gradients are blocked in attention."
            )

        # Per-env hierarchical-inference state. Sized lazily on the first
        # select_action() call once the batch size (number of parallel envs)
        # is known. ``last_subtasks[i]`` is the subtask currently conditioning
        # env ``i``'s action expert; scalar ``last_subtask`` mirrors env 0 for
        # back-compat (e.g. the eval video overlay).
        self.last_subtasks: list[str] | None = None
        self.last_subtasks_raw: list[str] | None = None
        self.last_subtasks_source: list[str] | None = None
        self._last_good_subtasks: list[str | None] | None = None
        self.last_subtask: str | None = None
        self.last_subtask_raw: str | None = None
        self.last_subtask_source: str = "unset"
        self.last_subtask_debug: str = ""

    def reset(self):
        """Reset action and high-level inference state."""
        # inlined PI05Policy.reset
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self.last_subtasks = None
        self.last_subtasks_raw = None
        self.last_subtasks_source = None
        self._last_good_subtasks = None
        self.last_subtask = None
        self.last_subtask_raw = None
        self.last_subtask_source = "unset"
        self.last_subtask_debug = ""

    # ------------------------------------------------------------------
    # Head unfreeze helper
    # ------------------------------------------------------------------

    def _unfreeze_lm_head(self) -> None:
        """Walk the PaliGemma submodules and re-enable gradients on
        ``lm_head`` + the immediately preceding norm / last text-model
        layer that ``PI05Policy`` typically freezes."""
        backbone = self.model.paligemma_with_expert.paligemma
        if hasattr(backbone, "lm_head"):
            for p in backbone.lm_head.parameters():
                p.requires_grad_(True)
        # The text model's final norm and last transformer block —
        # find them dynamically by walking up from the LM head so we
        # don't hard-code module names that may drift across transformers
        # versions.
        text_model = getattr(backbone, "model", None)
        text_model = getattr(text_model, "language_model", text_model)
        if text_model is None:
            return
        norm = getattr(text_model, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)
        layers = getattr(text_model, "layers", None)
        if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
            for p in layers[-1].parameters():
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward (dual loss: flow + text)
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Dual-head forward: flow-matching loss + text-CE loss.

        When ``text_labels`` isn't present in the batch (e.g. the
        recipe wasn't applied), we delegate to ``PI05Policy.forward``
        unchanged. Otherwise we compute both losses and sum them with
        ``flow_loss_weight`` / ``text_loss_weight``.
        """
        text_labels = batch.get("text_labels")
        predict_actions_t = batch.get("predict_actions")

        # Fall through to PI05Policy only on fully unannotated batches
        # (no recipe applied → no routing fields). For recipe-applied
        # batches we keep control of the loss dispatch even if all
        # samples are text-only — delegating would silently train flow
        # on text-only frames (PI05Policy.forward ignores
        # ``predict_actions``).
        if (
            text_labels is None
            and predict_actions_t is None
            and not getattr(self.config, "enable_fast_action_loss", False)
        ):
            return self._pi05_flow_forward(batch, reduction=reduction)

        run_flow = (
            self.config.flow_loss_weight > 0
            and (predict_actions_t is None or bool(predict_actions_t.any().item()))
        )
        run_text = self.config.text_loss_weight > 0 and text_labels is not None

        loss_dict: dict[str, Any] = {}
        total: Tensor | None = None

        # Decide which losses fire this step.
        run_fast = (
            getattr(self.config, "enable_fast_action_loss", False)
            and self.config.fast_action_loss_weight > 0
            and (predict_actions_t is None or bool(predict_actions_t.any().item()))
        )
        action_tokens = action_mask = action_code_mask = None
        if run_fast:
            from lerobot.utils.constants import (  # noqa: PLC0415
                ACTION_CODE_TOKEN_MASK,
                ACTION_TOKEN_MASK,
                ACTION_TOKENS,
            )

            action_tokens = batch.get(ACTION_TOKENS)
            action_mask = batch.get(ACTION_TOKEN_MASK)
            action_code_mask = batch.get(ACTION_CODE_TOKEN_MASK)
            if action_tokens is None or action_mask is None or action_code_mask is None:
                run_fast = False

        # ------------------------------------------------------------
        # Dispatch: full fusion when flow is active, otherwise the
        # prefix-only text+FAST helper (no suffix forward needed).
        #
        # Full fusion (flow ON):
        #   ONE backbone forward with prefix=[images, lang, FAST] +
        #   suffix=[noisy_actions], suffix→FAST attention masked out.
        #   All three losses computed from slices of the single output.
        #
        # Prefix-only fusion (flow OFF, e.g. text-only recipes):
        #   ONE prefix-only forward, both text + FAST losses computed
        #   from slices. No suffix forward → cheaper.
        # ------------------------------------------------------------
        if run_flow:
            flow_loss, text_loss, fast_loss = self._compute_all_losses_fused(
                batch,
                text_labels=text_labels if run_text else None,
                action_tokens=action_tokens if run_fast else None,
                action_mask=action_mask if run_fast else None,
                action_code_mask=action_code_mask if run_fast else None,
                predict_actions_t=predict_actions_t,
            )
            loss_dict["flow_loss"] = float(flow_loss.detach().item())
            total = self.config.flow_loss_weight * flow_loss
            if text_loss is not None:
                loss_dict["text_loss"] = float(text_loss.detach().item())
                total = total + self.config.text_loss_weight * text_loss
            if fast_loss is not None:
                loss_dict["fast_action_loss"] = float(fast_loss.detach().item())
                total = total + self.config.fast_action_loss_weight * fast_loss
        elif run_text or run_fast:
            text_loss, fast_loss = self._compute_text_and_fast_loss(
                batch,
                text_labels=text_labels if run_text else None,
                action_tokens=action_tokens if run_fast else None,
                action_mask=action_mask if run_fast else None,
                action_code_mask=action_code_mask if run_fast else None,
                predict_actions_t=predict_actions_t,
            )
            if text_loss is not None:
                loss_dict["text_loss"] = float(text_loss.detach().item())
                weighted = self.config.text_loss_weight * text_loss
                total = weighted if total is None else total + weighted
            if fast_loss is not None:
                loss_dict["fast_action_loss"] = float(fast_loss.detach().item())
                weighted = self.config.fast_action_loss_weight * fast_loss
                total = weighted if total is None else total + weighted

        if total is None:
            # Both flow and text disabled — make this an obvious bug
            # rather than a silent zero loss.
            raise RuntimeError(
                "PI052Policy.forward: both flow_loss_weight and "
                "text_loss_weight are 0 (or text_labels missing) — "
                "nothing to train."
            )

        loss_dict["loss"] = float(total.detach().item()) if total.dim() == 0 else float("nan")
        if reduction == "none":
            return total.expand(batch[OBS_LANGUAGE_TOKENS].shape[0]), loss_dict
        return total, loss_dict

    # ------------------------------------------------------------------
    # Text loss
    # ------------------------------------------------------------------

    def _compute_all_losses_fused(
        self,
        batch: dict[str, Tensor],
        text_labels: Tensor | None,
        action_tokens: Tensor | None,
        action_mask: Tensor | None,
        action_code_mask: Tensor | None,
        predict_actions_t: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Full fusion: flow + text + FAST in ONE backbone forward.

        Builds:

            prefix = [images, language, FAST (when provided)]
            suffix = [noisy_actions]   (action expert via gemma_expert)

        Then overrides the unified 2D attention mask to *explicitly*
        zero out ``suffix → FAST`` attention. Without this override
        the action expert would attend to the discrete FAST tokens
        and trivially decode them back to the same continuous
        actions it's supposed to predict via flow matching — the
        whole training signal collapses.

        Both prefix_out and suffix_out are captured from the same
        forward. From prefix_out we slice the language and FAST
        token positions and compute their CE losses. From suffix_out
        we run the existing flow path (action_out_proj → MSE).

        Returns ``(flow_loss, text_loss, fast_loss)`` where text/fast
        can be ``None`` when the caller didn't supply the
        corresponding inputs.
        """
        from lerobot.utils.constants import ACTION  # noqa: PLC0415

        from .pi05_backbone import make_att_2d_masks  # noqa: PLC0415

        # ---- preamble (mirrors PI05Pytorch.forward) ------------------
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ---- prefix: images + language + (optional FAST) -------------
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        non_fast_prefix_len = prefix_embs.shape[1]  # images + language only

        # Causal-mask the supervised text-target span so the text-CE is
        # genuine next-token prediction, not a bidirectional copy task
        # (see ``_mark_target_span_causal``).
        if text_labels is not None:
            lang_start = non_fast_prefix_len - text_labels.shape[1]
            if lang_start >= 0:
                prefix_att = _mark_target_span_causal(
                    prefix_att, text_labels, lang_start, non_fast_prefix_len
                )

        fast_len = 0
        if action_tokens is not None and action_mask is not None:
            # embed_language_tokens already applies the Gemma sqrt(hidden) scale (tf>=5.4.0);
            # do not scale FAST action tokens again (would double-scale).
            fast_emb = self.model.paligemma_with_expert.embed_language_tokens(action_tokens)
            fast_len = action_tokens.shape[1]
            ones_att = torch.ones(
                (action_tokens.shape[0], fast_len),
                dtype=torch.bool,
                device=prefix_embs.device,
            )
            prefix_embs = torch.cat([prefix_embs, fast_emb], dim=1)
            prefix_pad = torch.cat([prefix_pad, action_mask.to(prefix_pad.dtype)], dim=1)
            prefix_att = torch.cat([prefix_att, ones_att], dim=1)

        # ---- suffix: noisy actions ----------------------------------
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.model.embed_suffix(x_t, time)

        # ---- bf16 alignment (mirrors PI05Pytorch.forward) -----------
        first_layer = (
            self.model.paligemma_with_expert.paligemma.model.language_model.layers[0]
        )
        if first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # ---- combined attention -------------------------------------
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        # Critical: zero out suffix → FAST attention. Without this the
        # action expert reads the FAST tokens and trivially decodes
        # them back to the same continuous actions it's supposed to
        # predict from noise. Cumulative-block attention from
        # ``make_att_2d_masks`` doesn't enforce this on its own
        # because suffix tokens have a strictly higher cumsum than
        # FAST tokens and therefore attend to them by default.
        if fast_len > 0:
            fast_start = non_fast_prefix_len
            fast_end = non_fast_prefix_len + fast_len  # = prefix_pad.shape[1]
            att_2d_masks[:, fast_end:, fast_start:fast_end] = False

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.model._prepare_attention_masks_4d(
            att_2d_masks, dtype=prefix_embs.dtype
        )

        # ---- forward (capture BOTH expert outputs) ------------------
        (prefix_out, suffix_out), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # ---- flow loss (mirrors PI05Pytorch.forward) ----------------
        suffix_out_slice = suffix_out[:, -self.model.config.chunk_size :].to(
            dtype=torch.float32
        )
        v_t = self.model.action_out_proj(suffix_out_slice)
        flow_per_dim = F.mse_loss(u_t, v_t, reduction="none")
        # Truncate to the actual action dimensionality (PI05 pads
        # internally to max_action_dim).
        original_action_dim = self.config.output_features[ACTION].shape[0]
        flow_per_dim = flow_per_dim[:, :, :original_action_dim]
        per_sample_flow = flow_per_dim.mean(dim=(1, 2))
        flow_loss = _mask_per_sample(per_sample_flow, predict_actions_t)

        # ---- text + FAST CE from prefix_out ------------------------
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head

        text_loss: Tensor | None = None
        if text_labels is not None and prefix_out is not None:
            lang_len = text_labels.shape[1]
            if fast_len > 0:
                text_hidden = prefix_out[:, -(fast_len + lang_len) : -fast_len, :]
            else:
                text_hidden = prefix_out[:, -lang_len:, :]
            # Liger fused linear-CE: skip the explicit ``lm_head(...)``
            # materialisation; the kernel multiplies on-the-fly and
            # never holds the full (B, T, 257k) logits tensor.
            text_loss = _shifted_lin_ce(
                text_hidden,
                lm_head.weight,
                text_labels,
                z_loss_weight=getattr(self.config, "text_ce_z_loss_weight", 0.0),
            )

        fast_loss: Tensor | None = None
        if fast_len > 0 and prefix_out is not None and action_code_mask is not None:
            fast_hidden = prefix_out[:, -fast_len:, :]
            fast_loss = _fast_lin_ce(
                fast_hidden,
                lm_head.weight,
                action_tokens,
                action_code_mask,
                predict_actions_t,
            )

        return flow_loss, text_loss, fast_loss

    def _compute_text_and_fast_loss(
        self,
        batch: dict[str, Tensor],
        text_labels: Tensor | None,
        action_tokens: Tensor | None,
        action_mask: Tensor | None,
        action_code_mask: Tensor | None,
        predict_actions_t: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Single prefix forward → text CE + FAST CE.

        Embed [images, language] (and FAST when requested) once, run
        one backbone forward, then slice the resulting hidden states
        at the language and FAST positions to compute both CE losses.
        Bit-equivalent to running the two losses in separate forwards
        because the segment-aware ``make_att_2d_masks`` keeps FAST
        tokens invisible to language tokens, so adding FAST to the
        prefix doesn't perturb the hidden states at language positions.

        Returns ``(text_loss, fast_loss)``. Either can be ``None`` if
        the caller doesn't want that head.
        """
        from .pi05_backbone import make_att_2d_masks  # noqa: PLC0415

        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Causal-mask the supervised text-target span (see
        # ``_mark_target_span_causal``) before the FAST tokens are
        # appended — same fix as ``_compute_all_losses_fused``.
        if text_labels is not None:
            lang_start = prefix_embs.shape[1] - text_labels.shape[1]
            if lang_start >= 0:
                prefix_att = _mark_target_span_causal(
                    prefix_att, text_labels, lang_start, prefix_embs.shape[1]
                )

        fast_len = 0
        if action_tokens is not None and action_mask is not None:
            # embed_language_tokens already applies the Gemma sqrt(hidden) scale (tf>=5.4.0);
            # do not scale FAST action tokens again (would double-scale).
            fast_emb = self.model.paligemma_with_expert.embed_language_tokens(action_tokens)

            fast_len = action_tokens.shape[1]
            ones_att = torch.ones(
                (action_tokens.shape[0], fast_len),
                dtype=torch.bool,
                device=prefix_embs.device,
            )
            full_embs = torch.cat([prefix_embs, fast_emb], dim=1)
            full_pad = torch.cat([prefix_pad, action_mask.to(prefix_pad.dtype)], dim=1)
            full_att = torch.cat([prefix_att, ones_att], dim=1)
        else:
            full_embs = prefix_embs
            full_pad = prefix_pad
            full_att = prefix_att

        att_2d = make_att_2d_masks(full_pad, full_att)
        position_ids = torch.cumsum(full_pad, dim=1) - 1
        att_2d_4d = self.model._prepare_attention_masks_4d(att_2d, dtype=full_embs.dtype)

        (vlm_out, _), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[full_embs, None],
            use_cache=False,
        )
        if vlm_out is None:
            raise RuntimeError(
                "PI052 text+fast loss: VLM forward returned no hidden states."
            )

        lm_head = self.model.paligemma_with_expert.paligemma.lm_head

        text_loss: Tensor | None = None
        if text_labels is not None:
            lang_len = text_labels.shape[1]
            # embed_prefix lays out as [images, language]; with FAST
            # appended the full sequence is [images, language, FAST].
            if fast_len > 0:
                text_hidden = vlm_out[:, -(fast_len + lang_len):-fast_len, :]
            else:
                text_hidden = vlm_out[:, -lang_len:, :]
            text_loss = _shifted_lin_ce(
                text_hidden,
                lm_head.weight,
                text_labels,
                z_loss_weight=getattr(self.config, "text_ce_z_loss_weight", 0.0),
            )

        fast_loss: Tensor | None = None
        if (
            action_tokens is not None
            and action_code_mask is not None
            and fast_len > 0
        ):
            fast_hidden = vlm_out[:, -fast_len:, :]
            fast_loss = _fast_lin_ce(
                fast_hidden,
                lm_head.weight,
                action_tokens,
                action_code_mask,
                predict_actions_t,
            )

        return text_loss, fast_loss

    # ------------------------------------------------------------------
    # Diagnostic: forward + argmax for supervised text positions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def debug_text_predictions(
        self, batch: dict[str, Tensor], max_samples: int = 5
    ) -> dict[str, Tensor]:
        """Run the text-loss forward but return argmax predictions instead of CE.

        Lets a periodic training-loop hook compare what the LM head emits
        right now against what it *should* emit at every supervised
        position — the cheapest "is text training actually working"
        diagnostic. Returns CPU tensors keyed by ``input_ids``,
        ``attention_mask``, ``labels``, ``predictions``; predictions are
        aligned with input positions (``predictions[t]`` is the head's
        argmax after seeing ``input_ids[:t+1]``, so it should match
        ``input_ids[t+1]`` for next-token prediction). Returns ``{}``
        when the batch has no supervised text positions.
        """
        from .pi05_backbone import make_att_2d_masks  # noqa: PLC0415

        text_labels = batch.get("text_labels")
        if text_labels is None or not bool((text_labels != -100).any().item()):
            return {}

        was_training = self.training
        self.eval()
        try:
            n = min(max_samples, int(text_labels.shape[0]))
            sub: dict[str, Any] = {
                OBS_LANGUAGE_TOKENS: batch[OBS_LANGUAGE_TOKENS][:n],
                OBS_LANGUAGE_ATTENTION_MASK: batch[OBS_LANGUAGE_ATTENTION_MASK][:n],
            }
            for k, v in batch.items():
                if isinstance(k, str) and k.startswith("observation.images.") and torch.is_tensor(v):
                    sub[k] = v[:n]

            sub_labels = text_labels[:n]
            images, img_masks = self._preprocess_images(sub)
            lang_tokens = sub[OBS_LANGUAGE_TOKENS]
            lang_masks = sub[OBS_LANGUAGE_ATTENTION_MASK]

            prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            lang_start = prefix_embs.shape[1] - sub_labels.shape[1]
            if lang_start >= 0:
                prefix_att = _mark_target_span_causal(
                    prefix_att, sub_labels, lang_start, prefix_embs.shape[1]
                )

            att_2d = make_att_2d_masks(prefix_pad, prefix_att)
            position_ids = torch.cumsum(prefix_pad, dim=1) - 1
            att_2d_4d = self.model._prepare_attention_masks_4d(att_2d)
            backbone = self.model.paligemma_with_expert
            backbone_dtype = (
                backbone.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype
            )
            if att_2d_4d.dtype != backbone_dtype:
                att_2d_4d = att_2d_4d.to(dtype=backbone_dtype)

            (vlm_out, _), _ = backbone.forward(
                attention_mask=att_2d_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )
            text_hidden = vlm_out[:, -sub_labels.shape[1]:, :]
            lm_head = backbone.paligemma.lm_head
            text_logits = lm_head(text_hidden.to(lm_head.weight.dtype))
            preds = text_logits.argmax(dim=-1)

            return {
                "input_ids": lang_tokens.detach().cpu(),
                "attention_mask": lang_masks.detach().cpu(),
                "labels": sub_labels.detach().cpu(),
                "predictions": preds.detach().cpu(),
            }
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------
    # select_message — AR text generation at inference
    # ------------------------------------------------------------------

    def select_message(
        self,
        batch: dict[str, Tensor],
        *,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer: Any = None,
        suppress_loc_tokens: bool = False,
    ) -> str:
        """Generate text continuation from a multimodal prefix.

        Consumed by :class:`lerobot.policies.pi052.inference.PI052Runtime`
        for the high-level / VQA / memory-update text streams.

        ``suppress_loc_tokens`` masks PaliGemma's reserved ``<locDDDD>``
        ids ([256000, 257024)) to ``-inf`` before sampling. PaliGemma's
        pretraining puts heavy first-token mass on these ids for any
        ``Assistant:`` continuation; with a small fine-tuning text-CE
        budget (or aggressive LR decay) the LM head can drift back
        toward that prior even when teacher-forced argmax stays at
        100%. Callsites that legitimately emit ``<loc>`` (VQA spatial
        answers) must keep this ``False``; subtask / memory / plan
        generation should pass ``True``.
        """
        self.eval()

        if tokenizer is None:
            from transformers import AutoTokenizer  # noqa: PLC0415

            from .text_processor_pi052 import register_paligemma_loc_tokens  # noqa: PLC0415

            tokenizer = register_paligemma_loc_tokens(
                AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            )
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        special_ids: set[int] = set()
        try:
            for sid in (tokenizer.all_special_ids or []):
                if sid is not None:
                    special_ids.add(int(sid))
        except Exception:  # noqa: BLE001
            pass
        if eos_token_id is not None:
            special_ids.add(int(eos_token_id))

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )

        device = prefix_embs.device
        bsize = prefix_embs.shape[0]
        ones_step = torch.ones((bsize, 1), dtype=torch.bool, device=device)

        current_embs = prefix_embs
        current_pad = prefix_pad_masks
        current_att = prefix_att_masks
        generated: list[int] = []

        from .pi05_backbone import make_att_2d_masks  # noqa: PLC0415

        backbone = self.model.paligemma_with_expert
        lm_head = backbone.paligemma.lm_head

        # ``_prepare_attention_masks_4d`` always returns fp32 (0.0 / -inf
        # literals). When weights are bf16, HF's PaliGemma SDPA raises
        # "invalid dtype for bias - should match query's dtype". Pull the
        # dtype from an attention *projection* weight specifically:
        # ``to_bfloat16_for_selected_params`` keeps norms / embeddings in
        # fp32 even when the rest is bf16, so ``next(parameters())``
        # would land on one of those and we'd skip the cast. q_proj is
        # always cast with the rest, so its dtype is the one SDPA sees.
        backbone_dtype = (
            backbone.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )

        for _ in range(max_new_tokens):
            att_2d = make_att_2d_masks(current_pad, current_att)
            position_ids = torch.cumsum(current_pad, dim=1) - 1
            att_2d_4d = self.model._prepare_attention_masks_4d(att_2d, dtype=backbone_dtype)
            (vlm_out, _), _ = backbone.forward(
                attention_mask=att_2d_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[current_embs, None],
                use_cache=False,
            )
            if vlm_out is None:
                break
            last = vlm_out[:, -1:].to(lm_head.weight.dtype)
            logits_step = lm_head(last)[:, -1]  # (B, V)
            if special_ids and len(generated) < min_new_tokens:
                for sid in special_ids:
                    logits_step[..., sid] = float("-inf")
            # Mask FAST action tokens that fall *below* the ``<loc>`` block.
            # They are never valid text, but the action-trained head leaks
            # them as gibberish; unlike the loc/seg block this region is never
            # legitimately emitted (even by VQA), so suppress it on every call.
            vocab_size = logits_step.shape[-1]
            fast_skip = int(getattr(self.config, "fast_skip_tokens", 128))
            fast_lo = vocab_size - 1 - fast_skip - (_FAST_ACTION_VOCAB_SIZE - 1)
            if 0 < fast_lo < 256000:
                logits_step[..., fast_lo:256000] = float("-inf")
            if suppress_loc_tokens:
                logits_step[..., 256000:257024] = float("-inf")
            next_ids = self._sample_next_token(logits_step, temperature, top_p)
            tok_id = int(next_ids[0].item())
            generated.append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id:
                break

            # embed_language_tokens already applies the Gemma sqrt(hidden) scale (tf>=5.4.0).
            new_emb = backbone.embed_language_tokens(next_ids.unsqueeze(0))
            current_embs = torch.cat([current_embs, new_emb], dim=1)
            current_pad = torch.cat([current_pad, ones_step], dim=1)
            current_att = torch.cat([current_att, ones_step], dim=1)

        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not decoded and generated:
            try:
                self._last_select_message_debug = (
                    f"raw_ids={generated[:16]} "
                    f"decoded_w_special={tokenizer.decode(generated, skip_special_tokens=False)!r}"
                )
            except Exception:  # noqa: BLE001
                self._last_select_message_debug = f"raw_ids={generated[:16]}"
        else:
            self._last_select_message_debug = ""
        return decoded

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select an action via PI052's high-level → low-level inference path.

        At action-chunk boundaries, first generate a low-level subtask from
        the high-level task prompt. Then retokenize that subtask as the
        low-level action prompt before sampling the action chunk. This keeps
        the public policy API identical to PI05 (`Tensor` action out), while
        matching the PI052 training/runtime conditioning more closely.
        """
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        if len(self._action_queue) == 0:
            action_batch = self._with_low_level_subtask_prompt(batch)
            actions = self.predict_action_chunk(action_batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def _with_low_level_subtask_prompt(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        from .inference.steps import _build_text_batch  # noqa: PLC0415

        n = self._batch_size_from_observation(batch)
        self._ensure_subtask_state(n)
        tasks = self._tasks_from_batch(batch, n)

        # Generate one subtask per parallel env, each conditioned on that env's
        # own task + observation, then stack the per-env prompts into a single
        # (n, L) batch for the action expert. This keeps batch_size > 1 correct
        # (env i is conditioned on env i's subtask, not a broadcast of env 0).
        rows: list[tuple[Tensor, Tensor | None]] = []
        tokenizer = None
        for i in range(n):
            obs_i = self._slice_observation(batch, i)
            subtask = self._generate_low_level_subtask(obs_i, tasks[i], i)
            text_batch = _build_text_batch(
                self,
                [{"role": "user", "content": subtask}],
                add_generation_prompt=False,
            )
            rows.append((text_batch["lang_tokens"], text_batch["lang_masks"]))
            tokenizer = text_batch["tokenizer"]

        tokens, masks = self._stack_token_rows(rows, tokenizer)

        # Scalar aliases mirror env 0 for back-compat / single-env overlays.
        self.last_subtask = self.last_subtasks[0] if self.last_subtasks else None
        self.last_subtask_raw = self.last_subtasks_raw[0] if self.last_subtasks_raw else None
        self.last_subtask_source = (
            self.last_subtasks_source[0] if self.last_subtasks_source else "unset"
        )

        out = dict(batch)
        out[OBS_LANGUAGE_TOKENS] = tokens
        out[OBS_LANGUAGE_ATTENTION_MASK] = masks
        return out

    def _generate_low_level_subtask(self, obs_i: dict[str, Tensor], task: str, i: int) -> str:
        from .inference.steps import _generate_with_policy, _looks_like_gibberish  # noqa: PLC0415

        msg = ""
        if task:
            msg = _generate_with_policy(
                self,
                [{"role": "user", "content": task}],
                observation=obs_i,
                label=f"eval subtask gen[{i}]",
                suppress_loc_tokens=True,
            )
        self.last_subtasks_raw[i] = msg or ""

        # Faithful hierarchical inference: condition the action expert on the
        # model's own generated subtask verbatim (this is exactly what the
        # ``low_level_execution`` recipe did at training — ``user: ${subtask}``).
        if msg and not _looks_like_gibberish(msg):
            subtask = " ".join(msg.strip().split())
            self._last_good_subtasks[i] = subtask
            self.last_subtasks[i] = subtask
            self.last_subtasks_source[i] = "generated"
            logger.info("PI052 eval subtask[%d]: %r (task=%r)", i, subtask, task)
            return subtask

        # Generation unusable (empty / gibberish). Training never fed such a
        # prompt to the action expert, so the least-OOD choice is to reuse this
        # env's last accepted subtask; on the first chunk (none yet) derive one
        # from the task so the action expert still gets an imperative command
        # rather than the raw high-level instruction.
        debug = getattr(self, "_last_select_message_debug", "") or ""
        if not task:
            reason = "No task string was available in the batch."
        elif msg:
            reason = f"Rejected generated subtask: {msg!r}"
        else:
            reason = f"Empty generated subtask. {debug}".strip()
        if self._last_good_subtasks[i]:
            subtask = self._last_good_subtasks[i]
            source = "reuse_last"
        else:
            subtask = self._fallback_subtask_from_task(task)
            source = "fallback_task"
        self.last_subtasks[i] = subtask
        self.last_subtasks_source[i] = source
        logger.info(
            "PI052 eval subtask[%d] fallback (%s): %s | final=%r task=%r",
            i,
            source,
            reason,
            subtask,
            task,
        )
        return subtask

    def _ensure_subtask_state(self, n: int) -> None:
        """(Re)allocate per-env subtask buffers when the env count is first seen."""
        if self.last_subtasks is not None and len(self.last_subtasks) == n:
            return
        self.last_subtasks = ["" for _ in range(n)]
        self.last_subtasks_raw = ["" for _ in range(n)]
        self.last_subtasks_source = ["unset" for _ in range(n)]
        self._last_good_subtasks = [None for _ in range(n)]

    @staticmethod
    def _slice_observation(batch: dict[str, Tensor], i: int) -> dict[str, Tensor]:
        """Slice the per-env observation tensors for env ``i`` (images/state).

        Language keys are excluded so high-level generation uses the freshly
        tokenized task prompt, not the preprocessor's low-level fallback tokens.
        """
        out: dict[str, Tensor] = {}
        for k, v in batch.items():
            if not (isinstance(k, str) and k.startswith("observation.")):
                continue
            if k.startswith("observation.language"):
                continue
            if torch.is_tensor(v):
                out[k] = v[i : i + 1]
        return out

    @staticmethod
    def _stack_token_rows(
        rows: list[tuple[Tensor, Tensor | None]], tokenizer: Any
    ) -> tuple[Tensor, Tensor]:
        """Right-pad per-env ``(1, L_i)`` token/mask rows and stack to ``(n, L)``.

        Right-padding with a False attention mask matches the training-time
        tokenizer (``padding_side="right"``), so the action expert treats pad
        positions as masked.
        """
        max_len = max(t.shape[1] for t, _ in rows)
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        tok_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        for tokens, masks in rows:
            length = tokens.shape[1]
            if masks is None:
                masks = torch.ones((1, length), dtype=torch.bool, device=tokens.device)
            if length < max_len:
                pad = max_len - length
                tokens = torch.cat(
                    [tokens, torch.full((1, pad), pad_id, dtype=tokens.dtype, device=tokens.device)],
                    dim=1,
                )
                masks = torch.cat(
                    [masks, torch.zeros((1, pad), dtype=masks.dtype, device=masks.device)],
                    dim=1,
                )
            tok_rows.append(tokens)
            mask_rows.append(masks)
        return torch.cat(tok_rows, dim=0), torch.cat(mask_rows, dim=0)

    @staticmethod
    def _fallback_subtask_from_task(task: str) -> str:
        target = PI052Policy._navigation_target_from_task(task)
        if target:
            return f"go to {target}"
        if task.lower().startswith("open the stand mixer head"):
            return "pull stand mixer head"
        return task

    @staticmethod
    def _navigation_target_from_task(task: str) -> str:
        prefix = "navigate to "
        lower = task.lower().strip()
        if not lower.startswith(prefix):
            return ""
        return lower[len(prefix) :].strip().rstrip(".")

    @staticmethod
    def _tasks_from_batch(batch: dict[str, Any], n: int) -> list[str]:
        """Return one task string per env, padded/truncated to ``n``."""
        task = batch.get("task")
        if isinstance(task, list):
            raw = list(task)
        elif task is None:
            raw = []
        else:
            raw = [task]
        tasks: list[str] = []
        for t in raw:
            if hasattr(t, "item"):
                t = t.item()
            tasks.append(t if isinstance(t, str) else "")
        if len(tasks) < n:
            tasks += [tasks[-1] if tasks else ""] * (n - len(tasks))
        return tasks[:n]

    @staticmethod
    def _batch_size_from_observation(batch: dict[str, Any]) -> int:
        state = batch.get("observation.state")
        if torch.is_tensor(state) and state.ndim > 0:
            return int(state.shape[0])
        for key, value in batch.items():
            if isinstance(key, str) and key.startswith("observation.images.") and torch.is_tensor(value):
                return int(value.shape[0])
        return 1

    @staticmethod
    def _sample_next_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_ix = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_p, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_p = sorted_p.masked_fill(mask, 0.0)
            sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            choice = torch.multinomial(sorted_p, num_samples=1)
            return sorted_ix.gather(-1, choice).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------
    # Inlined from PI05Policy (vendored; pi052 does not inherit pi05).
    # Kept verbatim except PI05Policy.forward -> _pi05_flow_forward (the
    # flow-only fallback used by PI052Policy.forward on unannotated batches).
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
            embed_tokens_key = (
                "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            )
            if lm_head_key not in remapped_state_dict and embed_tokens_key in remapped_state_dict:
                remapped_state_dict[lm_head_key] = remapped_state_dict[embed_tokens_key].clone().float()
                print("Initialized PaliGemma lm_head from language token embeddings")
            elif lm_head_key in remapped_state_dict:
                remapped_state_dict[lm_head_key] = remapped_state_dict[lm_head_key].float()

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self):
        """Return policy parameters, optionally split into LR-scaled groups.

        When ``config.lm_head_lr_scale != 1.0``, the PaliGemma ``lm_head``
        and its tied ``embed_tokens`` are placed in their own param
        group with ``lr = base_lr * lm_head_lr_scale``. The cosine
        scheduler multiplies both groups by the same lambda each step,
        so the ratio is preserved across decay. Default ``1.0`` =
        return ``self.parameters()`` (back-compat with existing checkpoints
        and configs).
        """
        scale = float(getattr(self.config, "lm_head_lr_scale", 1.0))
        if scale == 1.0:
            return self.parameters()
        head_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        # Both ``lm_head.weight`` and the tied ``embed_tokens.weight`` —
        # boosting only the projection without the embedding pulls them
        # apart and breaks the tie that PaliGemma was pre-trained with.
        head_substrings = (
            "paligemma_with_expert.paligemma.lm_head.",
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.",
        )
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(s in name for s in head_substrings):
                head_params.append(p)
            else:
                other_params.append(p)
        base_lr = float(self.config.optimizer_lr)
        groups: list[dict[str, object]] = []
        if other_params:
            groups.append({"params": other_params, "lr": base_lr, "name": "policy"})
        if head_params:
            groups.append(
                {"params": head_params, "lr": base_lr * scale, "name": "lm_head"}
            )
        # Sanity: head_substrings must match at least one parameter, otherwise
        # the scale silently does nothing — surface that fast.
        if not head_params:
            raise RuntimeError(
                "lm_head_lr_scale != 1.0 but no parameters matched the LM-head "
                "name patterns: "
                f"{head_substrings!r}. Did the underlying PaliGemma module rename?"
            )
        logging.info(
            "PI05Policy: LM-head LR scale = %.3g (base=%.3g, head=%.3g) over "
            "%d head params + %d other params",
            scale,
            base_lr,
            base_lr * scale,
            len(head_params),
            len(other_params),
        )
        return groups


    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def _pi05_flow_forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
