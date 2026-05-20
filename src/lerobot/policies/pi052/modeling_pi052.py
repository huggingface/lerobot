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

The same multi-rate runtime that drives ``SmolVLA2Runtime`` (see
``lerobot.policies.smolvla2.inference``) can drive this policy too —
both expose ``predict_action_chunk`` for the action expert and
``select_message`` for the LM head.
"""

from __future__ import annotations

import logging
import math
import types
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..pi05.configuration_pi05 import PI05Config
from ..pi05.modeling_pi05 import PI05Policy
from .configuration_pi052 import PI052Config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Loss helpers (shared between fused and prefix-only paths)
# ----------------------------------------------------------------------


def _mask_per_sample(per_sample: Tensor, predict_actions_t: Tensor | None) -> Tensor:
    """Mean over samples where ``predict_actions_t`` is True, else over all."""
    if predict_actions_t is None:
        return per_sample.mean()
    mask = predict_actions_t.to(per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


def _shifted_ce(logits: Tensor, labels: Tensor) -> Tensor:
    """Next-token CE: hidden at t predicts label at t+1, ignore_index=-100.

    Mean over non-ignored positions across the batch. Returns 0 cleanly
    when no positions are supervised (clamp(min=1) on the denominator).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().long()
    valid = shift_labels != -100
    if not bool(valid.any().item()):
        return shift_logits.sum() * 0.0
    return F.cross_entropy(
        shift_logits[valid],
        shift_labels[valid],
        reduction="mean",
    )


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


def _fast_ce(
    fast_logits: Tensor,
    action_tokens: Tensor,
    action_code_mask: Tensor,
    predict_actions_t: Tensor | None,
) -> Tensor:
    """FAST action-code CE with token-span masking and per-sample action gating.

    ``action_code_mask`` is true only on the discrete action-code tokens,
    excluding the BOS / "Action: " / delimiter wrapper. Samples whose
    recipe sets ``predict_actions=False`` get all code positions masked
    out via the per-sample gate.
    """
    shift_logits = fast_logits[:, :-1, :].contiguous()
    shift_targets = action_tokens[:, 1:].contiguous().long()
    shift_valid = action_code_mask[:, 1:].contiguous().bool()
    if predict_actions_t is not None:
        sample_mask = predict_actions_t[:, None].expand_as(shift_valid)
        shift_valid = shift_valid & sample_mask
    if not bool(shift_valid.any().item()):
        return shift_logits.sum() * 0.0
    return F.cross_entropy(
        shift_logits[shift_valid],
        shift_targets[shift_valid],
        reduction="mean",
    )


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

    att_vlm, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        Q_vlm, K_for_vlm, V_for_vlm, mask_for_vlm, scaling,
    )
    att_action, _ = modeling_gemma.eager_attention_forward(
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
    from ..pi05.modeling_pi05 import layernorm_forward  # noqa: PLC0415

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


class PI052Policy(PI05Policy):
    """π0.5 with the PaliGemma LM head re-enabled."""

    config_class = PI052Config
    name = "pi052"

    def __init__(self, config: PI052Config, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
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
        # mirror SmolVLA2's logic, which finds these dynamically by
        # the trainable=False parameters that point at the head's
        # neighbourhood.
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
            return super().forward(batch, reduction=reduction)

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

        from ..pi05.modeling_pi05 import make_att_2d_masks  # noqa: PLC0415

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
            emb_dim = prefix_embs.shape[-1]
            fast_emb = self.model.paligemma_with_expert.embed_language_tokens(action_tokens)
            fast_emb = fast_emb * math.sqrt(emb_dim)
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
        att_2d_masks_4d = self.model._prepare_attention_masks_4d(att_2d_masks)

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
            text_logits = lm_head(text_hidden.to(lm_head.weight.dtype))
            text_loss = _shifted_ce(text_logits, text_labels)

        fast_loss: Tensor | None = None
        if fast_len > 0 and prefix_out is not None and action_code_mask is not None:
            fast_hidden = prefix_out[:, -fast_len:, :]
            fast_logits = lm_head(fast_hidden.to(lm_head.weight.dtype))
            fast_loss = _fast_ce(fast_logits, action_tokens, action_code_mask, predict_actions_t)

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
        from ..pi05.modeling_pi05 import make_att_2d_masks  # noqa: PLC0415

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
            emb_dim = prefix_embs.shape[-1]
            fast_emb = self.model.paligemma_with_expert.embed_language_tokens(action_tokens)
            fast_emb = fast_emb * math.sqrt(emb_dim)

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
        att_2d_4d = self.model._prepare_attention_masks_4d(att_2d)

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
            text_logits = lm_head(text_hidden.to(lm_head.weight.dtype))
            text_loss = _shifted_ce(text_logits, text_labels)

        fast_loss: Tensor | None = None
        if (
            action_tokens is not None
            and action_code_mask is not None
            and fast_len > 0
        ):
            fast_hidden = vlm_out[:, -fast_len:, :]
            fast_logits = lm_head(fast_hidden.to(lm_head.weight.dtype))
            fast_loss = _fast_ce(fast_logits, action_tokens, action_code_mask, predict_actions_t)

        return text_loss, fast_loss

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
    ) -> str:
        """Generate text continuation from a multimodal prefix.

        Mirrors ``SmolVLA2Policy.select_message`` so the same
        :class:`lerobot.policies.smolvla2.inference.SmolVLA2Runtime`
        can drive π0.5 v2 unchanged.
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
        emb_dim = prefix_embs.shape[-1]
        text_emb_scale = math.sqrt(emb_dim)
        ones_step = torch.ones((bsize, 1), dtype=torch.bool, device=device)

        current_embs = prefix_embs
        current_pad = prefix_pad_masks
        current_att = prefix_att_masks
        generated: list[int] = []

        from ..pi05.modeling_pi05 import make_att_2d_masks  # noqa: PLC0415

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
            att_2d_4d = self.model._prepare_attention_masks_4d(att_2d)
            if att_2d_4d.dtype != backbone_dtype:
                att_2d_4d = att_2d_4d.to(dtype=backbone_dtype)
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
            next_ids = self._sample_next_token(logits_step, temperature, top_p)
            tok_id = int(next_ids[0].item())
            generated.append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id:
                break

            new_emb = backbone.embed_language_tokens(next_ids.unsqueeze(0))
            new_emb = new_emb * text_emb_scale
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
