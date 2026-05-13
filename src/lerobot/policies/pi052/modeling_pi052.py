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
from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..pi05.configuration_pi05 import PI05Config
from ..pi05.modeling_pi05 import PI05Policy
from .configuration_pi052 import PI052Config

logger = logging.getLogger(__name__)


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

        run_flow = (
            self.config.flow_loss_weight > 0
            and (predict_actions_t is None or bool(predict_actions_t.any().item()))
        )
        run_text = self.config.text_loss_weight > 0 and text_labels is not None

        loss_dict: dict[str, Any] = {}
        total: Tensor | None = None

        if run_flow:
            flow_loss, flow_dict = super().forward(batch, reduction=reduction)
            for k, v in flow_dict.items():
                loss_dict[f"flow_{k}"] = v
            loss_dict["flow_loss"] = (
                flow_loss.item() if isinstance(flow_loss, Tensor) and flow_loss.dim() == 0 else float("nan")
            )
            total = self.config.flow_loss_weight * flow_loss

        if run_text:
            text_loss = self._compute_text_loss(batch, text_labels)
            loss_dict["text_loss"] = float(text_loss.detach().item())
            total = (
                self.config.text_loss_weight * text_loss
                if total is None
                else total + self.config.text_loss_weight * text_loss
            )

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

    def _compute_text_loss(self, batch: dict[str, Tensor], text_labels: Tensor) -> Tensor:
        """Cross-entropy on PaliGemma's LM head over the supervised span.

        Re-uses the same prefix-embedding path the flow head does:
        embed images + state + language tokens, run a forward pass,
        slice out the per-token logits at the supervised positions,
        compute CE.
        """
        from torch.nn import functional as F  # noqa: PLC0415

        images, img_masks = self.model._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        # PaliGemma's text path: forward the prefix through the
        # backbone *without* the action expert. We piggy-back on the
        # existing PaliGemmaWithExpertModel.forward — it accepts a
        # list of expert inputs and returns parallel outputs.
        from ..pi05.modeling_pi05 import make_att_2d_masks  # noqa: PLC0415

        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (vlm_out, _), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        if vlm_out is None:
            raise RuntimeError("PI052 text loss: VLM forward returned no hidden states.")

        # Logits over the vocab via the PaliGemma lm_head.
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        logits = lm_head(vlm_out.to(lm_head.weight.dtype))

        # Shift for next-token prediction: predict token[i+1] from
        # hidden[i]. Both ``logits`` and ``text_labels`` are over the
        # same sequence length, so shift logits[:-1] vs labels[1:].
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = text_labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

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

            tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
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

        images, img_masks = self.model._preprocess_images(batch)
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

        for _ in range(max_new_tokens):
            att_2d = make_att_2d_masks(current_pad, current_att)
            position_ids = torch.cumsum(current_pad, dim=1) - 1
            (vlm_out, _), _ = backbone.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[current_embs, None],
                use_cache=False,
                fill_kv_cache=True,
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
