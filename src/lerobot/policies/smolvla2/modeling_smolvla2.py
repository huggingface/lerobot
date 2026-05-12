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
"""SmolVLA2 modeling — dual-head subclass of SmolVLAPolicy.

Adds:

* an unfrozen SmolVLM ``lm_head`` so language tokens can be supervised,
* a forward path that runs the flow head, the text head, or both,
  driven by ``batch["predict_actions"]`` and ``batch["text_labels"]``
  produced by :class:`SmolVLA2ChatTokenizerStep` (the previous commit on
  this branch).

Per-sample routing — within one batch:

* ``predict_actions[i] = True`` ⇒ sample ``i`` contributes to the flow
  loss (action chunk supervision).
* ``predict_actions[i] = False`` ⇒ sample ``i`` is masked out of the
  flow loss; only its text tokens (where ``text_labels[i, t] != -100``)
  contribute to the LM-head cross-entropy.

Falls back to ``SmolVLAPolicy.forward`` cleanly when neither
``text_labels`` nor ``predict_actions`` is in the batch — unannotated
datasets keep working unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from .configuration_smolvla2 import SmolVLA2Config


class SmolVLA2Policy(SmolVLAPolicy):
    """SmolVLA + re-enabled SmolVLM language head."""

    config_class = SmolVLA2Config
    name = "smolvla2"

    def __init__(self, config: SmolVLA2Config, **kwargs):
        if not isinstance(config, SmolVLA2Config):
            config = SmolVLA2Config(
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                    if hasattr(config, f.name)
                }
            )
        super().__init__(config, **kwargs)
        if config.unfreeze_lm_head and config.text_loss_weight > 0:
            self._unfreeze_lm_head()

    # ------------------------------------------------------------------
    # Backbone surgery
    # ------------------------------------------------------------------

    def _unfreeze_lm_head(self) -> None:
        """Re-enable gradients on the SmolVLM ``lm_head`` (and the bits
        of the text path SmolVLA freezes) so the text-loss can flow back.
        """
        vlm_with_expert = getattr(self.model, "vlm_with_expert", None)
        if vlm_with_expert is None:
            return
        vlm = getattr(vlm_with_expert, "vlm", None)
        if vlm is None:
            return
        for name, param in vlm.named_parameters():
            if "lm_head" in name or "text_model.model.norm.weight" in name:
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """Forward pass with optional dual-head loss.

        Two routing knobs from the batch (produced by
        :class:`SmolVLA2ChatTokenizerStep`):

        * ``text_labels`` — per-token labels with ``-100`` for non-target
          positions. Triggers the text-loss path through ``lm_head``.
        * ``predict_actions`` — per-sample bool tensor. ``True`` ⇒
          include this sample's action chunk in the flow loss.

        When neither is present, delegate to ``SmolVLAPolicy.forward``.
        """
        text_labels = batch.get("text_labels")
        predict_actions_t = batch.get("predict_actions")

        has_text_data = (
            text_labels is not None
            and isinstance(text_labels, Tensor)
            and self.config.text_loss_weight > 0
        )
        has_per_sample_routing = (
            predict_actions_t is not None and isinstance(predict_actions_t, Tensor)
        )

        if not has_text_data and not has_per_sample_routing:
            return super().forward(batch, noise=noise, time=time, reduction=reduction)

        loss_dict: dict[str, Any] = {}
        device = batch[OBS_STATE].device
        total = torch.zeros((), device=device, dtype=torch.float32)

        # ------------------------------------------------------------
        # Flow loss path — only when at least one sample wants actions.
        # ------------------------------------------------------------
        run_flow = self.config.flow_loss_weight > 0 and (
            not has_per_sample_routing or bool(predict_actions_t.any().item())
        )
        if run_flow and ACTION in batch:
            per_sample_flow, flow_diag = super().forward(
                batch, noise=noise, time=time, reduction="none"
            )
            # ``per_sample_flow`` has shape (B,) from the SmolVLA
            # reduction="none" branch.
            if has_per_sample_routing:
                mask = predict_actions_t.to(per_sample_flow.dtype)
                masked = per_sample_flow * mask
                denom = mask.sum().clamp(min=1.0)
                flow_loss = masked.sum() / denom
            else:
                flow_loss = per_sample_flow.mean()
            total = total + self.config.flow_loss_weight * flow_loss
            loss_dict["flow_loss"] = float(flow_loss.detach().item())
            for k, v in flow_diag.items():
                loss_dict[f"flow_{k}"] = v

        # ------------------------------------------------------------
        # Text loss path — prefix-only forward → lm_head → CE.
        # ------------------------------------------------------------
        if has_text_data:
            text_loss = self._compute_text_loss(batch, text_labels)
            total = total + self.config.text_loss_weight * text_loss
            loss_dict["text_loss"] = float(text_loss.detach().item())

        loss_dict["loss"] = float(total.detach().item())

        if reduction == "none":
            # Per-sample loss isn't meaningfully defined for the dual
            # path; broadcast the scalar to (B,) for caller compat.
            return total.expand(batch[OBS_STATE].shape[0]), loss_dict
        return total, loss_dict

    # ------------------------------------------------------------------
    # Text-loss internals
    # ------------------------------------------------------------------

    def _compute_text_loss(self, batch: dict[str, Tensor], text_labels: Tensor) -> Tensor:
        """Cross-entropy on the SmolVLM ``lm_head`` over target tokens."""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prefix-only forward.
        out_pair, _ = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        prefix_out = out_pair[0] if isinstance(out_pair, (tuple, list)) else out_pair
        if prefix_out is None:
            raise RuntimeError(
                "SmolVLA2: vlm_with_expert.forward returned no prefix hidden "
                "states — text-loss path needs them."
            )

        # Lang token positions inside the prefix. ``embed_prefix`` lays
        # out the prefix as ``[image_blocks..., lang, state]`` so the
        # lang range is identifiable from the trailing state size and
        # the known lang length.
        num_lang = lang_tokens.shape[1]
        state_for_dim = state if state.ndim >= 2 else state[:, None]
        num_state = state_for_dim.shape[1] if state_for_dim.ndim >= 2 else 1
        if num_state < 1:
            num_state = 1
        prefix_len = prefix_out.shape[1]
        lang_end = prefix_len - num_state
        lang_start = lang_end - num_lang
        if lang_start < 0 or lang_end > prefix_len:
            raise RuntimeError(
                f"SmolVLA2: could not locate lang token range in prefix "
                f"(prefix_len={prefix_len}, num_lang={num_lang}, "
                f"num_state={num_state})."
            )

        vlm = self.model.vlm_with_expert.vlm
        lang_hidden = prefix_out[:, lang_start:lang_end].to(vlm.lm_head.weight.dtype)
        logits = vlm.lm_head(lang_hidden)  # (B, num_lang, vocab)

        if text_labels.shape[1] != num_lang:
            common = min(text_labels.shape[1], num_lang)
            logits = logits[:, :common]
            text_labels = text_labels[:, :common]

        # Standard next-token CE: hidden state at position t predicts
        # token at position t+1. Shift logits left, labels right by 1.
        # Without this, the loss is identity-mapped and the LM head
        # learns nothing useful — see HuggingFace ``LlamaForCausalLM``
        # for the same convention.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = text_labels[:, 1:].contiguous().long()
        valid_labels = shift_labels != -100
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return loss / valid_labels.sum().clamp(min=1)

    # ------------------------------------------------------------------
    # Inference: text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_message(
        self,
        batch: dict[str, Tensor],
        *,
        max_new_tokens: int = 256,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer: Any = None,
    ) -> str:
        """Generate text continuation from the chat-templated prompt.

        AR decoding with KV caching reused from SmolVLA's inference
        path. Batch size is assumed to be 1 (the runtime calls this
        per-event). Returns the decoded string of new tokens (the
        prompt itself is not included).

        Parameters
        ----------
        batch:
            Already through the SmolVLA2 preprocessor — expects
            ``OBS_IMAGES_*``, ``OBS_STATE``, ``OBS_LANGUAGE_TOKENS``,
            ``OBS_LANGUAGE_ATTENTION_MASK``.
        max_new_tokens:
            Hard cap on generated tokens; stops earlier on EOS.
        eos_token_id:
            Override the tokenizer's EOS. ``None`` ⇒ use the
            tokenizer's default.
        temperature, top_p:
            ``temperature=0`` does greedy argmax (default — matches
            training distribution most closely). Set ``temperature>0``
            with optional ``top_p<1`` for nucleus sampling.
        tokenizer:
            Optional pre-loaded tokenizer to avoid the cold-start
            ``AutoTokenizer.from_pretrained`` round-trip on every call.
        """
        self.eval()

        if tokenizer is None:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(self.config.vlm_model_name)
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        # Match training's text-loss forward path (see
        # ``_compute_text_loss`` above): build the full prefix via
        # ``embed_prefix`` so images + state conditioning is intact,
        # then loop AR with ``fill_kv_cache=True, use_cache=False``.
        # That flag combo routes every layer through
        # ``forward_attn_layer`` (which gracefully skips ``None``
        # expert inputs via ``if hidden_states is None or layer is
        # None: continue``) and short-circuits the cache-update logic
        # so we don't have to manage past_kv. Each step just
        # re-forwards the cumulative ``[prefix + generated]``
        # sequence.
        #
        # This is O(n²) in generated text length but cheap in
        # absolute terms: image encoding happens once via the initial
        # ``embed_prefix`` call, and the per-step cost is just one
        # SmolVLM transformer pass over a sequence that grows by one
        # token each time. Standard SmolVLM ``generate`` was the
        # other tempting path, but it can't accept SmolVLA's custom
        # ``state_proj`` output and its tile-grid expectations
        # disagree with our preprocessor — both lead to garbage
        # generation, which is what the prior approach produced.
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        device = prefix_embs.device
        bsize = prefix_embs.shape[0]
        vlm = self.model.vlm_with_expert.vlm
        emb_dim = prefix_embs.shape[-1]
        text_emb_scale = math.sqrt(emb_dim)

        current_embs = prefix_embs
        current_pad = prefix_pad_masks
        current_att = prefix_att_masks
        ones_step = torch.ones((bsize, 1), dtype=torch.bool, device=device)

        generated: list[int] = []
        for _ in range(max_new_tokens):
            full_2d = make_att_2d_masks(current_pad, current_att)
            full_pos = torch.cumsum(current_pad, dim=1) - 1

            out_pair, _ = self.model.vlm_with_expert.forward(
                attention_mask=full_2d,
                position_ids=full_pos,
                past_key_values=None,
                inputs_embeds=[current_embs, None],
                use_cache=False,
                fill_kv_cache=True,
            )
            prefix_out = out_pair[0] if isinstance(out_pair, (tuple, list)) else out_pair
            if prefix_out is None:
                raise RuntimeError(
                    "select_message: vlm_with_expert.forward returned no hidden states."
                )

            last_hidden = prefix_out[:, -1:].to(vlm.lm_head.weight.dtype)
            logits_step = vlm.lm_head(last_hidden)[:, -1]  # (B, V)
            # Suppress EOS until we've decoded ``min_new_tokens`` real
            # tokens. Without this, a memorised LM head whose argmax
            # at position 0 is EOS produces an empty completion every
            # time — confirmed in the real-robot run (the runtime's
            # ``subtask_empty_count`` climbed every chunk boundary
            # with no exception). Masking EOS for the first N steps
            # forces the head to commit to a real token before it can
            # close the turn.
            if (
                eos_token_id is not None
                and len(generated) < min_new_tokens
            ):
                logits_step[..., eos_token_id] = float("-inf")
            next_ids = self._sample_next_token(logits_step, temperature, top_p)
            tok_id = int(next_ids[0].item())
            generated.append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id:
                break

            new_emb = self.model.vlm_with_expert.embed_language_tokens(
                next_ids.unsqueeze(0)
            )
            new_emb = new_emb * text_emb_scale
            current_embs = torch.cat([current_embs, new_emb], dim=1)
            current_pad = torch.cat([current_pad, ones_step], dim=1)
            current_att = torch.cat([current_att, ones_step], dim=1)

        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _sample_next_token(
        logits: Tensor, temperature: float, top_p: float
    ) -> Tensor:
        """Pick one token id per batch row from ``logits``."""
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-6)
        probs = F.softmax(scaled, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            mask = cum > top_p
            # Always keep the most-likely token.
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            pick = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_idx.gather(-1, pick).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
