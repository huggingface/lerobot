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

"""SmolVLA-KI modeling. See configuration_smolvla_ki.py for the recipe.

Two training objectives on the SmolVLM2-500M backbone:
  * flow-matching action expert (cross-attending the first `expert_attend_layers`
    VLM layers) — the original SmolVLA action head;
  * an autoregressive FAST action-token cross-entropy on the VLM's own lm_head
    (injects spatial/action understanding into the backbone),
with a stop-gradient ("knowledge insulation") on the VLM features the expert
reads, so the freshly-initialised expert never corrupts the backbone.
"""

import torch
import torch.nn as nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from ..smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from .configuration_smolvla_ki import SmolVLAKIConfig


class _DetachKV(nn.Module):
    """Wrap a Linear so its INPUT is detached before the projection.

    Placed on the action expert's cross-attention ``k_proj`` / ``v_proj``, whose
    inputs are the VLM key/value states. Detaching the input cuts the gradient
    path from the (flow-matching) action expert back into the VLM backbone —
    this is the "knowledge insulation" stop-gradient (π0.5 KI, §III.B).
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    # The parent forward reads `k_proj.weight.dtype` / `v_proj.weight.dtype`
    # directly, so proxy these to the wrapped Linear.
    @property
    def weight(self) -> torch.Tensor:
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.detach())


class SmolVLMWithExpertModelKI(SmolVLMWithExpertModel):
    """SmolVLM2 + action expert for the KI recipe (Option A):

    * **full VLM** kept (``num_vlm_layers=-1``) so its native ``lm_head`` stays
      in-distribution for the FAST token loss;
    * the action expert has ``expert_attend_layers`` layers and cross-attends only
      the **first** ``expert_attend_layers`` VLM layers (contiguous), via the
      overridden :meth:`get_model_layers`; VLM layers beyond run standalone;
    * optional stop-gradient on the expert's cross-attn KV (knowledge insulation).
    """

    def __init__(
        self,
        *args,
        expert_attend_layers: int = 16,
        knowledge_insulation: bool = True,
        enable_fast_action_loss: bool = True,
        **kwargs,
    ):
        # Force: keep full VLM, expert depth == expert_attend_layers.
        kwargs.pop("num_vlm_layers", None)
        kwargs.pop("num_expert_layers", None)
        self._enable_fast_action_loss = enable_fast_action_loss
        super().__init__(
            *args,
            num_vlm_layers=-1,
            num_expert_layers=expert_attend_layers,
            **kwargs,
        )
        self.expert_attend_layers = expert_attend_layers
        self.knowledge_insulation = knowledge_insulation

        if knowledge_insulation and "cross" in self.attention_mode:
            # Wrap exactly the cross-attention expert projections (the same layers
            # the parent reshaped to read VLM-dim KV — i.e. the non-self-attn ones).
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                attn = self.lm_expert.layers[layer_idx].self_attn
                attn.k_proj = _DetachKV(attn.k_proj)
                attn.v_proj = _DetachKV(attn.v_proj)

        # set_requires_grad ran inside super().__init__ via VLAFlowMatching? No —
        # the parent SmolVLMWithExpertModel.__init__ calls self.set_requires_grad()
        # at the end (line ~145), which dispatches to OUR override below.

    def set_requires_grad(self):
        # Vision encoder
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for p in self.get_vlm_model().vision_model.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            # Hard-frozen VLM (arms A0 / A1-finetune): no VLM gradients at all.
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False
        else:
            # KI (arm A2): the VLM IS trained — by the FAST/text token loss — and
            # insulated from the expert via _DetachKV. Keep the WHOLE VLM incl.
            # lm_head trainable (unlike the parent, which freezes lm_head + the last
            # 1-2 layers; that would kill the FAST autoregressive objective).
            for p in self.vlm.parameters():
                p.requires_grad = True
        # The action expert's own (unused) lm_head stays frozen.
        for name, p in self.lm_expert.named_parameters():
            if "lm_head" in name:
                p.requires_grad = False

    def get_model_layers(self, models: list) -> list:
        # Contiguous coupling: expert layer i <-> VLM layer i for i < expert_attend_layers.
        # Beyond that the expert layer is None, so the VLM advances alone (its hidden
        # state feeds the lm_head; the expert stream is carried unchanged to its norm).
        vlm_layers = []
        expert_layers = []
        n_expert = len(models[1].layers)
        for i in range(self.num_vlm_layers):
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(models[1].layers[i] if i < min(self.expert_attend_layers, n_expert) else None)
        return [vlm_layers, expert_layers]

    # NOTE (partial insulation): the interleaved self-attention layers (every
    # `self_attn_every_n_layers`) still mix VLM/expert KV in `forward_attn_layer`,
    # so a small residual gradient path expert->VLM remains there. The dominant
    # cross-attention path is fully insulated. TODO: route the self-attn layers
    # through detached VLM KV too for strict insulation if ablations call for it.

    def forward(self, *args, **kwargs):
        """At inference, only run the first ``expert_attend_layers`` VLM layers.

        The VLM layers beyond ``expert_attend_layers`` exist ONLY to feed the
        ``lm_head`` for the training-time FAST token loss — the action expert reads
        just the first ``expert_attend_layers`` layers, and beyond them those expert
        slots are ``None`` so the expert hidden state is passed through unchanged
        (see ``forward`` lines that append ``hidden_states`` when ``layer is None``).
        Hence the expert's final output is bit-identical whether or not the extra
        VLM layers run.

        At train time we DO run them (full ``num_vlm_layers``) because the FAST loss
        needs the full-depth VLM output under ``lm_head``. But at inference (KV-cache,
        suffix-only denoise) the VLM contributes no new query in its self-attention
        layers, and beyond ``expert_attend_layers`` the expert has no layer either —
        so ``forward_attn_layer`` would build an empty query list and hit
        ``torch.cat([])``. Capping the layer count to ``expert_attend_layers`` both
        avoids that crash and halves deploy-time compute (the intended design).
        """
        if not self.training:
            orig = self.num_vlm_layers
            self.num_vlm_layers = min(self.expert_attend_layers, orig)
            try:
                return super().forward(*args, **kwargs)
            finally:
                self.num_vlm_layers = orig
        return super().forward(*args, **kwargs)


class VLAFlowMatchingKI(VLAFlowMatching):
    """Same as :class:`VLAFlowMatching` but builds the KI expert model and adds the
    FAST autoregressive action-token loss head.

    The __init__ mirrors the parent (so the projection sizes stay identical —
    expert_hidden_size and text hidden_size are unchanged) and only swaps the
    backbone class to inject the full-VLM + decoupled-expert-depth + KI behavior.
    """

    def __init__(self, config: SmolVLAKIConfig, rtc_processor=None):
        nn.Module.__init__(self)
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModelKI(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device=config.device if config.device is not None else "auto",
            expert_attend_layers=config.expert_attend_layers,
            knowledge_insulation=config.knowledge_insulation,
            enable_fast_action_loss=config.enable_fast_action_loss,
        )
        self.state_proj = nn.Linear(
            config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = config.prefix_length
        self.rtc_processor = rtc_processor

        # ── FAST autoregressive action-token head ────────────────────────────
        self.enable_fast_action_loss = config.enable_fast_action_loss
        self._fast_tokenizer = None
        if self.enable_fast_action_loss:
            from transformers import AutoProcessor

            try:
                self._fast_tokenizer = AutoProcessor.from_pretrained(
                    config.action_tokenizer_name, trust_remote_code=True
                )
            except Exception as e:  # noqa: BLE001
                # The FAST tokenizer is ONLY used by forward_fast at TRAIN time. At
                # inference (eval) it is never touched, so a load failure here — e.g.
                # HF_HUB_OFFLINE with an incompletely cached tokenizer — must not break
                # eval. Leave it None; forward_fast raises clearly if training needs it.
                import logging

                logging.warning(
                    f"FAST tokenizer '{config.action_tokenizer_name}' failed to load "
                    f"({type(e).__name__}); fine for inference, training would fail."
                )
                self._fast_tokenizer = None
            self.fast_skip_tokens = config.fast_skip_tokens
            self.max_action_tokens = config.max_action_tokens
            self.vlm_vocab_size = self.vlm_with_expert.vlm.config.text_config.vocab_size

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    @torch.no_grad()
    def _tokenize_action_chunk(self, actions: torch.Tensor, action_dim: int):
        """``actions``: (B, chunk, max_action_dim), normalised. Returns ``(ids, mask)``
        each (B, max_action_tokens): FAST tokens mapped into the VLM vocab's high
        (rarely-used) region, right-padded; ``mask`` True for real tokens."""
        device = actions.device
        a = actions[:, :, :action_dim].detach().float().cpu().numpy()
        bsize = a.shape[0]
        seq = self.max_action_tokens
        pad_id = self.vlm_vocab_size - 1
        ids = torch.full((bsize, seq), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros(bsize, seq, dtype=torch.bool, device=device)
        for i in range(bsize):
            toks = self._fast_tokenizer(a[i])  # FAST DCT+BPE encode -> token ids
            toks = torch.as_tensor(toks, dtype=torch.long).flatten()
            # Map FAST ids into the VLM vocab high region (cf. pi0_fast).
            vlm_ids = (self.vlm_vocab_size - 1 - self.fast_skip_tokens - toks).clamp_(0, self.vlm_vocab_size - 1)
            n = min(vlm_ids.numel(), seq)
            ids[i, :n] = vlm_ids[:n].to(device)
            mask[i, :n] = True
        return ids, mask

    def forward_fast(self, images, img_masks, lang_tokens, lang_masks, state, actions, action_dim) -> torch.Tensor:
        """Autoregressive FAST action-token CE loss on the VLM lm_head, conditioned
        on [images, language, state]. Uses the HF VLM's native causal-LM forward
        (shifted CE; prefix positions masked with -100).

        NB: builds UNSCALED embeddings — the HF-native LM forward does not expect
        the sqrt(dim) scaling that smolvla's custom `embed_prefix` applies for its
        own interleaved forward.
        """
        fast_ids, fast_mask = self._tokenize_action_chunk(actions, action_dim)

        # The HF VLM (and its lm_head) run in their loaded dtype (bfloat16); cast
        # every embedding to it so concat + the native forward stay consistent.
        vlm_dtype = self.vlm_with_expert.vlm.dtype

        embs, masks = [], []
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.vlm_with_expert.embed_image(img).to(vlm_dtype)
            b, n = img_emb.shape[:2]
            embs.append(img_emb)
            masks.append(img_mask[:, None].expand(b, n))
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens).to(vlm_dtype)
        embs.append(lang_emb)
        masks.append(lang_masks)
        state_emb = self.state_proj(state).to(vlm_dtype)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        masks.append(
            torch.ones(state_emb.shape[0], state_emb.shape[1], dtype=torch.bool, device=state_emb.device)
        )
        prefix_emb = torch.cat(embs, dim=1)
        prefix_mask = torch.cat(masks, dim=1)
        prefix_len = prefix_emb.shape[1]

        fast_emb = self.vlm_with_expert.embed_language_tokens(fast_ids).to(vlm_dtype)
        inputs_embeds = torch.cat([prefix_emb, fast_emb], dim=1)
        attn_mask = torch.cat([prefix_mask, fast_mask], dim=1).long()

        labels = torch.full(
            (fast_ids.shape[0], inputs_embeds.shape[1]), -100, dtype=torch.long, device=inputs_embeds.device
        )
        labels[:, prefix_len:] = torch.where(fast_mask, fast_ids, torch.full_like(fast_ids, -100))

        out = self.vlm_with_expert.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )
        return out.loss


class SmolVLAKIPolicy(SmolVLAPolicy):
    """SmolVLA with Knowledge-Insulation. Reuses SmolVLAPolicy; swaps in
    :class:`VLAFlowMatchingKI` and adds the weighted FAST action-token loss."""

    config_class = SmolVLAKIConfig
    name = "smolvla_ki"

    def __init__(self, config: SmolVLAKIConfig, **kwargs):
        from lerobot.utils.import_utils import require_package

        require_package("transformers", extra="smolvla")
        # Skip SmolVLAPolicy.__init__ (it would build the non-KI model); replicate
        # its body with the KI flow-matching model.
        super(SmolVLAPolicy, self).__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatchingKI(config, rtc_processor=self.rtc_processor)
        self.reset()

    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        """Flow-matching loss (+ weighted FAST action-token loss when enabled).

        A1 stage-1 (``fast_pretrain_only``): compute ONLY the FAST action-token loss
        to pretrain the VLM via its lm_head; the flow expert is skipped entirely.
        """
        if getattr(self.config, "fast_pretrain_only", False):
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            actions = self.prepare_action(batch)
            action_dim = self.config.action_feature.shape[0]
            fast_loss = self.model.forward_fast(
                images, img_masks, lang_tokens, lang_masks, state, actions, action_dim
            )
            return fast_loss, {"loss": float(fast_loss), "fast_action_loss": float(fast_loss)}

        flow_loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)
        total = self.config.flow_loss_weight * flow_loss
        loss_dict["flow_loss"] = float(flow_loss.mean()) if hasattr(flow_loss, "mean") else float(flow_loss)

        if reduction == "mean" and getattr(self.config, "enable_fast_action_loss", False) and self.model.enable_fast_action_loss:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            actions = self.prepare_action(batch)
            action_dim = self.config.action_feature.shape[0]
            fast_loss = self.model.forward_fast(
                images, img_masks, lang_tokens, lang_masks, state, actions, action_dim
            )
            total = total + self.config.fast_action_loss_weight * fast_loss
            loss_dict["fast_action_loss"] = float(fast_loss)

        loss_dict["loss"] = float(total) if total.dim() == 0 else float(total.mean())
        return total, loss_dict
