from __future__ import annotations

import functools
import os
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_lingbot_vla_v2 import (
    LingbotVLAV2Config as LeRobotLingbotVLAV2Config,
    resolve_robot_config_and_stats,
)
from .model_core.flex_attention import (
    build_block_mask,
    flex_attention_forward,
    flex_attention_with_block_mask,
)
from .model_core.modeling_lingbot_vla_v2_base import (
    FlowMatching as FlowMatchingV1,
    replace_lnorm_with_adanorm,
)
from .model_core.moe_loss import sequence_wise_balance_loss as triton_sequence_wise_balance_loss
from .model_core.qwen2_action_expert import (
    Qwen2ForCausalLM,
    Qwen2TokenMoeBlock,
)
from .model_core.qwen3vl_in_vla import (
    Qwen3VLForConditionalGeneration,
    apply_lingbot_qwen3_vl_patch,
    apply_rotary_pos_emb,
)
from .model_core.utils import (
    block_suffix_to_fv_,
    flash_varlen_prefix_attention,
    make_att_2d_masks,
    our_eager_attention_forward,
    our_sdpa_attention_forward,
    prefix_query_segments,
    prefix_query_token_spans,
)

try:
    from dinov3.hub.backbones import dinov3_vitb16
except ImportError:
    dinov3_vitb16 = None


logger = logging.get_logger(__name__)


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 0,
        use_lm_head: bool = False,
        attention_implementation: str = "flex_cached",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        num_layers = 36

        self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=expert_hidden_size,
            head_dim=action_head_dim,
            initializer_range=0.02,
            intermediate_size=expert_intermediate_size,
            max_position_embeddings=32768,
            max_window_layers=21,
            model_type="qwen2",
            num_attention_heads=action_num_attention_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=action_num_key_value_heads,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.57.3",
            use_cache=use_cache,
            use_sliding_window=False,
            vocab_size=151936,
        )
        logger.debug(
            "Initializing Action Expert V2: layers=%s, hidden=%s, q_heads=%s, kv_heads=%s, head_dim=%s",
            num_layers,
            expert_hidden_size,
            action_num_attention_heads,
            action_num_key_value_heads,
            action_head_dim,
        )
        super().__init__(**kwargs)


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, eval=False):
        super().__init__(config=config)
        self.config = config
        # The model relies on the patched Qwen3-VL classes (custom text decoder
        # layer / vision forward signature); apply the patch idempotently here so
        # building the model directly (without the processor) works as well.
        apply_lingbot_qwen3_vl_patch()
        # Map our attention_implementation to a transformers-valid attn class for the
        # HF model instantiation. "fa2" -> flash_attention_2; everything else (eager /
        # flex / flex_cached) builds with "eager" — the flex paths override attention in
        # the custom forward, and eager is required where flash-attn is absent (Jetson,
        # CPU, this A100 box without the flash_attn package).
        hf_attn = "flash_attention_2" if self.config.attention_implementation == "fa2" else "eager"
        # The vision tower reads the value straight through, so "fa2" needs the same
        # translation here (eager / sdpa are already transformers-valid).
        hf_vit_attn = (
            "flash_attention_2"
            if self.config.vit_attn_implementation == "fa2"
            else self.config.vit_attn_implementation
        )
        vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path)
        if self.config.vocab_size not in (0, 257152):
            vlm_config.text_config.vocab_size = self.config.vocab_size
        vlm_config._attn_implementation = hf_attn
        vlm_config.text_config._attn_implementation = hf_attn
        vlm_config.vision_config._attn_implementation = hf_vit_attn
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()

        self.config.qwen_expert_config._attn_implementation = hf_attn
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )

        self._install_moe_blocks()
        self.pos_embeds = None
        self.position_embeddings = None
        self.cu_seqlens = None
        self.visual_split_sizes = None
        self.visual_max_seqlen = None
        # Capture-context flag: set by the full-prefix CUDA-graph wrapper so the vision
        # grid metadata (pos_embeds / cu_seqlens / split_sizes / max_seqlen) is hoisted
        # out of the per-call host-sync path and cached once. See ``_prefix_graphed``.
        self._capture_grid_cache = False

        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            if dinov3_vitb16 is None:
                raise ImportError("dinov3 is required when enable_expert_vision=True")
            if "dinov3_vitb16" in self.config.expert_vision_type:
                self.expert_visual = dinov3_vitb16(pretrained=False)
            self.expert_visual_mlp = nn.Sequential(
                nn.Linear(self.expert_visual.embed_dim, self.expert_visual.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.expert_visual.embed_dim * 2, self.config.qwen_expert_config.hidden_size),
            )

        self.attention_interface = self.get_attention_interface()
        self.set_requires_grad()

    def _install_moe_blocks(self):
        if not getattr(self.config, "use_moe", False):
            return
        bias_update_speed = getattr(self.config, "bias_update_speed", 0.001)
        hidden_size = self.config.qwen_expert_config.hidden_size
        token_moe_layers = getattr(self.config, "token_moe_layers", None) or []

        _moe_impl = getattr(self.config, "_moe_implementation", None)

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=getattr(self.config, "token_num_experts", 32),
                num_experts_per_tok=getattr(self.config, "token_top_k", 1),
                norm_topk_prob=True,
                hidden_size=hidden_size,
                moe_intermediate_size=getattr(self.config, "token_moe_intermediate_size", 256),
                shared_expert_intermediate_size=getattr(self.config, "token_shared_intermediate_size", 256),
                output_router_logits=False,
            )
            token_config.bias_update_speed = bias_update_speed
            token_config._moe_implementation = _moe_impl
            token_config.moe_dense_max_tokens = getattr(self.config, "moe_dense_max_tokens", 512)
            token_config.moe_backend = getattr(self.config, "moe_backend", "sparse_static")
            token_config.router_activation = getattr(self.config, "router_activation", "softmax")
            token_config.routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
            token_config.use_shared_expert_gate = getattr(self.config, "use_shared_expert_gate", True)
            for idx in token_moe_layers:
                self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwenvl.model.visual.eval()
            for params in self.qwenvl.model.visual.parameters():
                params.requires_grad = False
        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.model.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        precompute_grid_thw = getattr(self.config, "precompute_grid_thw", False)
        # Hoist the host-syncing grid preprocess when (a) the precompute flag wants it
        # cached and it is not yet, or (b) the capture grid cache is armed but empty
        # (first warm-up pass of a vision-graph capture). Once populated, subsequent
        # calls — including the CUDA-graph capture itself — skip preprcess_grid_thw
        # entirely, which is what makes the tower capturable (its .item()/.tolist()
        # host syncs cannot be captured).
        grid_cache_ready = self.position_embeddings is not None
        if (precompute_grid_thw and not grid_cache_ready) or (
            self._capture_grid_cache and not grid_cache_ready
        ):
            (
                self.pos_embeds,
                self.position_embeddings,
                self.cu_seqlens,
                self.visual_split_sizes,
                self.visual_max_seqlen,
            ) = self.qwenvl.model.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
        image_embeds, deepstack_image_embeds = self.qwenvl.model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            pos_embeds=self.pos_embeds,
            position_embeddings=self.position_embeddings,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.visual_max_seqlen,
        )
        split_sizes = self.visual_split_sizes
        if split_sizes is None:
            split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.model.visual.spatial_merge_size**2).tolist()
        image_chunks = list(torch.split(image_embeds, split_sizes))
        deepstack_chunks = [
            list(torch.split(deepstack_embeds, split_sizes)) for deepstack_embeds in deepstack_image_embeds
        ]
        image_embeds = torch.stack(image_chunks, dim=0)
        deepstack_image_embeds = [torch.stack(chunks, dim=0) for chunks in deepstack_chunks]
        return image_embeds, deepstack_image_embeds

    def embed_image(self, image: torch.Tensor, image_grid_thw: torch.LongTensor):
        return self.get_image_features(
            image,
            image_grid_thw=image_grid_thw,
        )

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(self, token_id: int, batch: int, count: int, device, dtype):
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        emb = self.embed_language_tokens(token).to(dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def build_prefix_position_ids(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
        # transformers>=5.5 externalized modality detection: get_rope_index now takes an
        # explicit ``mm_token_type_ids`` (0=text, 1=image, 2=video) instead of matching
        # the placeholder token ids internally. Reconstruct it from the vision token ids.
        vlm_cfg = self.qwenvl.config
        image_token_id = getattr(vlm_cfg, "image_token_id", None)
        video_token_id = getattr(vlm_cfg, "video_token_id", None)
        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        # transformers 4.57 (which the 6B checkpoint was trained under) filled masked
        # (padding) rope positions with 1; transformers 5.5 fills them with 0. Restore the
        # 4.57 convention so cached-prefix rope matches the trained weights exactly.
        if attention_mask is not None:
            pad = attention_mask == 0
            while pad.dim() < position_ids.dim():
                pad = pad.unsqueeze(0)
            position_ids = position_ids.masked_fill(pad.expand_as(position_ids), 1)
        return position_ids

    def apply_mrope(self, query_states, key_states, position_ids=None, position_embeddings=None):
        if position_embeddings is None:
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, *position_embeddings, unsqueeze_dim=2)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
        return key_states, value_states, past_key_values

    def _apply_deepstack(self, hidden_states, layer_idx, visual_pos_masks, deepstack_visual_embeds):
        """Add the level's dense deepstack delta.

        The embeds arrive pre-laid-out over the full prefix length (zeros at
        non-visual positions; see :meth:`embed_prefix`), so the injection is a
        plain shape-static add — no bool indexing, hence no nonzero()/host sync
        and CUDA-graph capturable. ``visual_pos_masks`` is kept for signature
        compatibility; its nonzero rows agree with the dense layout by
        construction. Numerically identical to the transformers
        ``_deepstack_process`` scatter it replaces (x + 0.0 == x).
        """
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            hidden_states = hidden_states + deepstack_visual_embeds[layer_idx].to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        vlm_position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: list[torch.FloatTensor] = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask=None,
    ):
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers
        action_num_layers = self.config.qwen_expert_config.num_hidden_layers
        router_logits_list = []

        assert action_num_layers == num_layers, (
            "Action expert and VLM must have the same number of layers "
            f"(got action={action_num_layers}, vlm={num_layers})."
        )

        # Attention runs in the model (half) dtype by default; attention_fp32=True
        # restores the original fp32 upcast used for bit-exact parity checks.
        attn_fp32 = getattr(self.config, "attention_fp32", False)

        # mrope cos/sin depend only on position_ids (and dtype/device) — compute once
        # per forward instead of once per layer. Callers with a loop-invariant
        # position_ids (e.g. the flow-matching denoise loop) can pass a precomputed
        # ``position_embeddings`` to skip this entirely.
        if position_embeddings is None:
            rep = next(h for h in inputs_embeds if h is not None)
            # rotary_emb casts cos/sin to the representative tensor's dtype; under
            # attention_fp32 keep the old fp32 cos/sin for bit-exact parity.
            if attn_fp32:
                rep = rep.float()
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(rep, position_ids)

        _full_block_mask = block_mask
        if _full_block_mask is None and self.config.attention_implementation == "flex_cached":
            # Build once per forward (not per layer). q_len is the concatenated stream
            # length; with a filled KV cache the kv side additionally covers the prefix.
            q_len = sum(h.shape[1] for h in inputs_embeds if h is not None)
            kv_len = q_len
            if use_cache and not fill_kv_cache and past_key_values:
                kv_len += past_key_values[0]["key_states"].shape[1]
            _full_block_mask = build_block_mask(
                attention_mask,
                self.qwenvl.config.text_config.num_attention_heads,
                q_len,
                kv_len,
            )

        use_gradient_checkpointing = (
            getattr(self.config, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
            and not use_cache
        )

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds, layer_router_logits = self._checkpointed_layer(
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_embeddings,
                    ada_cond,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    _full_block_mask,
                    attn_fp32,
                )
                router_logits_list.extend(layer_router_logits)
                continue
            inputs_embeds, layer_router_logits, _full_block_mask, past_key_values = self._layer_forward(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_embeddings,
                past_key_values,
                use_cache,
                fill_kv_cache,
                ada_cond,
                visual_pos_masks,
                deepstack_visual_embeds,
                _full_block_mask,
                attn_fp32,
            )
            router_logits_list.extend(layer_router_logits)

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb, _ = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values, router_logits_list

    def _layer_forward(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_embeddings,
        past_key_values,
        use_cache,
        fill_kv_cache,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
        block_mask,
        attn_fp32,
    ):
        """One dual-stream layer: per-stream QKV -> joint attention -> per-stream out/MLP."""
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        router_logits_list = []
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            if i == 1:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ada_cond=ada_cond)
            else:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
            if attn_fp32:
                q, k, v = q.float(), k.float(), v.float()
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        query_states, key_states = self.apply_mrope(
            query_states, key_states, position_embeddings=position_embeddings
        )
        key_states, value_states, past_key_values = self.handle_kv_cache(
            key_states,
            value_states,
            layer_idx,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        if self.config.attention_implementation == "flex_cached":
            if block_mask is None:
                block_mask = build_block_mask(
                    attention_mask,
                    self.qwenvl.config.text_config.num_attention_heads,
                    query_states.shape[1],
                    key_states.shape[1],
                )
            att_output = flex_attention_with_block_mask(
                query_states,
                key_states,
                value_states,
                block_mask,
                query_states.shape[1],
                force_fp32=attn_fp32,
            )
        elif self.config.attention_implementation == "flex":
            att_output = flex_attention_forward(
                query_states, key_states, value_states, attention_mask, force_fp32=attn_fp32
            )
        else:
            split = (
                getattr(self.config, "attn_split_prefix_suffix", False)
                and inputs_embeds
                and inputs_embeds[0] is not None
                and attention_mask is not None
                and attention_mask.dim() == 3
            )
            if split:
                prefix_len = inputs_embeds[0].shape[1]
                # The prefix (VLM) stream must have NO trainable params (expert-only
                # LoRA): detach its Q/K/V so autograd never tracks the frozen VLM
                # activations (the joint call tracks them spuriously via the
                # concatenated Q/K/V and the gradient dies at the frozen params).
                # Guard: if any VLM param is trainable (A-arm regex / full FT), keep
                # the graph — the prefix path then carries real adapter gradients.
                models_local = [self.qwenvl.model.language_model, self.qwen_expert.model]
                detach_prefix = not inputs_embeds[0].requires_grad and not any(
                    p.requires_grad for p in models_local[0].parameters()
                )
                q_p, k_p, v_p = (
                    query_states[:, :prefix_len],
                    key_states[:, :prefix_len],
                    value_states[:, :prefix_len],
                )
                if detach_prefix:
                    q_p, k_p, v_p = q_p.detach(), k_p.detach(), v_p.detach()
                mask_p = attention_mask[:, :prefix_len, :prefix_len]
                if (
                    getattr(self.config, "attn_split_prefix_backend", "flash") == "flash"
                    and q_p.dtype in (torch.bfloat16, torch.float16)
                ):
                    att_p = flash_varlen_prefix_attention(q_p, k_p, v_p, mask_p)
                else:
                    att_p = self.attention_interface(q_p, k_p, v_p, mask_p)
                att_s = self.attention_interface(
                    query_states[:, prefix_len:], key_states, value_states, attention_mask[:, prefix_len:, :]
                )
                att_output = torch.cat([att_p, att_s], dim=1)
            else:
                att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            if i == 1:
                out_emb, router_logits = models[i].layers[layer_idx](
                    hidden_states,
                    att_output,
                    start,
                    end,
                    output_atten=True,
                    ada_cond=ada_cond,
                )
                if router_logits is not None:
                    router_logits_list.append(router_logits)
            else:
                # Under the split with a grad-free prefix stream, hand the VLM
                # stream a detached att_output: without this, cat([no-grad prefix,
                # grad suffix]) re-couples gradients into the prefix rows and the
                # whole frozen-VLM backward keeps running. Values are identical;
                # the prefix gradient is discarded at frozen params anyway.
                att_output_stream = (
                    att_output.detach() if (i == 0 and split and detach_prefix) else att_output
                )
                out_emb = models[i].layers[layer_idx](
                    hidden_states, att_output_stream, start, end, output_atten=True
                )
                out_emb = self._apply_deepstack(out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds)
            outputs_embeds.append(out_emb)
            start = end
        return outputs_embeds, router_logits_list, block_mask, past_key_values

    def _checkpointed_layer(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_embeddings,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
        block_mask,
        attn_fp32,
    ):
        """Gradient-checkpointed layer step (training only, KV cache disabled)."""
        outputs_embeds, router_logits_list, _, _ = torch_checkpoint(
            self._layer_forward,
            layer_idx,
            inputs_embeds,
            attention_mask,
            position_embeddings,
            None,  # past_key_values
            False,  # use_cache
            False,  # fill_kv_cache
            ada_cond,
            visual_pos_masks,
            deepstack_visual_embeds,
            block_mask,
            attn_fp32,
            use_reentrant=False,
        )
        return outputs_embeds, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "flex":
            logger.debug("Using Flex attention")
            return flex_attention_forward
        if self.config.attention_implementation == "flex_cached":
            logger.debug("Using Flex Cached attention with prebuilt BlockMask")
            return flex_attention_forward
        if self.config.attention_implementation == "sdpa":
            sdpa_backend = getattr(self.config, "sdpa_backend", None)
            # cuDNN SDPA's backward kernel produces NaN gradients for this model
            # (bf16, these seq lens / head dims) on torch 2.8 / cuDNN: every step's
            # forward is fine but step 1's backward writes NaN into the weights and
            # the whole run goes NaN. It is a *forward/inference-only* fast path —
            # silently fall back to torch auto-selection under autograd so training
            # can never enable it.
            if sdpa_backend is not None and torch.is_grad_enabled():
                logger.warning(
                    f"sdpa_backend={sdpa_backend} is inference-only (cuDNN SDPA backward "
                    f"returns NaN grads); using torch auto-selected SDPA backend for training."
                )
                sdpa_backend = None
            if sdpa_backend is not None:
                return functools.partial(our_sdpa_attention_forward, sdpa_backend=sdpa_backend)
            return our_sdpa_attention_forward
        if self.config.attention_implementation == "eager":
            logger.debug("Using Eager attention")
            return our_eager_attention_forward
        raise ValueError(f"Invalid attention implementation: {self.config.attention_implementation}")


class FlowMatchingV2(FlowMatchingV1):
    def __init__(self, config, eval):
        nn.Module.__init__(self)
        self.config = config
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
            expert_hidden_size=getattr(self.config, "expert_hidden_size", 768),
            expert_intermediate_size=getattr(self.config, "expert_intermediate_size", 2752),
            action_num_attention_heads=getattr(self.config, "action_num_attention_heads", 32),
            action_num_key_value_heads=getattr(self.config, "action_num_key_value_heads", 8),
            action_head_dim=getattr(self.config, "action_head_dim", 128),
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "precompute_grid_thw",
            "vit_attn_implementation",
            "attention_fp32",
            "sdpa_backend",
            "attn_split_prefix_suffix",
            "attn_split_prefix_backend",
            "gradient_checkpointing",
            "use_moe",
            "bias_update_speed",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
            "_moe_implementation",
            "moe_dense_max_tokens",
            "moe_backend",
        ]:
            if hasattr(config, name):
                setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(
            qwenvl_with_export_config.qwen_expert_config, "initializer_range", None
        )

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.config.align_params = getattr(self.config, "align_params", None) or {}
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
            self.use_future_video = self.config.align_params.get("use_future_video", False)
            if self.use_future_video:
                self.init_video_heads(self.config.align_params)
        else:
            self.use_depth_align = False
            self.use_future_video = False
            self.use_future_video_patch = False
            self.use_current_video_patch = False
            self.use_current_shared_task_proj = False
            self.use_future_video_cls = False
            self.use_shared_future_task_proj = False
            self.future_video_share_future_depth_query = False
            self.block_future_depth_to_action = False

        self.set_requires_grad()

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
        vision_outputs=None,
    ):
        if image_grid_thw is None:
            raise ValueError("LingbotVLAV2Policy requires image_grid_thw from the Qwen3-VL image processor.")
        bsize = images.shape[0]
        device = images.device
        if images.ndim == 3:
            bsize = 1
            num_images = images.shape[0]
        else:
            num_images = images.shape[1] if images.ndim >= 4 else 1
        if images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if image_grid_thw.ndim == 3:
            flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        else:
            flat_grid_thw = image_grid_thw

        if vision_outputs is None:
            img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(
                images,
                flat_grid_thw,
            )
        else:
            # Pre-computed vision tower outputs (e.g. from the captured vision graph in
            # the use_cudagraph_prefix_full path) — skip the eager ViT pass. Same
            # ``(b n) l d`` flat layout as embed_image returns.
            img_emb, deepstack_embs = vision_outputs
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images)
        deepstack_embs = [
            einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images) for x in deepstack_embs
        ]
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id

        if getattr(self.config, "qwen3vl_use_vision_boundaries", True):
            start_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
            )
            end_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
            )
            img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
            image_token_len = num_patch + 2
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = torch.zeros_like(image_pad_masks)
            image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(
                img_masks, "b n -> b n l", l=num_patch
            )
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )
            fake_image_ids[:, :, 0] = cfg.vision_start_token_id
            fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        else:
            img_chunks = img_emb
            image_token_len = num_patch
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = image_pad_masks
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )

        img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        visual_pos_masks = einops.rearrange(image_visual_masks, "b n l -> b (n l)")
        fake_image_ids = einops.rearrange(fake_image_ids, "b n l -> b (n l)")

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(dtype=embed_dtype)

        if self.use_depth_align and self.align_type == "query":

            def _get_align_tokens(tokens):
                tk_weights = tokens.view(
                    self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1]
                )
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_pad_masks = torch.ones(bsize, self.num_task_tokens, device=device, dtype=lang_masks.dtype)
            fake_align_ids = torch.full(
                (bsize, self.num_task_tokens), cfg.text_config.eos_token_id, dtype=torch.long, device=device
            )

            current_task = _get_align_tokens(self.depth_align_embs)
            if (
                getattr(self, "use_future_video", False)
                and getattr(self, "use_current_video_patch", False)
                and getattr(self, "use_current_shared_task_proj", False)
            ):
                current_video_task = _get_align_tokens(self.current_video_align_embs)
                current_task = self.current_shared_task_proj(
                    torch.cat([current_task, current_video_task], dim=-1)
                )
            align_embs = current_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
            parts = [img_emb]
            masks = [image_pad_masks]
            input_ids = [fake_image_ids]
            visual_masks = [visual_pos_masks]

            def _append(
                tokens,
                token_masks,
                token_ids,
                token_visual_masks=None,
            ):
                parts.append(tokens)
                masks.append(token_masks)
                input_ids.append(token_ids)
                if token_visual_masks is None:
                    token_visual_masks = torch.zeros_like(token_masks)
                visual_masks.append(token_visual_masks)

            future_align_embs = None
            if self.use_future_depth:
                future_task = _get_align_tokens(self.future_depth_align_embs)
                if (
                    getattr(self, "use_future_video", False)
                    and getattr(self, "use_future_video_patch", True)
                    and getattr(self, "future_video_share_future_depth_query", False)
                    and getattr(self, "use_shared_future_task_proj", False)
                ):
                    future_video_task = _get_align_tokens(self.future_video_align_embs)
                    future_task = self.future_shared_task_proj(
                        torch.cat([future_task, future_video_task], dim=-1)
                    )
                future_align_embs = future_task.repeat(img_emb.size(0), 1, 1).to(
                    img_emb.device, img_emb.dtype
                )

            if (
                not self.use_future_depth
                and getattr(self, "use_future_video", False)
                and getattr(self, "future_video_share_future_depth_query", False)
            ):
                raise ValueError("share_future_depth_query=True requires depth.use_future_depth=True.")

            for segment_name in prefix_query_segments(
                use_depth_align=True,
                use_future_depth=self.use_future_depth,
                use_future_video=getattr(self, "use_future_video", False),
                use_future_video_cls=getattr(self, "use_future_video_cls", False),
                use_future_video_patch=getattr(self, "use_future_video_patch", True),
                future_video_share_future_depth_query=getattr(
                    self,
                    "future_video_share_future_depth_query",
                    False,
                ),
            ):
                if segment_name == "language":
                    _append(
                        lang_emb,
                        lang_masks,
                        lang_tokens.to(device),
                    )
                elif segment_name == "current_depth":
                    _append(align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_video_cls":
                    future_video_cls_align_emb = self.future_video_cls_align_emb.weight.repeat(
                        img_emb.size(0), 1, 1
                    ).to(img_emb.device, img_emb.dtype)
                    cls_align_pad_masks = torch.ones(
                        bsize,
                        1,
                        device=device,
                        dtype=lang_masks.dtype,
                    )
                    fake_cls_align_ids = torch.full(
                        (bsize, 1),
                        cfg.text_config.eos_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    _append(future_video_cls_align_emb, cls_align_pad_masks, fake_cls_align_ids)
                elif segment_name == "future_video":
                    future_video_align_embs = (
                        _get_align_tokens(self.future_video_align_embs)
                        .repeat(img_emb.size(0), 1, 1)
                        .to(img_emb.device, img_emb.dtype)
                    )
                    _append(future_video_align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_depth":
                    _append(future_align_embs, align_pad_masks, fake_align_ids)
                else:
                    raise ValueError(f"Unsupported prefix query segment: {segment_name}")

            embs = torch.cat(parts, dim=1)
            pad_masks = torch.cat(masks, dim=1)
            prefix_input_ids = torch.cat(input_ids, dim=1)
            full_visual_pos_masks = torch.cat(visual_masks, dim=1)
        else:
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
            prefix_input_ids = torch.cat([fake_image_ids, lang_tokens.to(device)], dim=1)
            full_visual_pos_masks = torch.cat([visual_pos_masks, torch.zeros_like(lang_masks)], dim=1)

        if getattr(self.config, "vlm_causal", False):
            att_masks = torch.ones((bsize, embs.shape[1]), device=device, dtype=torch.bool)
        else:
            att_masks = torch.zeros((bsize, embs.shape[1]), device=device, dtype=torch.bool)

        flat_img_masks = einops.rearrange(img_masks, "b n -> (b n)")
        rope_grid_thw = flat_grid_thw[flat_img_masks]
        if rope_grid_thw.numel() == 0:
            rope_grid_thw = flat_grid_thw[:1]
        prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            prefix_input_ids,
            pad_masks.long(),
            image_grid_thw=rope_grid_thw,
            video_grid_thw=None,
        )
        # Dense (capture-safe) deepstack layout: zero-masked per-camera embeds
        # laid out over the full prefix length instead of bool-filtered rows.
        # _apply_deepstack then does a plain shape-static add; the old filtered
        # form fed transformers' _deepstack_process, whose bool indexing
        # (nonzero -> device/host sync) breaks CUDA-graph capture and graph-
        # breaks torch.compile. x + 0.0 == x keeps the injection numerically
        # identical (only -0.0 can flip to +0.0), verified bitwise in
        # bench/deepstack_dense_parity.py.
        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        img_len = img_emb.shape[1]  # boundary-wrapped image part, [b, n*l, d]
        tail_len = embs.shape[1] - img_len
        dense_deepstack = []
        for deepstack in deepstack_embs:
            level = deepstack * img_visual_only.unsqueeze(-1).to(deepstack.dtype)
            if image_token_len != num_patch:  # vision boundaries: patches sit at [1 : 1+num_patch]
                wrapped = level.new_zeros(bsize, num_images, image_token_len, level.shape[-1])
                wrapped[:, :, 1 : 1 + num_patch] = level
                level = wrapped
            level = einops.rearrange(level, "b n l d -> b (n l) d")
            if tail_len > 0:  # language (and depth-align query) tail stays zero
                level = torch.cat([level, level.new_zeros(bsize, tail_len, level.shape[-1])], dim=1)
            dense_deepstack.append(level.to(embed_dtype))

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            dense_deepstack,
        )
        return result

    def _build_full_position_ids(self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks):
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        query_spans = prefix_query_token_spans(
            prefix_len=hidden_states.shape[1],
            num_task_tokens=self.num_task_tokens,
            use_depth_align=True,
            use_future_depth=getattr(self, "use_future_depth", False),
            use_future_video=getattr(self, "use_future_video", False),
            use_future_video_cls=getattr(self, "use_future_video_cls", False),
            use_future_video_patch=getattr(self, "use_future_video_patch", True),
            future_video_share_future_depth_query=getattr(
                self,
                "future_video_share_future_depth_query",
                False,
            ),
        )
        start, end = query_spans["current_depth"]
        return hidden_states[:, start:end, :]

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        loss_type="fm",
        depth_targets=None,
        image_grid_thw=None,
        future_depth_targets=None,
        future_video_targets=None,
        future_video_cls_targets=None,
        future_video_current_patch=None,
        collect_metrics=True,
    ) -> Tensor:
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)
        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        prefix_len = prefix_pad_masks.shape[1]
        if self.block_future_depth_to_action:
            att_2d_masks = block_suffix_to_fv_(
                att_2d_masks,
                suffix_row_start=prefix_len,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )

        att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
            att_2d_masks,
            suffix_row_start=prefix_len,
            prefix_len=prefix_len,
        )
        position_ids = self._build_full_position_ids(prefix_position_ids, prefix_pad_masks, suffix_pad_masks)

        (outputs_embeds, suffix_out), _, router_logits_list = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            # Training never reuses a KV cache — filling one here only wastes memory
            # (a full-sequence fp32/bf16 K/V copy per layer, discarded immediately).
            use_cache=False,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        align_metrics = {}
        if self.config.align_params != {}:
            loss_depth, loss_future_depth, depth_preds, future_depth_preds = self.depth_emb_forward(
                outputs_embeds,
                depth_targets,
                img_masks,
                future_depth_targets,
            )
            loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
            loss_future_depth = loss_future_depth * self.config.align_params.get(
                "future_depth_loss_weight", 1.0
            )
            loss_future_video = 0
            future_video_preds = None
            current_video_preds = None
            if getattr(self, "use_future_video", False):
                loss_video, future_video_preds, video_metrics = self.video_emb_forward(
                    outputs_embeds,
                    future_video_targets,
                    future_video_cls_targets=future_video_cls_targets,
                    future_video_current_patch=future_video_current_patch,
                )
                video_total_loss = loss_video
                if getattr(self, "use_current_video_patch", False) and future_video_current_patch is not None:
                    current_video_loss, current_video_preds, current_video_metrics = (
                        self.current_video_emb_forward(
                            outputs_embeds,
                            future_video_current_patch,
                        )
                    )
                    video_total_loss = video_total_loss + current_video_loss
                    video_metrics.update(current_video_metrics)
                    video_metrics["align/current_video_loss"] = current_video_loss.detach()
                video_cfg = self.config.align_params.get("video", {})
                video_weight = video_cfg.get(
                    "future_video_loss_weight",
                    self.config.align_params.get(
                        "future_video_loss_weight",
                        self.config.align_params["depth_loss_weight"],
                    ),
                )
                loss_future_video = video_total_loss * video_weight
                align_metrics.update(video_metrics)
                if "align/current_video_loss" in align_metrics:
                    align_metrics["align/current_video_loss_weighted"] = (
                        align_metrics["align/current_video_loss"] * video_weight
                    )
                align_metrics["align/future_video_loss"] = loss_video.detach()
                align_metrics["align/future_video_loss_weighted"] = (loss_video * video_weight).detach()
                align_metrics["align/video_loss"] = video_total_loss.detach()
                align_metrics["align/video_loss_weighted"] = loss_future_video.detach()
            self.steps += 1
        else:
            loss_depth = 0
            loss_future_depth = 0
            loss_future_video = 0
            depth_preds = None
            future_depth_preds = None
            future_video_preds = None
            current_video_preds = None

        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)

        if loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
        elif loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type!r} (expected 'fm' or 'L1_fm').")

        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(
            router_logits_list, losses, collect_metrics=collect_metrics
        )
        if align_metrics:
            moe_metrics.update(align_metrics)
        if os.environ.get("ALIGN_DEBUG"):
            def _s(name, t):
                if torch.is_tensor(t):
                    tf = t.float()
                    logging.get_logger(__name__).info(
                        f"[ALIGN_DEBUG] {name}: shape={tuple(t.shape)} "
                        f"nan={torch.isnan(tf).any().item()} absmax={tf.abs().max().item():.4f}"
                    )
            _s("outputs_embeds", outputs_embeds)
            _s("suffix_out", suffix_out)
            _s("v_t", v_t)
            _s("u_t", u_t)
            _s("losses(fm)", losses)
            _s("loss_depth", loss_depth)
            _s("loss_future_depth", loss_future_depth)
            _s("loss_future_video", loss_future_video)
            _s("future_video_targets", future_video_targets)
            _s("future_video_current_patch", future_video_current_patch)
            _s("depth_targets", depth_targets)
        return (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            seq_wise_loss,
            router_z_loss,
            moe_metrics,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        )

    def _embed_and_fill_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw):
        """Prefix half of sample_actions as one compilable unit: embed_prefix
        (vision tower + language embedding + mrope position ids) followed by the
        36-layer KV fill. Returns exactly what the denoise loop consumes."""
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return prefix_pad_masks, prefix_position_ids, past_key_values

    def _compile_with_mode(self, fn):
        """torch.compile with the shared mode. The default mode keeps
        triton.cudagraphs off via options (torch.compile forbids mode+options
        together; the *-no-cudagraphs modes already keep CUDA graphs off)."""
        mode = getattr(self, "_compile_predict_velocity_mode", "default")
        if mode == "default":
            return torch.compile(fn, fullgraph=False, dynamic=False, options={"triton.cudagraphs": False})
        return torch.compile(fn, fullgraph=False, dynamic=False, mode=mode)

    def _get_rtc_processor(self):
        """Lazily build the RTC guidance processor.

        Defaults come from ``RTCConfig``; policy-config fields named ``rtc_<field>``
        (e.g. ``rtc_max_guidance_weight``) override them for deployment tuning.
        """
        proc = getattr(self, "_rtc_processor", None)
        if proc is None:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor

            cfg = RTCConfig()
            for field_name in ("max_guidance_weight", "execution_horizon"):
                override = getattr(self.config, f"rtc_{field_name}", None)
                if override is not None:
                    setattr(cfg, field_name, override)
            proc = RTCProcessor(cfg)
            self._rtc_processor = proc
        return proc

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """Do a full Qwen3-VL inference forward and compute the action."""
        if not getattr(self.config, "use_cache", True):
            raise ValueError(
                "sample_actions requires config.use_cache=True: the denoise loop reuses "
                "the prefix KV cache, and with use_cache=False the prefix fill returns "
                "past_key_values=None. (Training forward does not go through "
                "sample_actions and is unaffected.)"
            )
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        if getattr(self, "_use_prefix_graph", False):
            # CUDA-graphed prefix (see config docs). Falls through to the
            # compiled/eager paths below only for the non-CUDA / use_cache
            # guards; capture failures finish eagerly from the already-run
            # embed_prefix (no second vision pass).
            got = self._prefix_graphed(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
            if got is not None:
                prefix_pad_masks, prefix_position_ids, past_key_values = got
            else:
                prefix_pad_masks, prefix_position_ids, past_key_values = self._embed_and_fill_prefix(
                    images, img_masks, lang_tokens, lang_masks, image_grid_thw
                )
        elif getattr(self, "_use_compile_prefix", False):
            prefix_fn = getattr(self, "_compiled_prefix", None)
            if prefix_fn is None:
                prefix_fn = self._compile_with_mode(self._embed_and_fill_prefix)
                self._compiled_prefix = prefix_fn
            prefix_pad_masks, prefix_position_ids, past_key_values = prefix_fn(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                image_grid_thw,
            )
        else:
            prefix_pad_masks, prefix_position_ids, past_key_values = self._embed_and_fill_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                image_grid_thw,
            )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        # Precompute the timestep schedule without any host read-back: a
        # `while time >= -dt / 2` condition forces a GPU->CPU sync every
        # denoise step, draining the pipeline and exposing host launch
        # overhead. The values below come from the same iterative `time + dt`
        # accumulation, so the schedule is bit-identical to the while form
        # (exactly num_steps entries; accumulation error stays far below the
        # old -dt/2 threshold).
        time = torch.tensor(1.0, dtype=dtype, device=device)
        time_values = []
        for _ in range(self.config.num_steps):
            time_values.append(time)
            time = time + dt
        count = 0
        predict_velocity_fn = self.predict_velocity
        if getattr(self, "_use_compile_predict_velocity", False):
            predict_velocity_fn = getattr(self, "_compiled_predict_velocity", None)
            if predict_velocity_fn is None:
                predict_velocity_fn = self._compile_with_mode(self.predict_velocity)
                self._compiled_predict_velocity = predict_velocity_fn

        guided = prev_chunk_left_over is not None
        if guided and any(p.requires_grad for p in self.parameters()):
            # Guided (RTC) steps run under grad mode: RTCProcessor.denoise_step needs
            # autograd from x_t to the denoiser output. With trainable params the graph
            # would additionally carry a dead full-network backward per step whose saved
            # activations OOM smaller GPUs (measured on a 24GB 4090). Inference never
            # updates weights, so freeze once on first guided use.
            for p in self.parameters():
                p.requires_grad_(False)

        if getattr(self.config, "use_cudagraph_denoise", False) and not guided:
            # The captured graph has no guidance hook; replay it only unguided.
            graphed = self._denoise_loop_graphed(
                predict_velocity_fn,
                state,
                prefix_pad_masks,
                past_key_values,
                noise,
                prefix_position_ids,
                time_values,
                dt,
            )
            if graphed is not None:
                logger.debug("Denoised %s steps (single CUDA graph replay)", len(time_values))
                return graphed
            # Shape change or capture failure — fall through to the plain loop.

        # Loop-invariant tensors (suffix 2D masks / position ids / mrope cos-sin /
        # flex BlockMask) are computed on the first predict_velocity call and reused
        # for the remaining denoise steps — they depend on the prefix masks only,
        # not on x_t or the timestep.
        denoise_cache: dict = {}
        for step_time in time_values:
            count += 1
            expanded_time = step_time.expand(bsize)

            if guided:
                # RTC guidance (bench/rtc_bench.py parity): per-step local denoise cache —
                # a shared cache would alias step k's tensors into step k+1's autograd graph.
                def _pv_step(input_x_t, _et=expanded_time):
                    return predict_velocity_fn(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        input_x_t,
                        _et,
                        prefix_position_ids=prefix_position_ids,
                        _denoise_cache=None,
                    )

                with torch.enable_grad():
                    v_t = self._get_rtc_processor().denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=step_time,
                        original_denoise_step_partial=_pv_step,
                        execution_horizon=int(prev_chunk_left_over.shape[0]),
                    )
                v_t = v_t.to(x_t.dtype)  # guidance math upcasts to f32; keep x_t dtype
            else:
                v_t = predict_velocity_fn(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                    prefix_position_ids=prefix_position_ids,
                    _denoise_cache=denoise_cache,
                )

            x_t += dt * v_t
        logger.debug("Denoised %s steps%s", count, " (RTC guided)" if guided else "")
        return x_t

    @torch.no_grad()
    def _denoise_loop_graphed(
        self,
        predict_velocity_fn,
        state,
        prefix_pad_masks,
        past_key_values,
        noise,
        prefix_position_ids,
        time_values,
        dt,
    ):
        """Run the denoise loop as a single captured CUDA graph.

        Returns the denoised action chunk, or None when the graph is
        unavailable — non-CUDA input, `use_cache=False`, or a warm-up/capture
        failure — so the caller falls back to the plain loop. An
        observation-shape change drops the stale graph and re-captures.
        """
        if not state.is_cuda:
            return None
        if getattr(self, "_denoise_graph_disabled", False):
            return None
        if past_key_values is None:
            # use_cache=False: there is no KV cache to freeze into the graph.
            return None
        kv_items = tuple(sorted(past_key_values.items()))
        sig = (
            tuple(noise.shape),
            noise.dtype,
            str(noise.device),
            tuple(state.shape),
            state.dtype,
            tuple(prefix_pad_masks.shape),
            tuple(prefix_position_ids.shape),
            tuple(
                (idx, tuple(kv["key_states"].shape), tuple(kv["value_states"].shape)) for idx, kv in kv_items
            ),
            len(time_values),
            # Alias guard: when the static KV are the prefix graph's pool
            # outputs, their generation must match too — any prefix-graph
            # transition (re-capture/drop/disable) changes this term and
            # forces a re-capture here instead of a stale-alias replay.
            self._prefix_kv_gen(past_key_values),
        )
        gs = getattr(self, "_denoise_graph_state", None)
        if gs is not None and gs["sig"] != sig:
            warned = getattr(self, "_denoise_graph_warned", None)
            if warned is None:
                warned = self._denoise_graph_warned = set()
            if sig not in warned:
                logger.warning(
                    "use_cudagraph_denoise: observation shapes or prefix-KV alias changed; "
                    "re-capturing the denoise graph"
                )
                warned.add(sig)
            gs = None  # drop the stale graph (frees its private pool) and re-capture below
        if gs is None:
            gs = self._capture_denoise_graph(
                predict_velocity_fn,
                state,
                prefix_pad_masks,
                past_key_values,
                noise,
                prefix_position_ids,
                time_values,
                dt,
                sig,
            )
            if gs is None:
                return None
            self._denoise_graph_state = gs

        # Replay: copy the live prefix outputs into the static buffers the
        # graph reads, then re-execute the recorded kernel sequence. One
        # _foreach_copy_ for the whole set (76 tensors for a 36-layer prefix):
        # per-tensor copy_ launches would cost ~7ms of host time per chunk.
        # The aliased case skips the KV copies: the signature above matched
        # the identity-derived generation, so these KV *are* the prefix
        # graph's pool outputs the graph was captured reading.
        dsts = [gs["state"], gs["prefix_pad_masks"], gs["prefix_position_ids"], gs["x_t"]]
        srcs = [state, prefix_pad_masks, prefix_position_ids, noise]
        if not gs.get("kv_aliased", False):
            for idx, kv in kv_items:
                dsts.append(gs["kv"][idx]["key_states"])
                srcs.append(kv["key_states"])
                dsts.append(gs["kv"][idx]["value_states"])
                srcs.append(kv["value_states"])
        torch._foreach_copy_(dsts, srcs)
        gs["graph"].replay()
        return gs["out"].clone()

    @torch.no_grad()
    def _capture_denoise_graph(
        self,
        predict_velocity_fn,
        state,
        prefix_pad_masks,
        past_key_values,
        noise,
        prefix_position_ids,
        time_values,
        dt,
        sig,
    ):
        # When this chunk's KV are exactly the prefix graph's pool outputs,
        # alias them as the static buffers instead of cloning: the denoise
        # graph then reads the storage the prefix graph replays into, and the
        # per-chunk KV copy below disappears. The signature's generation term
        # (see _denoise_loop_graphed) keeps the alias valid.
        kv_aliased = self._prefix_kv_gen(past_key_values) is not None
        static = {
            "state": state.clone(),
            "prefix_pad_masks": prefix_pad_masks.clone(),
            "prefix_position_ids": prefix_position_ids.clone(),
            "kv": {
                idx: {
                    "key_states": kv["key_states"] if kv_aliased else kv["key_states"].clone(),
                    "value_states": kv["value_states"] if kv_aliased else kv["value_states"].clone(),
                }
                for idx, kv in past_key_values.items()
            },
            "x_t": noise.clone(),
            "dt": dt.clone(),
            "time_values": [t.clone() for t in time_values],
        }
        bsize = state.shape[0]

        def run_loop():
            # A fresh cache per call: step 1 takes the fill branch, later steps
            # the cached branch — exactly matching the plain loop. Capturing
            # with an already-populated cache would record the all-cached graph,
            # a different compiled artifact whose bf16 fusion differences
            # integrate over the denoise steps (measured as visible drift).
            cache: dict = {}
            x = static["x_t"]
            for step_time in static["time_values"]:
                v_t = predict_velocity_fn(
                    static["state"],
                    static["prefix_pad_masks"],
                    static["kv"],
                    x,
                    step_time.expand(bsize),
                    prefix_position_ids=static["prefix_position_ids"],
                    _denoise_cache=cache,
                )
                x = x + static["dt"] * v_t
            return x

        # Warm on the default stream so both compiled branches (fill +
        # cached) exist, then once on a side stream so the allocator sees
        # the loop's allocations outside the graph pool. A warm-up failure is
        # not a capture failure: disable the graph and let the plain loop
        # surface the real error.
        try:
            run_loop()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_loop()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_denoise: warm-up pass failed (%s: %s); using the plain denoise loop",
                type(exc).__name__,
                exc,
            )
            self._denoise_graph_disabled = True  # don't retry on every call
            return None

        try:
            graph = torch.cuda.CUDAGraph()
            # A recompile mid-capture would enqueue autotuning work on the
            # capture stream; refuse it instead.
            with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
                static_out = run_loop()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_denoise: capture failed (%s: %s); using the plain denoise loop",
                type(exc).__name__,
                exc,
            )
            self._denoise_graph_disabled = True  # don't retry on every call
            return None

        logger.info(
            "use_cudagraph_denoise: captured the %s-step denoise loop as one CUDA graph%s",
            len(time_values),
            " (prefix-KV aliased)" if kv_aliased else "",
        )
        return {"sig": sig, "graph": graph, "out": static_out, "kv_aliased": kv_aliased, **static}

    def _prefix_llm_forward(
        self,
        prefix_embs,
        prefix_att_2d_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """The capture-scope unit: the 36-layer KV fill only (no vision tower,
        no embed glue — their host syncs forbid capture). Sync-free since the
        dense-deepstack refactor; compilable and CUDA-graph capturable."""
        _, past_kv, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return past_kv

    def _prefix_kv_gen(self, past_key_values):
        """Generation id of the prefix graph whose pool outputs are exactly
        these KV tensors, else ``None`` (fresh eager tensors).

        This identity check is the alias-safety guard: any prefix-graph state
        transition — re-capture (new pool tensors, new gen), drop, or disable
        with an eager fallback — yields tensors that fail the ``is`` test, so
        the caller's shape signature changes and the stale denoise graph is
        re-captured or re-copied instead of replaying against old KV.
        """
        pgs = getattr(self, "_prefix_graph_state", None)
        if pgs is None:
            return None
        try:
            if all(
                pgs["out"][idx][part] is kv[part]
                for idx, kv in past_key_values.items()
                for part in ("key_states", "value_states")
            ):
                return pgs["gen"]
        except KeyError:
            pass
        return None

    @torch.no_grad()
    def _vision_tower_graphed(self, images, flat_grid_thw):
        """Run the vision tower (``embed_image``) as a captured CUDA graph.

        The grid-derived metadata (pos_embeds / cu_seqlens / split_sizes / max_seqlen)
        is hoisted once into ``core``'s capture cache, so the per-call host syncs in
        ``preprcess_grid_thw`` / ``get_image_features`` become replay-time constants and
        the tower is capture-safe. Returns ``(img_emb, deepstack_embs)`` — the graph
        pool's outputs — or ``None`` on failure (caller falls back to eager).
        """
        if not images.is_cuda or getattr(self, "_vision_graph_disabled", False):
            return None
        sig = (tuple(images.shape), images.dtype, str(images.device), tuple(flat_grid_thw.shape))
        gs = getattr(self, "_vision_graph_state", None)
        if gs is not None and gs["sig"] != sig:
            self._vision_graph_gen = getattr(self, "_vision_graph_gen", 0) + 1
            gs = None
        if gs is None:
            if getattr(self, "_vision_recaptures", 0) >= 4:
                self._vision_graph_disabled = True
                logger.warning("use_cudagraph_prefix_full: vision re-capture limit reached; eager ViT")
                return None
            gs = self._capture_vision_graph(images, flat_grid_thw, sig)
            if gs is None:
                return None
            self._vision_recaptures = getattr(self, "_vision_recaptures", 0) + 1
            self._vision_graph_state = gs
        gs["images"].copy_(images)
        gs["graph"].replay()
        return gs["img_emb"], gs["deepstack"]

    @torch.no_grad()
    def _capture_vision_graph(self, images, flat_grid_thw, sig):
        core = self.qwenvl_with_expert
        prev_flag = core._capture_grid_cache
        prev_precompute = getattr(core.config, "precompute_grid_thw", False)
        prev_grid = (core.pos_embeds, core.position_embeddings, core.cu_seqlens,
                     core.visual_split_sizes, core.visual_max_seqlen)
        static_in = images.clone()
        try:
            # Arm + seed the capture grid cache: populate pos_embeds/cu_seqlens/
            # split_sizes/max_seqlen once so the vision tower is sync-free. The cache
            # stays live on `core` afterwards — the captured graph closes over it.
            core._capture_grid_cache = True
            core.config.precompute_grid_thw = True
            core.pos_embeds = None
            core.position_embeddings = None
            core.cu_seqlens = None
            core.visual_split_sizes = None
            core.visual_max_seqlen = None

            def run_vit():
                return core.embed_image(static_in, flat_grid_thw)

            run_vit()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_vit()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix_full: vision warm-up failed (%s: %s); eager ViT",
                type(exc).__name__, exc,
            )
            core._capture_grid_cache = prev_flag
            core.config.precompute_grid_thw = prev_precompute
            (core.pos_embeds, core.position_embeddings, core.cu_seqlens,
             core.visual_split_sizes, core.visual_max_seqlen) = prev_grid
            self._vision_graph_disabled = True
            return None
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
                img_emb, deepstack = run_vit()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix_full: vision capture failed (%s: %s); eager ViT",
                type(exc).__name__, exc,
            )
            core._capture_grid_cache = prev_flag
            core.config.precompute_grid_thw = prev_precompute
            (core.pos_embeds, core.position_embeddings, core.cu_seqlens,
             core.visual_split_sizes, core.visual_max_seqlen) = prev_grid
            self._vision_graph_disabled = True
            return None
        finally:
            core._capture_grid_cache = prev_flag
            core.config.precompute_grid_thw = prev_precompute
            # grid cache stays live for replay.
        logger.info("use_cudagraph_prefix_full: captured the vision tower as one CUDA graph")
        return {"sig": sig, "graph": graph, "images": static_in, "img_emb": img_emb, "deepstack": deepstack}

    @torch.no_grad()
    def _prefix_graphed(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw,
    ):
        """Run the prefix 36-layer KV fill as one captured CUDA graph.

        Returns ``(prefix_pad_masks, prefix_position_ids, past_key_values)``
        — the same contract as :meth:`_embed_and_fill_prefix` — or ``None``
        when the graph path is unavailable (non-CUDA input, ``use_cache``
        off, or disabled after earlier failures) so the caller falls back.
        The vision tower and embed glue stay eager (their host syncs forbid
        capture); only ``qwenvl_with_expert.forward(..., fill_kv_cache=True)``
        is captured. Capture failures finish eagerly from the already-computed
        embeds (no second vision pass). The returned KV are the graph pool's
        output tensors; the denoise graph aliases them (see ``_prefix_kv_gen``).
        """
        if not images.is_cuda or getattr(self, "_prefix_graph_disabled", False):
            return None
        if not getattr(self.config, "use_cache", True):
            return None

        if getattr(self.config, "use_cudagraph_prefix_full", False):
            # Vision tower as its own graph (grid metadata cached on `core`); the embed
            # glue (language embed / masks / mrope ids / dense deepstack) stays eager —
            # it is light and get_rope_index's data-dependent host syncs cannot be
            # captured. The 36-layer KV fill graph below is unchanged.
            flat_grid_thw = (
                einops.rearrange(image_grid_thw, "b n d -> (b n) d")
                if image_grid_thw.ndim == 3
                else image_grid_thw
            )
            vision_outputs = self._vision_tower_graphed(images, flat_grid_thw)
            if vision_outputs is None:
                return None  # vision graph unavailable — caller falls back to eager prefix
        else:
            vision_outputs = None

        # Capture target: the thin 36-layer fill, optionally torch.compile'd
        # (mirroring how the denoise graph captures the compiled
        # predict_velocity). Post-dense-deepstack this region is sync-free,
        # so it compiles and captures; the compiled kernels keep their
        # inductor fusion inside the graph.
        prefix_llm_fn = self._prefix_llm_forward
        if getattr(self, "_use_compile_prefix", False):
            prefix_llm_fn = getattr(self, "_compiled_prefix_llm", None)
            if prefix_llm_fn is None:
                prefix_llm_fn = self._compile_with_mode(self._prefix_llm_forward)
                self._compiled_prefix_llm = prefix_llm_fn

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
            vision_outputs=vision_outputs,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        def _eager_finish():
            # ``_prefix_llm_forward`` returns the KV cache alone (the 3-tuple
            # unpack belongs to the ``qwenvl_with_expert.forward`` call inside it).
            past_kv = self._prefix_llm_forward(
                prefix_embs,
                prefix_att_2d_masks,
                prefix_position_ids,
                visual_pos_masks,
                deepstack_visual_embeds,
            )
            return prefix_pad_masks, prefix_position_ids, past_kv

        sig = (
            tuple(prefix_embs.shape),
            prefix_embs.dtype,
            str(prefix_embs.device),
            tuple(prefix_att_2d_masks.shape),
            prefix_att_2d_masks.dtype,
            tuple(prefix_position_ids.shape),
            prefix_position_ids.dtype,
            tuple(visual_pos_masks.shape),
            visual_pos_masks.dtype,
            tuple(tuple(d.shape) for d in deepstack_visual_embeds),
            tuple(d.dtype for d in deepstack_visual_embeds),
        )
        gs = getattr(self, "_prefix_graph_state", None)
        if gs is not None and gs["sig"] != sig:
            warned = getattr(self, "_prefix_graph_warned", None)
            if warned is None:
                warned = self._prefix_graph_warned = set()
            if sig not in warned:
                logger.warning(
                    "use_cudagraph_prefix: prefix shapes changed; re-capturing the prefix graph"
                )
                warned.add(sig)
            # Drop the stale graph. The aliased denoise graph still holds
            # references to the old pool tensors until its own signature
            # check (which includes the generation below) drops them.
            self._prefix_graph_gen = getattr(self, "_prefix_graph_gen", 0) + 1
            gs = None
        if gs is None:
            if getattr(self, "_prefix_recaptures", 0) >= 4:
                # Circuit breaker: shape flicker must not re-capture forever.
                self._prefix_graph_disabled = True
                logger.warning(
                    "use_cudagraph_prefix: re-capture limit reached; using the eager prefix"
                )
                return _eager_finish()
            gs = self._capture_prefix_graph(
                prefix_llm_fn,
                prefix_embs,
                prefix_att_2d_masks,
                prefix_position_ids,
                visual_pos_masks,
                deepstack_visual_embeds,
                sig,
            )
            if gs is None:
                # Warm-up/capture failure: already warned and disabled.
                return _eager_finish()
            self._prefix_recaptures = getattr(self, "_prefix_recaptures", 0) + 1
            self._prefix_graph_state = gs

        # Replay: copy the live values into the static inputs, then replay.
        # Always executed — including right after capture, whose pool outputs
        # are uninitialized until the first replay.
        dsts = [
            gs["prefix_embs"],
            gs["att_2d"],
            gs["position_ids"],
            gs["visual_pos_masks"],
            *gs["deepstack"],
        ]
        srcs = [prefix_embs, prefix_att_2d_masks, prefix_position_ids, visual_pos_masks, *deepstack_visual_embeds]
        torch._foreach_copy_(dsts, srcs)
        gs["graph"].replay()
        return prefix_pad_masks, prefix_position_ids, gs["out"]

    @torch.no_grad()
    def _capture_prefix_graph(
        self,
        prefix_llm_fn,
        prefix_embs,
        prefix_att_2d_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        sig,
    ):
        static = {
            "prefix_embs": prefix_embs.clone(),
            "att_2d": prefix_att_2d_masks.clone(),
            "position_ids": prefix_position_ids.clone(),
            "visual_pos_masks": visual_pos_masks.clone(),
            "deepstack": [d.clone() for d in deepstack_visual_embeds],
        }

        def run_prefix():
            return prefix_llm_fn(
                static["prefix_embs"],
                static["att_2d"],
                static["position_ids"],
                static["visual_pos_masks"],
                static["deepstack"],
            )

        # Warm-up discipline mirrors _capture_denoise_graph: default stream,
        # side stream, synchronize — so the allocator sees the allocations
        # outside the graph pool and any failure surfaces as a fallback, not
        # a broken capture.
        try:
            run_prefix()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_prefix()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix: warm-up pass failed (%s: %s); using the eager prefix",
                type(exc).__name__,
                exc,
            )
            self._prefix_graph_disabled = True  # don't retry on every call
            return None

        try:
            graph = torch.cuda.CUDAGraph()
            with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
                static_out = run_prefix()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix: capture failed (%s: %s); using the eager prefix",
                type(exc).__name__,
                exc,
            )
            self._prefix_graph_disabled = True  # don't retry on every call
            return None

        self._prefix_graph_gen = getattr(self, "_prefix_graph_gen", 0) + 1
        logger.info(
            "use_cudagraph_prefix: captured the %s-layer prefix KV fill as one CUDA graph",
            len(static_out),
        )
        return {"sig": sig, "gen": self._prefix_graph_gen, "graph": graph, "out": static_out, **static}

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
        _denoise_cache: dict | None = None,
    ):
        """Predict velocity at time t using cached Qwen3-VL prefix states.

        ``_denoise_cache`` (optional) is a dict that persists across the denoise
        loop: the suffix attention mask, position ids, mrope cos/sin and flex
        BlockMask are loop-invariant, so they are computed on the first step and
        reused afterwards.
        """
        if prefix_position_ids is None:
            raise ValueError("FlowMatchingV2.predict_velocity requires Qwen3-VL prefix_position_ids.")

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        cache = _denoise_cache if _denoise_cache is not None else {}
        if "full_att_2d_masks" not in cache:
            batch_size = prefix_pad_masks.shape[0]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size,
                suffix_len,
                prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            if self.block_future_depth_to_action:
                # Query rows here are all suffix (state/action), so row start is 0.
                full_att_2d_masks = block_suffix_to_fv_(
                    full_att_2d_masks,
                    suffix_row_start=0,
                    prefix_len=prefix_len,
                    num_task_tokens=self.num_task_tokens,
                )
            full_att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
                full_att_2d_masks,
                suffix_row_start=0,
                prefix_len=prefix_len,
            )

            full_position_ids = self._build_full_position_ids(
                prefix_position_ids,
                prefix_pad_masks,
                suffix_pad_masks,
            )
            position_ids = full_position_ids[:, :, -suffix_len:]
            core = self.qwenvl_with_expert
            rep = suffix_embs.float() if getattr(core.config, "attention_fp32", False) else suffix_embs
            position_embeddings = core.qwenvl.model.language_model.rotary_emb(rep, position_ids)
            cache["full_att_2d_masks"] = full_att_2d_masks
            cache["position_ids"] = position_ids
            cache["position_embeddings"] = position_embeddings
            if core.config.attention_implementation == "flex_cached":
                cache["block_mask"] = build_block_mask(
                    full_att_2d_masks,
                    core.qwenvl.config.text_config.num_attention_heads,
                    suffix_len,
                    prefix_len + suffix_len,
                )

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=cache["full_att_2d_masks"],
            position_ids=cache["position_ids"],
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
            position_embeddings=cache["position_embeddings"],
            block_mask=cache.get("block_mask"),
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        return v_t

    def _moe_losses_and_metrics(self, router_logits_list, losses, collect_metrics=True):
        router_z_loss_coeff = getattr(self.config, "router_z_loss_coeff", 0)
        router_z_loss = losses.new_zeros(())
        router_z_layer_losses = None  # per-layer raw z-loss (pre-coeff), for monitoring
        if router_z_loss_coeff > 0 and router_logits_list:
            router_z_layer_losses = [
                torch.logsumexp(logits.float(), dim=-1).pow(2).mean() for logits in router_logits_list
            ]
            router_z_loss = router_z_loss_coeff * torch.stack(router_z_layer_losses).mean()

        seq_wise_loss_coeff = getattr(self.config, "sequence_wise_loss_coeff", 0)
        seq_wise_loss = 0
        seqwise_layer_losses = None  # per-layer raw seq-wise balance loss (pre-coeff), for monitoring
        if seq_wise_loss_coeff > 0 and router_logits_list:
            # router_logits are [B*T, E] (action-expert tokens, fixed length T per sample).
            # per_sequence -> balance experts within each sample's T tokens (DeepSeek-V3 intent);
            # global -> treat the whole B*T batch as one sequence.
            mode = getattr(self.config, "sequence_wise_mode", "per_sequence")
            score_func = getattr(self.config, "router_activation", "softmax")
            if mode == "global":
                seq_lengths = None
            else:
                batch = losses.shape[0]
                n_tokens = router_logits_list[0].shape[0]
                seq_lengths = [n_tokens // batch] * batch
            seqwise_moe_layer_ids = sorted(getattr(self.config, "token_moe_layers", None) or [])
            # Per-layer e_score_correction_bias so the loss's f_i top-k matches the
            # router's actual (bias-corrected) selection.
            seqwise_router_biases = tuple(
                getattr(
                    self.qwenvl_with_expert.qwen_expert.model.layers[
                        seqwise_moe_layer_ids[i] if i < len(seqwise_moe_layer_ids) else i
                    ].mlp,
                    "e_score_correction_bias",
                    None,
                )
                for i in range(len(router_logits_list))
            )
            seqwise_layer_losses = triton_sequence_wise_balance_loss(
                router_logits_list=tuple(router_logits_list),
                top_k=getattr(self.config, "token_top_k", 4),
                seq_lengths=seq_lengths,
                padding_len=0,
                score_func=score_func,
                e_score_correction_bias_list=seqwise_router_biases,
            )
            if seqwise_layer_losses:
                seq_wise_loss = seq_wise_loss_coeff * torch.stack(seqwise_layer_losses).mean()

        moe_metrics = {}
        # Monitoring-only block (per-layer MaxVio/entropy/dead-expert stats + the
        # per-metric .item() syncs in the caller). Gated by collect_metrics so it
        # runs on logging steps only, not every training step.
        if collect_metrics and router_logits_list:
            token_moe_layers_list = sorted(getattr(self.config, "token_moe_layers", None) or [])
            all_moe_indices = token_moe_layers_list
            token_expert_counts = []
            # Per-layer token-MoE stats, collected for moe_summary/* cross-layer aggregates.
            tok_maxvio, tok_minvio, tok_minload, tok_entropy, tok_sigmoid = [], [], [], [], []
            tok_bias = []  # per-layer max(|e_score_correction_bias|) (loss-free); >1 -> bias dominates sigmoid score
            any_dead = None  # OR-accumulated bool: any token-MoE layer with a 0-count expert
            with torch.no_grad():
                for i, logits in enumerate(router_logits_list):
                    layer_id = all_moe_indices[i] if i < len(all_moe_indices) else i
                    num_experts = logits.shape[-1]
                    routing_probs = F.softmax(logits, dim=1, dtype=torch.float)
                    moe_block = self.qwenvl_with_expert.qwen_expert.model.layers[layer_id].mlp
                    _, selected = torch.topk(routing_probs, 1, dim=-1)
                    counts = F.one_hot(selected.squeeze(-1), num_classes=num_experts).float().sum(dim=0)
                    avg_load = counts.mean()
                    denom = avg_load.clamp(min=1e-9)
                    maxvio = (counts.max() - avg_load) / denom  # peak overload  (>=0, larger=worse)
                    minvio = (avg_load - counts.min()) / denom  # valley underload (=1 -> dead expert)
                    min_load_ratio = counts.min() / denom  # =0 -> dead expert
                    # entropy is rank-local (this rank's routing_probs, last micro-batch).
                    per_sample_entropy = -(routing_probs * routing_probs.clamp(min=1e-9).log()).sum(dim=-1)
                    entropy = per_sample_entropy.mean()
                    ll = f"{layer_id:02d}"
                    token_expert_counts.append((layer_id, counts))
                    moe_metrics[f"moe_maxvio/layer{ll}"] = maxvio
                    moe_metrics[f"moe_minvio/layer{ll}"] = minvio
                    moe_metrics[f"moe_minload/layer{ll}"] = min_load_ratio
                    moe_metrics[f"moe_entropy_rank0/layer{ll}"] = entropy
                    tok_maxvio.append(maxvio)
                    tok_minvio.append(minvio)
                    tok_minload.append(min_load_ratio)
                    tok_entropy.append(entropy)
                    dead = counts.min() == 0
                    any_dead = dead if any_dead is None else (any_dead | dead)
                    if hasattr(moe_block, "avg_topk_sigmoid_score"):
                        sig = moe_block.avg_topk_sigmoid_score.detach().reshape(()).to(denom)
                        moe_metrics[f"moe_topksigmoid_rank0/layer{ll}"] = sig
                        tok_sigmoid.append(sig)
                    if hasattr(moe_block, "e_score_correction_bias"):
                        bias_absmax = moe_block.e_score_correction_bias.detach().abs().max().to(denom)
                        moe_metrics[f"moe_bias/layer{ll}"] = bias_absmax
                        tok_bias.append(bias_absmax)
                # ---- moe_summary/* : cross-layer aggregates over token-MoE layers (written every step) ----
                if tok_maxvio:
                    moe_metrics["moe_summary/maxvio_avg"] = torch.stack(tok_maxvio).mean()
                    moe_metrics["moe_summary/maxvio_max"] = torch.stack(tok_maxvio).max()
                    moe_metrics["moe_summary/minvio_avg"] = torch.stack(tok_minvio).mean()
                    moe_metrics["moe_summary/minvio_max"] = torch.stack(tok_minvio).max()
                    moe_metrics["moe_summary/min_load_ratio"] = torch.stack(tok_minload).min()
                    moe_metrics["moe_summary/has_dead_expert"] = any_dead.float()
                    moe_metrics["moe_summary/entropy_avg_rank0"] = torch.stack(tok_entropy).mean()
                if tok_sigmoid:
                    moe_metrics["moe_summary/topk_sigmoid_avg_rank0"] = torch.stack(tok_sigmoid).mean()
                if tok_bias:
                    moe_metrics["moe_summary/bias_absmax"] = torch.stack(tok_bias).max()
                # ---- moe_seqwise/* : per-layer raw sequence-wise balance loss (pre-coeff) + average ----
                if seqwise_layer_losses and len(seqwise_layer_losses) == len(all_moe_indices):
                    sw_vals = []
                    for lid, sw in zip(all_moe_indices, seqwise_layer_losses, strict=True):
                        v = sw.detach()
                        moe_metrics[f"moe_seqwise/layer{lid:02d}"] = v
                        sw_vals.append(v)
                    moe_metrics["moe_seqwise/avg"] = torch.stack(sw_vals).mean()
                # ---- moe_zloss/* : per-layer raw router z-loss (pre-coeff) + average/weighted loss ----
                if router_z_layer_losses and len(router_z_layer_losses) == len(all_moe_indices):
                    zl_vals = []
                    for lid, zl in zip(all_moe_indices, router_z_layer_losses, strict=True):
                        v = zl.detach()
                        moe_metrics[f"moe_zloss/layer{lid:02d}"] = v
                        zl_vals.append(v)
                    moe_metrics["moe_zloss/avg_raw"] = torch.stack(zl_vals).mean()
                    moe_metrics["moe_zloss/weighted"] = router_z_loss.detach()
                if token_expert_counts:
                    moe_metrics["_token_moe_expert_counts"] = token_expert_counts
        return seq_wise_loss, router_z_loss, moe_metrics


# ============================================================================
# LeRobot policy wrapper
# ============================================================================
# The classes above are vendored/adapted from the upstream LingBot-VLA 2.0 repo
# (Robbyant/lingbot-vla-v2). The wrapper below exposes them through LeRobot's
# ``PreTrainedPolicy`` interface (train ``forward`` + rolling ``select_action``),
# mirroring the v1 ``lingbot_vla`` policy. The LeRobot dataclass config carries
# every field ``FlowMatchingV2`` reads, so it is passed straight through.


class LingbotVLAV2Policy(PreTrainedPolicy):
    """LingBot-VLA 2.0 policy for cross-embodiment robotic control.

    Couples a Qwen3-VL-4B vision-language backbone with a sparse-MoE action
    expert (pi0-style dual-stream) and predicts action chunks via flow matching.
    Native-resolution image tokens are described by ``image_grid_thw``.

    The model expects already model-ready tensors in the batch (produced by the
    lingbot_vla_v2 processor / feature transform):
        - ``images``: patchified pixels for Qwen3-VL
        - ``img_masks``: per-view validity mask
        - ``lang_tokens`` / ``lang_masks``: tokenized instruction + mask
        - ``image_grid_thw``: (num_images, 3) temporal/height/width patch grid
        - ``observation.state``: (B, max_state_dim) padded state
        - ``action``: (B, chunk_size, max_action_dim) padded action (training)
        - optional ``joint_mask``: (B, chunk_size, max_action_dim) valid-slot mask
    """

    config_class = LeRobotLingbotVLAV2Config
    name = "lingbot_vla_v2"
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock", "Qwen2DecoderLayer"]

    def __init__(self, config: LeRobotLingbotVLAV2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = FlowMatchingV2(config, eval=False)

        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
            # With lm_head removed, the backbone's final decoder layer output and the
            # final RMSNorm feed no loss: the prefix representation consumed downstream
            # is taken before them, so their weights get requires_grad=True yet a None
            # gradient every step. Under DDP such params make the reducer emit NaN into
            # the shared gradient bucket and corrupt the whole graph; freeze them.
            # They still run in forward (inference sample_actions fills the KV cache
            # through them) — only their gradient update is disabled.
            tail = self.model.qwenvl_with_expert.qwenvl.model.language_model
            last = tail.layers[-1]
            for mod in (last.self_attn.o_proj, last.mlp, last.post_attention_layernorm, tail.norm):
                for p in mod.parameters():
                    p.requires_grad_(False)
        del self.model.qwenvl_with_expert.qwen_expert.lm_head

        # The Qwen3-VL backbone builds in bfloat16 while our added projection/AdaRMSNorm
        # heads build in float32. Cast the whole model to one dtype so the dual streams
        # stay consistent (mixed dtypes raise "mat1 and mat2 must have the same dtype").
        model_dtype = getattr(torch, getattr(self.config, "dtype", "bfloat16"))
        if isinstance(model_dtype, torch.dtype) and model_dtype.is_floating_point:
            self.model.to(model_dtype)

        # Inference-time action de-normalizer: an unapply-only FeatureTransform (built
        # without the image processor / tokenizer) that inverts the per-slot normalization
        # and the canonical slot mapping on the model's actions. Training-only instances
        # may omit robot_config, but inference refuses to emit normalized canonical values.
        self._action_unapply_ft = self._build_action_unapply_transform()

        # Opt-in torch.compile for the denoise inner loop (see config docs).
        if getattr(self.config, "compile_predict_velocity", False):
            self.model._use_compile_predict_velocity = True
            self.model._compile_predict_velocity_mode = getattr(
                self.config, "compile_predict_velocity_mode", "default"
            )
            if getattr(self.config, "compile_prefix", False):
                self.model._use_compile_prefix = True
            # handle_kv_cache specializes per layer_idx (36 layers); the default
            # recompile limit (8) silently falls parts of the fn back to eager
            # mid-run. Raising it here covers the deployment path — the bench
            # harness only raises it for its own --compile flag.
            import torch._dynamo as _dynamo

            _dynamo.config.recompile_limit = max(
                getattr(_dynamo.config, "recompile_limit", 8), 64
            )
        # Independent of the compile flags (see config docs): the prefix CUDA
        # graph supersedes compile_prefix when both are set.
        if getattr(self.config, "use_cudagraph_prefix", False):
            self.model._use_prefix_graph = True
        # use_cudagraph_prefix_full lives inside _prefix_graphed (vision graph + LLM
        # fill graph), so it implies the prefix graph is active.
        if getattr(self.config, "use_cudagraph_prefix_full", False):
            self.model._use_prefix_graph = True

        # Frozen distillation teachers (native depth / DINO-video), built lazily on
        # the first *training* forward when align_params is set. Plain attribute on
        # purpose: see the "Distillation teachers" block below get_optim_params.
        self._align_teachers = None

        self.reset()
        torch.set_float32_matmul_precision("high")

    def _build_action_unapply_transform(self):
        """Build a lightweight (processor-free) FeatureTransform for inference unapply."""
        cfg = self.config
        try:
            resolve_robot_config_and_stats(cfg)
        except OSError as exc:
            raise RuntimeError(
                "Could not load the robot config for the inference action de-normalizer. "
                "Returning normalized canonical actions as robot commands is unsafe; "
                f"fix the checkpoint's robot_config assets. ({exc})"
            ) from exc
        if not getattr(cfg, "robot_config", None):
            # Training-only instances (forward/loss path) never call select_action;
            # keep them constructible. _postprocess_actions raises if used for inference.
            return None
        try:
            from .configuration_lingbot_vla_v2 import build_feature_transform_configs
            from .preprocessing.feature_transform import FeatureTransform

            data_config, model_config = build_feature_transform_configs(cfg)
            return FeatureTransform(
                robot_config_path=cfg.robot_config_path,
                data_config=data_config,
                model_config=model_config,
                processor=None,
                chunk_size=cfg.chunk_size,
                norm_stats_path=cfg.norm_stats_path,
                robot_config=cfg.robot_config,
                norm_stats=cfg.norm_stats,
            )
        except Exception as exc:  # noqa: BLE001 - de-normalizer is build-time critical
            raise RuntimeError(
                "Could not build the inference action de-normalizer. Returning normalized "
                f"canonical actions as robot commands is unsafe. ({exc})"
            ) from exc

    def _postprocess_actions(self, actions: Tensor, batch: dict) -> Tensor:
        """Invert normalization + the canonical slot mapping on a model action chunk.

        ``actions`` is ``(B, chunk, max_action_dim)`` in the normalized canonical space.
        Uses the per-joint masks and observation state carried in the (preprocessed) batch.
        Raises when those safety-critical inputs are unavailable rather than emitting
        normalized canonical values as robot commands.
        """
        ft = self._action_unapply_ft
        action_joint_mask = batch.get("action_joint_mask")
        state_joint_mask = batch.get("state_joint_mask")
        state = batch.get(OBS_STATE)
        if ft is None:
            raise RuntimeError(
                "No action de-normalizer available: this policy instance was built without a "
                "robot_config, so select_action cannot map canonical actions back to robot "
                "commands. Refusing to return normalized canonical values as robot commands."
            )
        if action_joint_mask is None or state_joint_mask is None or state is None:
            raise RuntimeError(
                "Batch is missing the joint masks / observation state required to invert the "
                "canonical slot mapping (got "
                f"action_joint_mask={action_joint_mask is not None}, "
                f"state_joint_mask={state_joint_mask is not None}, state={state is not None}). "
                "Refusing to fall back to a truncation of normalized canonical actions."
            )

        recovered = []
        for i in range(actions.shape[0]):
            item = {
                "actions": actions[i].detach().to("cpu", torch.float32),
                "action_joint_mask": action_joint_mask[i].detach().to("cpu"),
                "state": state[i].detach().to("cpu", torch.float32),
                "state_joint_mask": state_joint_mask[i].detach().to("cpu"),
            }
            recovered.append(ft.unapply(item)[ACTION])
        return torch.stack(recovered, dim=0).to(actions.device)

    def reset(self):
        """Reset the rolling action queue used by select_action."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict[str, torch.nn.Parameter]:
        # Frozen parameters never receive a gradient, so the optimizer would only
        # carry them along unused. Filtering here keeps optimizer state memory at
        # the trainable subset — with PEFT that is ~0.2B adapter params instead of 6B.
        # Name-keyed so the optimizer preset can group by FQN (lingbot_adamw's MoE
        # expert-LR scaling matches `...layers.<N>.mlp.experts...` by name).
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    # ==================== Distillation teachers (native depth / DINO-video) ====================
    # Port of the upstream trainer's per-micro-batch teacher block (tasks/vla/
    # train_lingbotvla.py): frozen MoGe/MoRGBD (+ optional DINO-video) teachers run
    # under no_grad + bf16 autocast on the raw pre-Qwen camera frames the processor
    # carries as ``pil_images`` / ``future_pil_images``, and their outputs are fed
    # to the model as loss targets. The bundle is a plain attribute — never an
    # nn.Module registration — so teacher weights stay out of optimizers, DDP/FSDP
    # wrapping, and saved checkpoints. Each DDP rank builds its own frozen copy.

    def _ensure_align_teachers(self) -> DepthTeacherBundle:  # noqa: F821
        if self._align_teachers is None:
            from .teachers.depth_teachers import DepthTeacherBundle

            device = next(self.model.parameters()).device
            self._align_teachers = DepthTeacherBundle.build(self.config.align_params, device)
        return self._align_teachers

    def _compute_align_targets(self, batch: dict) -> dict:
        """Teacher targets for one training batch, as model-forward kwargs."""
        params = self.config.align_params
        use_future_depth = bool(params["depth"].get("use_future_depth", False))
        use_future_video = bool(params.get("use_future_video", False))

        pil_images = batch.get("pil_images")
        if pil_images is None:
            raise RuntimeError(
                "align_params is enabled but the batch carries no 'pil_images'. The "
                "preprocessor step was built without use_depth_align=True — this happens "
                "when training from a checkpoint whose saved processor predates the "
                "distillation wiring, or when the processor was constructed from a "
                "config without align_params. Rebuild with the align_params-carrying "
                "config (see docs/source/lingbot_vla_v2_depth_dino_README.md)."
            )

        teachers = self._ensure_align_teachers()
        targets: dict = {}
        with torch.no_grad():
            targets["depth_targets"] = teachers.depth_targets(pil_images)
            if use_future_depth:
                future_pil = batch.get("future_pil_images")
                if future_pil is None:
                    raise RuntimeError(
                        "align_params.depth.use_future_depth is enabled but the batch carries no "
                        "'future_pil_images'. Set --policy.dataset_fps and confirm the dataset "
                        "delta sampling is active (future-frame keys are produced only when the "
                        "processor step has use_future_image=True)."
                    )
                targets["future_depth_targets"] = teachers.depth_targets(future_pil)
            if use_future_video:
                future_pil = batch.get("future_pil_images")
                if future_pil is None:
                    raise RuntimeError(
                        "align_params.use_future_video is enabled but the batch carries no "
                        "'future_pil_images'. The DINO-video teacher needs the future camera "
                        "frame; confirm the processor step has use_future_image=True "
                        "(--policy.dataset_fps must be resolvable, see the depth/DINO README)."
                    )
                bundle = teachers.video_targets(
                    pil_images,
                    future_pil,
                    params["video"],
                    effective_fps=batch.get("future_video_effective_fps"),
                )
                if isinstance(bundle, dict):
                    targets["future_video_targets"] = bundle["patch"]
                    targets["future_video_cls_targets"] = bundle.get("cls")
                    targets["future_video_current_patch"] = bundle.get("current_patch")
                elif isinstance(bundle, tuple):
                    targets["future_video_targets"], targets["future_video_cls_targets"] = bundle
                else:
                    targets["future_video_targets"] = bundle
        return targets

    # ==================== PEFT (LoRA) integration ====================
    # The community PEFT path (lerobot `--peft.*` CLI → wrap_with_peft →
    # save/resume via adapter checkpoints) works with this policy through the two
    # subclass hooks below. Targeting notes specific to this architecture:
    # - Both the Qwen3-VL LLM and the action expert name their attention
    #   projections q/k/v/o_proj, so one suffix list covers both streams.
    # - The vision tower uses a fused `qkv` projection and is not matched (it is
    #   frozen via freeze_vision_encoder regardless).
    # - The MoE router (`...mlp.gate`, a hidden×num_experts Linear) stays fully
    #   trainable via modules_to_save: freezing the routing distribution hurts
    #   fine-tuning on new robot data, and it is tiny.
    # - The routed experts are stored as fused grouped GEMMs (Qwen2FusedExperts,
    #   plain Parameters — not nn.Linear), so stock LoRA cannot target them.

    def _get_default_peft_targets(self) -> dict[str, any] | None:
        return {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "modules_to_save": ["gate"],
        }

    def _validate_peft_config(self, peft_config) -> None:
        super()._validate_peft_config(peft_config)
        targets = getattr(peft_config, "target_modules", None) or []
        if isinstance(targets, str):
            targets = [targets]
        mlp_targets = {"gate_proj", "up_proj", "down_proj"} & set(targets)
        if mlp_targets:
            logging.get_logger(__name__).warning(
                "PEFT target_modules %s only match the shared-expert MLP (nn.Linear); the routed "
                "experts use fused grouped-GEMM storage (Qwen2FusedExperts) and will NOT be adapted.",
                sorted(mlp_targets),
            )
        if not getattr(self.config, "gradient_checkpointing", False):
            logging.get_logger(__name__).warning(
                "LoRA adapters are inside every decoder layer, so backward still traverses the "
                "frozen backbone's activations. Consider --policy.gradient_checkpointing=true to "
                "cut activation memory."
            )

    def _extract_model_inputs(self, batch: dict):
        dtype = next(self.parameters()).dtype
        images = batch["images"].to(dtype=dtype)
        img_masks = batch["img_masks"]
        lang_tokens = batch["lang_tokens"]
        lang_masks = batch["lang_masks"]
        state = batch[OBS_STATE].to(dtype=dtype)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        image_grid_thw = batch.get("image_grid_thw")
        return images, img_masks, lang_tokens, lang_masks, state, image_grid_thw

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        """Training forward pass returning the flow-matching loss (lerobot convention)."""
        images, img_masks, lang_tokens, lang_masks, state, image_grid_thw = self._extract_model_inputs(batch)
        actions = batch[ACTION].to(dtype=state.dtype)
        action_dim = actions.shape[-1]
        actions = F.pad(actions, (0, self.config.max_action_dim - action_dim))

        # MoE monitoring metrics (per-layer stats + .item() syncs) only on logging steps.
        self._train_step_count = getattr(self, "_train_step_count", 0) + 1
        interval = max(1, int(getattr(self.config, "moe_metrics_interval", 1)))
        collect_metrics = self._train_step_count % interval == 0

        # External targets remain supported for diagnostic/unit-test callers, but
        # real training computes them from frozen teachers when the native-depth
        # branch is configured. Inference follows predict_action_chunk instead and
        # never builds/runs teachers.
        align_targets = self._compute_align_targets(batch) if self.training and self.config.align_params else {}

        (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            _depth_preds,
            seq_wise_loss,
            router_z_loss,
            moe_metrics,
            _future_depth_preds,
            _future_video_preds,
            _current_video_preds,
        ) = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise=batch.get("noise"),
            time=batch.get("time"),
            loss_type=self.config.loss_type,
            depth_targets=align_targets.get("depth_targets", batch.get("depth_targets")),
            image_grid_thw=image_grid_thw,
            future_depth_targets=align_targets.get("future_depth_targets", batch.get("future_depth_targets")),
            future_video_targets=align_targets.get("future_video_targets", batch.get("future_video_targets")),
            future_video_cls_targets=align_targets.get(
                "future_video_cls_targets", batch.get("future_video_cls_targets")
            ),
            future_video_current_patch=align_targets.get(
                "future_video_current_patch", batch.get("future_video_current_patch")
            ),
            collect_metrics=collect_metrics,
        )

        joint_mask = batch.get("joint_mask")
        if joint_mask is not None:
            masked_losses = losses * joint_mask
            loss_vla = masked_losses.sum() / joint_mask.sum().clamp(min=1)
        else:
            loss_vla = losses[:, :, :action_dim].mean()

        loss_dict: dict = {"l2_loss": loss_vla.item()}
        total_loss = loss_vla
        for loss_name, term in (
            ("depth_loss", loss_depth),
            ("future_depth_loss", loss_future_depth),
            ("future_video_loss", loss_future_video),
            ("seq_wise_loss", seq_wise_loss),
            ("router_z_loss", router_z_loss),
        ):
            if torch.is_tensor(term):
                loss_dict[loss_name] = term.item()
                total_loss = total_loss + term
        if moe_metrics:
            loss_dict.update({k: (v.item() if torch.is_tensor(v) else v) for k, v in moe_metrics.items()})
        loss_dict["loss"] = total_loss.item()
        return total_loss, loss_dict

    @staticmethod
    def supports_rtc() -> bool:
        """Declare RTC inference support: ``lerobot-rollout --inference.type=rtc``."""
        return True

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict,
        noise: Tensor | None = None,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """Run flow-matching denoising and return a de-normalized action chunk (B, chunk, action_dim).

        ``sample_actions`` returns the normalized 55-D canonical action; this inverts the
        per-slot normalization and the canonical slot mapping back to the raw dataset action
        (see ``_postprocess_actions``).

        RTC args (from the rollout RTC engine): ``prev_chunk_left_over`` is the
        normalized leftover prefix of the previous chunk (already truncated to the
        execution horizon); ``inference_delay`` is the chunk's reaction lag in action
        steps. Passing ``None`` (default) reproduces the plain unguided sampling.
        """
        self.eval()
        images, img_masks, lang_tokens, lang_masks, state, image_grid_thw = self._extract_model_inputs(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            image_grid_thw=image_grid_thw,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
        )
        # The RTC engine keeps a guidance reference in the SAMPLING (normalized) space;
        # this policy's public output is already de-normalized, so stash the pre-inversion
        # chunk for it (see get_last_normalized_chunk).
        self._last_normalized_chunk = actions.detach()
        return self._postprocess_actions(actions, batch)

    def get_last_normalized_chunk(self) -> Tensor:
        """Normalized sampling-space output of the most recent ``predict_action_chunk``."""
        return self._last_normalized_chunk

    @torch.no_grad()
    def select_action(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        """Select a single action for environment execution, buffering chunks in a queue."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()
