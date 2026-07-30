# Copyright (C) 2026 Tencent.  All rights reserved.
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

# ruff: noqa: B023, N806

"""Hy-VLA dual-tower module.

``HyDualTower`` pairs a VLM with an action-expert decoder under a
shared-attention forward. Both slots are architecture-neutral
``nn.Module`` attributes; any HuggingFace-compatible CausalLM that
exposes the standard transformers decoder-layer contract
(``layers[i].{input_layernorm, self_attn, post_attention_layernorm,
mlp}``, plus the MoT extensions ``input_layernorm_v`` /
``post_attention_layernorm_v`` / ``self_attn.{q,k,v,o}_proj_v`` and
``mlp_v`` for the modality-aware variant) can be dropped in.
"""

import types

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache

from .hunyuan_vl_mot import HunYuanVLMoTForConditionalGeneration
from .hunyuan_vl_mot.modeling_hunyuan_vl_mot import (
    _HunYuanVLMoTTextForCausalLM,
)


def mask_apply(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    text_funcs,
    vision_funcs,
    out_dims=None,
):
    """Batch-flattened modality routing for the MoT dual-tower forward.

    Args:
        hidden_states: ``(B, S, D)`` token features.
        mask: ``(B, S)`` bool / int. ``True`` (or ``1``) -> vision token,
            ``False`` (or ``0``) -> text token.
        text_funcs: callables applied to text tokens (one per output).
        vision_funcs: callables applied to vision tokens (one per output).
        out_dims: optional list of per-output last-dim sizes. ``None``
            means each output keeps the input ``D``.

    Returns:
        ``list[Tensor]`` with shape ``(B, S, out_dim_i)``; entries the
        functions did not write are zeros (``torch.empty`` slots that
        were never indexed are explicitly zero-initialised when neither
        modality covers the full sequence, see below).
    """
    B, S, D = hidden_states.size()
    flat = hidden_states.reshape(B * S, D)
    mask_flat = mask.reshape(B * S).bool()

    if out_dims is None:
        out_flat = [torch.zeros(B * S, D, device=flat.device, dtype=flat.dtype) for _ in text_funcs]
    else:
        out_flat = [torch.zeros(B * S, od, device=flat.device, dtype=flat.dtype) for od in out_dims]

    text_idx = ~mask_flat
    if text_idx.any():
        hs_t = flat[text_idx]
        for i, fn in enumerate(text_funcs):
            out_flat[i][text_idx] = fn(hs_t)

    vis_idx = mask_flat
    if vis_idx.any():
        hs_v = flat[vis_idx]
        for i, fn in enumerate(vision_funcs):
            out_flat[i][vis_idx] = fn(hs_v)

    return [o.view(B, S, -1) for o in out_flat]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to the query and key tensors.

    Args:
        q: query tensor.
        k: key tensor.
        cos: cosine part of the rotary embedding.
        sin: sine part of the rotary embedding.
        position_ids: unused, kept for signature compatibility.
        unsqueeze_dim: which axis to unsqueeze on for broadcasting.

    Returns:
        ``(q_rotated, k_rotated)``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class HyDualTowerConfig(PretrainedConfig):
    """Config for :class:`HyDualTower`.

    Both ``vlm_config`` and ``expert_config`` are full ``PretrainedConfig``
    instances and are passed in directly.
    """

    model_type = "hy_dual_tower"
    sub_configs = {"vlm_config": AutoConfig, "expert_config": AutoConfig}

    def __init__(
        self,
        vlm_config: PretrainedConfig | None = None,
        expert_config: PretrainedConfig | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.vlm_config = vlm_config
        self.expert_config = expert_config

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation

        # Optional reference to the outer ``HyVLAConfig`` (used by
        # HyVLAFlowMatching to keep its proj_width in sync with the
        # expert tower's hidden_size).
        self.config = kwargs.pop("config", None)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )
        if self.attention_implementation != "eager":
            raise ValueError(
                f"Wrong value provided for `attention_implementation` "
                f"({self.attention_implementation}). Expected 'eager'."
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HyDualTower(PreTrainedModel):
    """Plug-in dual-tower container: VLM + action expert with shared attention.

    The two slot attributes ``self.vlm`` and ``self.expert`` are
    architecture-neutral. The default factory uses
    :class:`HunYuanVLMoTForConditionalGeneration` and
    :class:`HunYuanDenseV1MoTForCausalLM`; any HuggingFace-style decoder
    that satisfies the modality-aware MoT layer contract can be plugged
    in via :meth:`from_components`.

    State-dict layout:
        ``vlm.<rest>``     -- VLM weights
        ``expert.<rest>``  -- expert weights
    """

    config_class = HyDualTowerConfig

    def __init__(self, config: HyDualTowerConfig):
        super().__init__(config=config)
        self.config = config
        self.vlm = HunYuanVLMoTForConditionalGeneration(config=config.vlm_config)
        # Action-expert: inner LM-head wrapper of the same HunYuanVLMoT
        # family. Picking *ForCausalLM* preserves the attribute layout
        # ``expert.model.{layers, norm, rotary_emb, embed_tokens}`` and
        # the external ``expert.lm_head``.
        self.expert = _HunYuanVLMoTTextForCausalLM(config=config.expert_config)
        # The expert reuses the VLM tokenizer's text embeddings (set
        # externally by ``HyVLAFlowMatching``); drop the unused
        # expert-side embed_tokens to save memory.
        self.expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    # ------------------------------------------------------------------
    # Component-level factory (the "plug-in" contract).
    # ------------------------------------------------------------------
    @classmethod
    def from_components(
        cls,
        *,
        vlm: PreTrainedModel,
        expert: PreTrainedModel,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        outer_config: PretrainedConfig | None = None,
    ) -> "HyDualTower":
        """Build a ``HyDualTower`` from pre-instantiated VLM / expert modules.

        The returned tower assumes ownership of both modules; the caller
        should not keep separate references. Useful for swapping in
        custom backbones without subclassing.

        Example::

            from lerobot.policies.hy_vla.modeling.modeling_dual_tower import HyDualTower

            tower = HyDualTower.from_components(
                vlm=MyCustomVLM.from_pretrained("..."),
                expert=MyCustomExpert.from_pretrained("..."),
            )
        """
        cfg = HyDualTowerConfig(
            vlm_config=vlm.config,
            expert_config=expert.config,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            attention_implementation=attention_implementation,
            config=outer_config,
        )
        instance = cls.__new__(cls)
        PreTrainedModel.__init__(instance, config=cfg)
        instance.config = cfg
        instance.vlm = vlm
        instance.expert = expert
        if instance.expert.model.embed_tokens is not None:
            instance.expert.model.embed_tokens = None
        instance.to_bfloat16_like_physical_intelligence()
        instance.set_requires_grad()
        return instance

    # ------------------------------------------------------------------
    # Training-mode toggles
    # ------------------------------------------------------------------
    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.vlm.model.visual.eval()
            for params in self.vlm.model.visual.parameters():
                params.requires_grad = False
        else:
            self._unfreeze_vision_tower_inplace(self.vlm.model.visual)

        if self.config.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.vlm.model.visual.eval()

        if self.config.train_expert_only:
            self.vlm.eval()

    @staticmethod
    def _unfreeze_vision_tower_inplace(visual_module: nn.Module) -> None:
        # Step 1: parameters
        for p in visual_module.parameters():
            p.requires_grad = True

        # Step 2: forward patch. Bind once -- subsequent calls overwrite
        # with the same closure so re-entry is harmless.
        def _forward_with_grad(self, images, cal_attn_pool=False):
            # Mirrors HYViT2_400MAnyRes.forward but without ``no_grad``.
            image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)
            if isinstance(images, list):
                image_features = [
                    self.merger(x, s).squeeze(0) for x, s in zip(image_features, img_size, strict=False)
                ]
            else:
                image_features = self.merger(image_features, img_size)
                C = image_features.shape[-1]
                image_features = [image_features.reshape(-1, C)]
            return image_features

        visual_module.forward = types.MethodType(_forward_with_grad, visual_module)

    def to_bfloat16_like_physical_intelligence(self):
        """Mirror the openpi-style precision policy.

        Cast the entire VLM tower + the layer parameters of both towers
        to bfloat16; everything outside ``layers`` / ``visual`` keeps its
        original dtype (typically fp32 for embeddings and projections).
        """
        self.vlm = self.vlm.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "expert.model.layers",
            "visual",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Visual + language token embedders (called by the outer policy)
    # ------------------------------------------------------------------
    def embed_image(self, image: torch.Tensor):
        """Encode RGB inputs through the VLM vision tower.

        Args:
            image: ``(C, H, W)`` (single frame), ``(B, C, H, W)`` (batch
                of single frames), or ``(B, K, C, H, W)`` (MEM video
                stack -- routed to the SpaceTime-augmented ViT).

        Returns:
            ``(B, N, D)`` patch features, where ``N`` is the per-frame
            (or per-stack) token count and ``D`` is the visual hidden
            dim.
        """
        if image.dim() == 5:
            # Wrapper returns [(B*N, C)] (batch flattened); restore to (B, N, C).
            B = image.shape[0]
            feat = self.vlm.visual(image)[0]
            return feat.view(B, -1, feat.shape[-1]).contiguous()

        image_list = list(image.unsqueeze(1) if image.dim() == 3 else image.split(1, dim=0))
        # image_list: list of (1, 3, h, w)
        image_features = self.vlm.visual(image_list)  # list of (num_tokens, 2048)
        image_features = torch.stack(image_features, dim=0)
        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Look up text-token embeddings via the VLM language tower."""
        return self.vlm.language_model.model.embed_tokens(tokens)

    # ------------------------------------------------------------------
    # Shared-attention dual-tower forward
    # ------------------------------------------------------------------
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        modality_masks: list[torch.FloatTensor] = None,
    ):
        models = [self.vlm.language_model.model, self.expert.model]
        att_vis_output = []
        prefix_emb_layer_outputs = []
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.vlm.config.num_hidden_layers

        # ``position_embeddings`` are constant across layers; compute once.
        # The first arg picks output dtype/device, so we pass a float
        # tensor (not the int64 ``position_ids``).
        _dtype_ref = next(h for h in inputs_embeds if h is not None)
        position_embeddings = models[0].rotary_emb(_dtype_ref.float(), position_ids)

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []

            # Per-tower sequence length (used to slice the concatenated
            # q/k tensors before the per-tower q/k layernorm).
            seq_len_list = []

            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                modality_mask = modality_masks[i]

                hidden_states = mask_apply(
                    hidden_states,
                    modality_mask,
                    [lambda x: layer.input_layernorm(x)],
                    [lambda x: layer.input_layernorm_v(x)],
                )[0]

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                # Batch-flattened modality routing (see ``mask_apply``
                # docstring). The dual-tower forward always supplies a
                # non-None ``modality_mask`` so we bypass the bundled model's
                # per-sample fallback path entirely.
                query_state, key_state, value_state = mask_apply(
                    hidden_states,
                    modality_mask,
                    [
                        lambda x: layer.self_attn.q_proj(x),
                        lambda x: layer.self_attn.k_proj(x),
                        lambda x: layer.self_attn.v_proj(x),
                    ],
                    [
                        lambda x: layer.self_attn.q_proj_v(x),
                        lambda x: layer.self_attn.k_proj_v(x),
                        lambda x: layer.self_attn.v_proj_v(x),
                    ],
                    out_dims=[
                        self.config.vlm_config.num_attention_heads * layer.self_attn.head_dim,
                        self.config.vlm_config.num_key_value_heads * layer.self_attn.head_dim,
                        self.config.vlm_config.num_key_value_heads * layer.self_attn.head_dim,
                    ],
                )

                # (batch_size, num_heads, seq_len, head_dim)
                query_state = query_state.view(hidden_shape).transpose(1, 2)
                key_state = key_state.view(hidden_shape).transpose(1, 2)
                value_state = value_state.view(hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
                seq_len_list.append(hidden_states.shape[1])

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            q_parts = query_states.split(seq_len_list, dim=2)
            k_parts = key_states.split(seq_len_list, dim=2)
            q_normed = []
            k_normed = []

            vlm_layer = models[0].layers[layer_idx]
            for q_part, k_part in zip(q_parts, k_parts, strict=False):
                q_normed.append(vlm_layer.self_attn.query_layernorm(q_part))
                k_normed.append(vlm_layer.self_attn.key_layernorm(k_part))
            query_states = torch.cat(q_normed, dim=2)
            key_states = torch.cat(k_normed, dim=2)

            # (batch_size, seq_len, num_heads, head_dim)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )
                    past_key_values[layer_idx]["key_states"] = key_states
                    past_key_values[layer_idx]["value_states"] = value_states

            attention_interface = self.get_attention_interface()
            att_output, probs = attention_interface(
                attention_mask,
                batch_size,
                layer.self_attn.head_dim,
                query_states,
                key_states,
                value_states,
            )

            att_output = att_output.to(dtype=torch.bfloat16)  # (b, seq_vlm, ...)
            att_vis_output.append(probs)  # probs (b, 8, seq, seq)

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                modality_mask = modality_masks[i]
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

                    out_emb = mask_apply(
                        att_output[:, start:end],
                        modality_mask,
                        [lambda x: layer.self_attn.o_proj(x)],
                        [lambda x: layer.self_attn.o_proj_v(x)],
                        out_dims=[models[i].config.hidden_size],
                    )[0]

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()
                    out_emb = mask_apply(
                        out_emb,
                        modality_mask,
                        [lambda x: layer.mlp(layer.post_attention_layernorm(x))],
                        [lambda x: layer.mlp_v(layer.post_attention_layernorm_v(x))],
                    )[0]

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            prefix_emb_layer_outputs.append(outputs_embeds[0])
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values, att_vis_output, prefix_emb_layer_outputs

    # ------------------------------------------------------------------
    # Attention backends
    # ------------------------------------------------------------------
    def get_attention_interface(self):
        return self.eager_attention_forward

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.vlm_config.num_attention_heads
        num_key_value_heads = self.config.vlm_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # bf16 -inf approximation

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output, probs


__all__ = ["HyDualTowerConfig", "HyDualTower"]
