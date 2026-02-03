#!/usr/bin/env python

# Copyright 2026 S-Lab and The HuggingFace Inc. team. All rights reserved.
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

import collections
import copy
import logging

import torch
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel


class VLMWithExpertModel(torch.nn.Module):
    def __init__(
        self,
        model_id: str,
        vlm: PreTrainedModel,
        freeze_text_model: bool = False,
        freeze_connector: bool = False,
        freeze_vision_model: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_expert_skip_layers: int = 0,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
    ) -> None:
        super().__init__()
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # VLM
        self.vlm = vlm
        self.vlm_config = self.vlm.config
        if hasattr(self.vlm, "lm_head"):
            del self.vlm.lm_head
        if num_vlm_layers > 0:
            logging.info(
                "Reducing the number of VLM layers from %d to %d ..."
                % (
                    len(self.get_vlm_model().text_model.layers),
                    num_vlm_layers,
                )
            )
            del self.get_vlm_model().text_model.layers[num_vlm_layers:]

        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        # Action Expert
        lm_expert_config = self._get_expert_config(
            self.vlm_config.text_config, num_expert_layers, expert_width_multiplier
        )
        self.lm_expert = self._get_expert_model(
            lm_expert_config,
            attention_mode,
            num_expert_skip_layers,
            self_attn_every_n_layers,
        )
        # Remove token embeddings
        if hasattr(self.lm_expert, "embed_tokens"):
            del self.lm_expert.embed_tokens

        self.num_attention_heads = self.vlm_config.text_config.num_attention_heads
        self.num_key_value_heads = self.vlm_config.text_config.num_key_value_heads
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.num_expert_layers = len(self.lm_expert.layers) - num_expert_skip_layers
        self.num_expert_skip_layers = num_expert_skip_layers
        self.self_attn_every_n_layers = self_attn_every_n_layers

        self.freeze_vision_model = freeze_vision_model
        self.freeze_connector = freeze_connector
        self.freeze_text_model = freeze_text_model
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self._set_requires_grad()

    def _get_expert_config(
        self, text_config: PretrainedConfig, num_layers: int, width_multiplier: float
    ) -> PretrainedConfig:
        expert_config = copy.deepcopy(text_config)
        hidden_size = expert_config.hidden_size
        expert_config.hidden_size = int(hidden_size * width_multiplier)
        expert_config.intermediate_size = self._get_intermediate_size(
            expert_config.hidden_size
        )
        num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        expert_config.num_hidden_layers = num_vlm_layers
        if num_layers > 0:
            assert (
                len(num_vlm_layers) % num_layers == 0
            ), "Number of layers in the VLM %d are not multiple of %d " % (
                num_vlm_layers,
                num_layers,
            )
            expert_config.num_hidden_layers = num_layers

        return expert_config

    def _get_intermediate_size(
        self, hidden_dim: int, ffn_dim_multiplier: int = 4, multiple_of: int = 256
    ) -> int:
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        return hidden_dim

    def _get_expert_model(
        self,
        expert_config: PretrainedConfig,
        attention_mode: str,
        num_expert_skip_layers: int,
        self_attn_every_n_layers: int,
    ) -> PreTrainedModel:
        text_config = self.vlm_config.text_config
        expert_model = AutoModel.from_config(expert_config)
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(num_expert_skip_layers, len(expert_model.layers)):
                if (
                    self_attn_every_n_layers > 0
                    and layer_idx % self_attn_every_n_layers == 0
                ):
                    continue

                # Remove projectors for key (as ROPE has been applied in VLM)
                expert_model.layers[layer_idx].self_attn.v_proj = torch.nn.Linear(
                    text_config.num_key_value_heads * text_config.head_dim,
                    expert_config.num_key_value_heads * expert_config.head_dim,
                    bias=expert_config.attention_bias,
                )
                del expert_model.layers[layer_idx].self_attn.k_proj

        return expert_model

    def get_vlm_model(self) -> PreTrainedModel:
        return self.vlm.model

    def _set_requires_grad(self) -> None:
        if self.freeze_vision_model:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.freeze_text_model:
            self.get_vlm_model().text_model.eval()
            for params in self.get_vlm_model().text_model.parameters():
                params.requires_grad = False
        if self.freeze_connector and hasattr(self.get_vlm_model(), "connector"):
            self.get_vlm_model().connector.eval()
            for params in self.get_vlm_model().connector.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if self.freeze_vision_model:
            self.get_vlm_model().vision_model.eval()
        if self.freeze_connector and hasattr(self.get_vlm_model(), "connector"):
            self.get_vlm_model().connector.eval()
        if self.freeze_text_model:
            self.get_vlm_model().text_model.eval()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) in [
            4,
            5,
        ], f"Image should be [B, C, H, W] or [B, N, C, H, W], got {image.shape}"
        if len(image.shape) == 4:
            image = image.unsqueeze(1)  # [B, 1, C, H, W]

        return self.get_vlm_model().get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def _qkv_proj_layer(
        self,
        layer: torch.nn.Module,
        position_ids: torch.Tensor,
        q_in: torch.Tensor,
        k_in: torch.Tensor | None = None,
        v_in: torch.Tensor | None = None,
    ) -> tuple[
        dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]
    ]:
        assert q_in is not None, "q_in should not be None"

        attn_layer = layer.self_attn
        q_in = layer.input_layernorm(q_in).to(dtype=attn_layer.q_proj.weight.dtype)
        if k_in is None and v_in is None:
            k_in = q_in
            v_in = q_in
        else:
            v_in = v_in.to(dtype=attn_layer.v_proj.weight.dtype)
            if hasattr(attn_layer, "k_proj"):
                k_in = k_in.to(dtype=attn_layer.k_proj.weight.dtype)

        v_shape = (*v_in.shape[:-1], -1, attn_layer.head_dim)
        q_shape = (*q_in.shape[:-1], -1, attn_layer.head_dim)
        if hasattr(attn_layer, "k_proj"):
            k_shape = (*k_in.shape[:-1], -1, attn_layer.head_dim)

        q_states, k_states, v_states = {}, {}, {}
        if position_ids.ndim == 2:  # 1D Rope
            v_states["t"] = attn_layer.v_proj(v_in).view(v_shape)
            q_states["t"] = attn_layer.q_proj(q_in).view(q_shape)
            if hasattr(attn_layer, "q_norm"):
                q_states["t"] = attn_layer.q_norm(q_states["t"])
            if hasattr(attn_layer, "k_proj"):
                k_states["t"] = attn_layer.k_proj(k_in).view(k_shape)
                if hasattr(attn_layer, "k_norm"):
                    k_states["t"] = attn_layer.k_norm(k_states["t"])
        elif position_ids.ndim == 3 and position_ids.shape[2] == 3:  # 3D Rope
            qtr_head_dim = attn_layer.head_dim // 4
            # Value
            v_states["t"] = attn_layer.v_proj(v_in).view(v_shape)
            # Query
            _query_states = attn_layer.q_norm(attn_layer.q_proj(q_in).view(q_shape))
            q_states["t"] = _query_states[..., : qtr_head_dim * 2]
            q_states["h"] = _query_states[..., qtr_head_dim * 2 : -qtr_head_dim]
            q_states["w"] = _query_states[..., -qtr_head_dim:]
            # Key (can be skipped in cross-attention)
            if hasattr(attn_layer, "k_proj"):
                _key_states = attn_layer.k_norm(attn_layer.k_proj(k_in).view(k_shape))
                k_states["t"] = _key_states[..., : qtr_head_dim * 2]
                k_states["h"] = _key_states[..., qtr_head_dim * 2 : -qtr_head_dim]
                k_states["w"] = _key_states[..., -qtr_head_dim:]
        else:
            raise ValueError(f"Unknown position_ids shape: {position_ids.shape}")

        return q_states, k_states, v_states

    def apply_rope(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        wavelength: int = 10_000,
    ):
        if positions.ndim == 2:  # 1D Rope
            return self._apply_rope(
                hidden_states["t"].unsqueeze(0), positions.unsqueeze(0), wavelength
            ).squeeze(0)
        elif positions.ndim == 3 and positions.shape[2] == 3:  # 3D Rope
            roped_states = self._apply_rope(
                torch.stack(
                    [
                        hidden_states["t"],
                        hidden_states["h"].repeat_interleave(2, dim=-1),
                        hidden_states["w"].repeat_interleave(2, dim=-1),
                    ],
                    dim=0,
                ),
                positions.permute(2, 0, 1),
                wavelength,
            )
            half_indexes = torch.arange(
                0, hidden_states["t"].size(-1), 2, device=roped_states.device
            )
            roped_states = roped_states.permute(1, 2, 3, 4, 0)
            return torch.cat(
                [
                    roped_states[..., 0],
                    roped_states[..., half_indexes, 1],
                    roped_states[..., half_indexes, 2],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown position ID shape: {positions.shape}")

    def _apply_rope(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        max_wavelength: int = 10_000,
    ):
        """
        Applies RoPE positions [B, L, N] to hidden_states [B, L, H, D].
        """
        # Cache the sin/cos values for efficiency
        d_half = hidden_states.size(-1) // 2
        dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)

        freq_exponents = (4.0 / d_half) * torch.arange(
            d_half, dtype=torch.float32, device=positions.device
        )
        timescale = max_wavelength**freq_exponents
        radians = positions[..., None] / timescale[None, None, None, :]
        radians = radians[..., None, :]
        cos = torch.cos(radians)
        sin = torch.sin(radians)

        x1, x2 = hidden_states.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin
        return res.to(dtype)

    def _self_attn_layer(
        self,
        model_layer: list[torch.nn.Module],
        inputs_embeds: list[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_dim: int,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        query_states = collections.defaultdict(list)
        key_states = collections.defaultdict(list)
        value_states = collections.defaultdict(list)
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layer[i]
            if hidden_states is None or layer is None:
                continue

            q_states, k_states, v_states = self._qkv_proj_layer(
                layer, position_ids, hidden_states
            )
            for dst, src in (
                (query_states, q_states),
                (key_states, k_states),
                (value_states, v_states),
            ):
                for k, v in src.items():
                    dst[k].append(v)

        # B, L, H, D with L sequence length, H number of heads, D head dim
        # Concatenate on the number of embeddings/tokens
        for states in (query_states, key_states, value_states):
            for k, v_list in states.items():
                states[k] = torch.cat(v_list, dim=1)

        # Both VLM and Expert are empty. May happen during inference.
        if len(query_states["t"]) == 0:
            return None, past_key_values

        seq_len = query_states["t"].shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        # Rotary Position Embedding
        value_states = value_states["t"]
        query_states = self.apply_rope(query_states, _position_ids)
        key_states = self.apply_rope(key_states, _position_ids)

        # KV Cache
        if use_cache and past_key_values is not None:
            if fill_kv_cache:
                past_key_values = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:  # TODO: some optimization can be done, similar to `StaticCache`
                key_states = torch.cat(
                    [past_key_values["key_states"], key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values["value_states"], value_states], dim=1
                )

        # Eager Attention
        att_output = self._eager_attention(
            head_dim,
            query_states,
            key_states,
            value_states,
            _attention_mask,
        )
        return [att_output], past_key_values

    def _cross_attn_layer(
        self,
        model_layer: list[torch.nn.Module],
        inputs_embeds: list[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_dim: int,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        assert len(inputs_embeds) == 2 or (
            use_cache and past_key_values is not None and not fill_kv_cache
        )

        att_outputs = []
        # VLM
        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            prefix_position_ids, suffix_position_ids = (
                position_ids[:, :seq_len],
                position_ids[:, seq_len:],
            )
            _query_states, _key_states, value_states = self._qkv_proj_layer(
                model_layer[0], prefix_position_ids, inputs_embeds[0]
            )
            # Rotary Position Embedding
            value_states = value_states["t"]
            query_states = self.apply_rope(_query_states, prefix_position_ids)
            key_states = self.apply_rope(_key_states, prefix_position_ids)
            # Eager Attention
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]
            att_output = self._eager_attention(
                head_dim,
                query_states,
                key_states,
                value_states,
                prefix_attention_mask,
            )
            att_outputs.append(att_output)
        else:
            suffix_position_ids = position_ids

        if use_cache and past_key_values is not None:
            if fill_kv_cache:
                past_key_values = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:  # TODO: some optimization can be done, similar to `StaticCache`
                key_states = past_key_values["key_states"]
                value_states = past_key_values["value_states"]

        # Expert
        expert_layer = model_layer[1]
        if expert_layer is not None:
            # NOTE: key_states has been ROPEd before. Directly use it here.
            _query_states, _, _value_states = self._qkv_proj_layer(
                expert_layer,
                suffix_position_ids,
                inputs_embeds[1],
                None,
                value_states.view(*value_states.shape[:2], -1),
            )
            suffix_position_ids = (
                suffix_position_ids
                - torch.min(suffix_position_ids, dim=1, keepdim=True).values
            )  # start from 0
            # Rotary Position Embedding
            query_states = self.apply_rope(_query_states, suffix_position_ids)
            value_states = _value_states["t"]
            # Eager Attention
            suffix_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : key_states.shape[1] :
            ]
            att_output = self._eager_attention(
                head_dim,
                query_states,
                key_states,
                value_states,
                suffix_attention_mask,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def _get_model_layers(
        self, models: list[PreTrainedModel]
    ) -> list[list[torch.nn.Module]]:
        vlm_layers = []
        expert_layers = []
        for i in range(self.num_expert_skip_layers):
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(None)

        multiple_of = (
            self.num_vlm_layers - self.num_expert_skip_layers
        ) // self.num_expert_layers
        for i in range(self.num_expert_skip_layers, self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]

            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)

        assert len(vlm_layers) == len(expert_layers)
        return [(vlm_layers[i], expert_layers[i]) for i in range(len(vlm_layers))]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict[int, dict[str, torch.FloatTensor]] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ) -> tuple[list[torch.FloatTensor], list[torch.FloatTensor] | None]:
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self._get_model_layers(models)

        # Decoder Layers
        past_key_values = (
            {i: {} for i in range(self.num_vlm_layers)}
            if use_cache and past_key_values is None
            else past_key_values
        )
        for layer_idx in range(self.num_vlm_layers):
            attn_layer = None
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or layer_idx < self.num_expert_skip_layers
                or (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                )
            ):
                attn_layer = self._self_attn_layer
            else:
                attn_layer = self._cross_attn_layer

            start = 0
            outputs_embeds = []
            att_outputs, _past_key_values = attn_layer(
                model_layers[layer_idx],
                inputs_embeds,
                position_ids,
                attention_mask,
                self.vlm.config.text_config.head_dim,
                use_cache,
                fill_kv_cache,
                past_key_values[layer_idx] if past_key_values is not None else None,
            )
            if att_outputs is None:
                continue
            if past_key_values is not None:
                past_key_values[layer_idx] = _past_key_values

            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[layer_idx][i]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue

                    end = start + hidden_states.shape[1]
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    # Fully Connected Layer in Decoder Layer
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # Final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, seq_len, num_key_value_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states

        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, num_key_value_heads, n_rep, head_dim
        )
        return hidden_states.reshape(
            batch, seq_len, num_key_value_heads * n_rep, head_dim
        )

    def _eager_attention(
        self,
        head_dim: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        batch_size = attention_mask.size(0)
        key_states = self._repeat_kv(key, num_key_value_groups)
        value_states = self._repeat_kv(value, num_key_value_groups)

        # Attention here is upcasted to float32 to match the original implementation.
        att_weights = (head_dim**-0.5) * torch.matmul(
            query.transpose(1, 2).to(dtype=torch.float32),
            key_states.permute(0, 2, 3, 1).to(dtype=torch.float32),
        )

        att_weights = att_weights.to(dtype=torch.float32)
        masked_att_weights = torch.where(
            attention_mask[:, None, :, :],
            att_weights,
            torch.finfo(att_weights.dtype).min,
        )
        probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )
        return att_output
