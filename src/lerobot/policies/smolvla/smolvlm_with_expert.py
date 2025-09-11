# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import copy

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
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

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
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

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
