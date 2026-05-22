#!/usr/bin/env python

# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

# ruff: noqa

"""Inference utilities for MolmoAct2"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from collections.abc import Iterable, Sequence

import torch
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig


@dataclass
class _ActionFlowInputs:
    trajectory: torch.Tensor
    context: Any
    modulations: Sequence[Any]
    action_dim_is_pad: torch.Tensor | None


@dataclass
class _ActionFlowCudaGraph:
    key: tuple[Any, ...]
    graph: torch.cuda.CUDAGraph
    static_inputs: _ActionFlowInputs
    output: torch.Tensor


@dataclass
class _DepthDecodeCudaGraphLayerStage:
    residual: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class _DepthDecodeCudaGraphPostStage:
    graph: torch.cuda.CUDAGraph
    attn_context: torch.Tensor


@dataclass
class _DepthDecodeCudaGraph:
    cache_key: tuple[Any, ...]
    pre_graph: torch.cuda.CUDAGraph
    token_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    positions: torch.Tensor
    stages: Sequence[_DepthDecodeCudaGraphLayerStage]
    post_graphs: Sequence[_DepthDecodeCudaGraphPostStage]
    output: torch.Tensor


@dataclass
class _DepthDecodeCudaGraphSpec:
    eligible: bool
    cache_key_prefix: tuple[Any, ...]
    num_hidden_layers: int
    head_dim: int
    num_attention_heads: int


def _cache_seq_len_int(past_key_values: Cache | None) -> int:
    if past_key_values is None:
        return 0
    seq_len = past_key_values.get_seq_length()
    if torch.is_tensor(seq_len):
        return int(seq_len.item())
    return int(seq_len)


def _cache_max_len_int(past_key_values: Cache | None) -> int:
    if past_key_values is None:
        return -1
    max_len = past_key_values.get_max_cache_shape()
    if torch.is_tensor(max_len):
        return int(max_len.item())
    return int(max_len)


def _iter_cache_key_values(
    past_key_values: Cache,
) -> Iterable[tuple[torch.Tensor | None, torch.Tensor | None]]:
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None)
        return
    for layer in past_key_values:
        yield layer[0], layer[1]


class _DepthDecodeStaticLayerCache:
    is_compileable = False
    is_sliding = False

    def __init__(self, max_cache_len: int) -> None:
        self.max_cache_len = int(max_cache_len)
        self.cumulative_length = 0
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def _allocate(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        bsz, n_heads = key_states.shape[:2]
        self.keys = torch.empty(
            (bsz, n_heads, self.max_cache_len, key_states.shape[-1]),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.empty(
            (bsz, n_heads, self.max_cache_len, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self._allocate(key_states, value_states)
        start = self.cumulative_length
        end = start + key_states.shape[-2]
        if end > self.max_cache_len:
            raise RuntimeError(f"KV cache length {end} exceeds max_cache_len={self.max_cache_len}.")
        self.keys[:, :, start:end, :].copy_(key_states)
        self.values[:, :, start:end, :].copy_(value_states)
        self.cumulative_length = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self.cumulative_length = 0


class _DepthDecodeStaticCache(Cache):
    def __init__(self, config: PretrainedConfig, max_cache_len: int) -> None:
        text_config = config.get_text_config(decoder=True)
        super().__init__(
            layers=[
                _DepthDecodeStaticLayerCache(max_cache_len=max_cache_len)
                for _ in range(text_config.num_hidden_layers)
            ]
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


class ActionCudaGraphManager:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.enabled = True
        self.action_flow_graph: _ActionFlowCudaGraph | None = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def can_use_action_flow(self, inputs: _ActionFlowInputs) -> bool:
        action_model = self.model
        if not self.enabled:
            return False
        if action_model.training or action_model._require_action_expert().training:
            return False
        if inputs.trajectory.device.type != "cuda":
            return False

        def all_on_cuda():
            yield inputs.trajectory
            for k, v in inputs.context.kv_contexts:
                yield k
                yield v
            for t in (
                inputs.context.cross_mask,
                inputs.context.self_mask,
                inputs.context.valid_action,
                inputs.action_dim_is_pad,
            ):
                if t is not None:
                    yield t
            if inputs.context.rope_cache is not None:
                yield from inputs.context.rope_cache
            for step in inputs.modulations:
                yield step.conditioning
                for block_modulation in step.block_modulations:
                    yield from block_modulation
                yield from step.final_modulation

        return all(t.device.type == "cuda" for t in all_on_cuda())

    def run_action_flow(
        self,
        inputs: _ActionFlowInputs,
        steps: int,
        run_loop,
    ) -> torch.Tensor:
        key = _cuda_graph_key(inputs, steps)
        cache = self.action_flow_graph
        if cache is None or cache.key != key:
            static_inputs = _clone_static_inputs(inputs)
            graph, output = _capture_cuda_graph(
                lambda: run_loop(static_inputs, steps),
                inputs.trajectory.device,
                after_warmup=lambda: static_inputs.trajectory.copy_(inputs.trajectory),
            )
            cache = _ActionFlowCudaGraph(
                key=key,
                graph=graph,
                static_inputs=static_inputs,
                output=output,
            )
            self.action_flow_graph = cache
        else:
            _copy_inputs_(cache.static_inputs, inputs)

        cache.graph.replay()
        return cache.output.clone()


class DepthDecodeCudaGraphManager:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.backbone = model.model
        self.enabled = True
        self.graph: _DepthDecodeCudaGraph | None = None
        self.graph_spec: _DepthDecodeCudaGraphSpec | None = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def make_static_cache(self, max_cache_len: int) -> _DepthDecodeStaticCache:
        return _DepthDecodeStaticCache(
            config=self.model.config.text_config,
            max_cache_len=max_cache_len,
        )

    def _depth_decode_spec(self) -> _DepthDecodeCudaGraphSpec:
        static = self.graph_spec
        if static is None:
            cfg = self.backbone.transformer.config
            rotary_emb = getattr(self.backbone.transformer, "rotary_emb", None)
            static = _DepthDecodeCudaGraphSpec(
                eligible=(
                    not cfg.norm_after
                    and cfg.rope_scaling_layers is None
                    and getattr(rotary_emb, "rope_type", None) == "default"
                    and cfg._attn_implementation == "sdpa"
                ),
                cache_key_prefix=(
                    cfg.hidden_size,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.num_hidden_layers,
                    cfg.use_qk_norm,
                    cfg.qk_norm_type,
                    cfg._attn_implementation,
                ),
                num_hidden_layers=cfg.num_hidden_layers,
                head_dim=cfg.head_dim,
                num_attention_heads=cfg.num_attention_heads,
            )
            self.graph_spec = static
        return static

    def can_use(
        self,
        next_input_ids: torch.Tensor,
        *,
        past_key_values: Cache,
        attention_bias: torch.Tensor,
    ) -> bool:
        if not self.enabled or self.model.training or self.backbone.transformer.training:
            return False
        if next_input_ids.device.type != "cuda":
            return False
        if next_input_ids.ndim != 2 or next_input_ids.shape[0] != 1 or next_input_ids.shape[1] != 1:
            return False
        if not isinstance(past_key_values, _DepthDecodeStaticCache):
            return False
        if not torch.is_tensor(attention_bias) or attention_bias.device != next_input_ids.device:
            return False
        return self._depth_decode_spec().eligible

    def _depth_decode_key(
        self,
        next_input_ids: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[Any, ...]:
        device = next_input_ids.device
        return (
            self._depth_decode_spec().cache_key_prefix,
            device.type,
            device.index,
            self.model.lm_head.weight.dtype,
            attention_bias.shape[-1],
        )

    def _select_depth_decode_rope(self, cos: torch.Tensor, sin: torch.Tensor, *, past_length: int) -> None:
        emb = self.backbone.transformer.rotary_emb
        cos.copy_(emb._pos_cos_cache[0, :, past_length : past_length + 1, :])
        sin.copy_(emb._pos_sin_cache[0, :, past_length : past_length + 1, :])

    def _depth_decode_pre_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block = self.backbone.transformer.blocks[layer_idx]
        attention = block.self_attn
        residual = hidden_states
        hidden_states = block.attn_norm(hidden_states)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        qkv = attention.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(attention.fused_dims, dim=-1)
        value_states = value_states.view(hidden_shape)

        apply_qk_norm = attention.q_norm is not None and attention.k_norm is not None
        norm_after_view = apply_qk_norm and attention.qk_norm_type == "qwen3"

        if apply_qk_norm and not norm_after_view:
            query_states = attention.q_norm(query_states)
            key_states = attention.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)

        if norm_after_view:
            query_states = attention.q_norm(query_states)
            key_states = attention.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return residual, query_states, key_states, value_states

    def _depth_decode_pre0(
        self,
        token_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds = self.model._embed_base_tokens(token_ids)
        return self._depth_decode_pre_layer(0, inputs_embeds, cos, sin)

    def _depth_decode_post_layer(
        self,
        layer_idx: int,
        residual: torch.Tensor,
        attn_context: torch.Tensor,
    ) -> torch.Tensor:
        block = self.backbone.transformer.blocks[layer_idx]
        attention = block.self_attn
        input_shape = residual.shape[:-1]
        attn_output = attn_context.reshape(*input_shape, -1).contiguous()
        attn_output = attention.attn_out(attn_output)
        hidden_states = residual + block.dropout(attn_output)

        residual = hidden_states
        hidden_states = block.ff_norm(hidden_states)
        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + block.dropout(hidden_states)
        return hidden_states

    def _depth_decode_post_and_pre_next(
        self,
        layer_idx: int,
        residual: torch.Tensor,
        attn_context: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self._depth_decode_post_layer(layer_idx, residual, attn_context)
        return self._depth_decode_pre_layer(layer_idx + 1, hidden_states, cos, sin)

    def _depth_decode_last_post(
        self,
        layer_idx: int,
        residual: torch.Tensor,
        attn_context: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._depth_decode_post_layer(layer_idx, residual, attn_context)
        return self.backbone.transformer.ln_f(hidden_states)

    def _build_depth_decode_graph(
        self,
        next_input_ids: torch.Tensor,
        *,
        past_length: int,
        attention_bias: torch.Tensor,
    ) -> _DepthDecodeCudaGraph:
        text_config = self.backbone.transformer.config
        device = next_input_ids.device
        dtype = self.model.lm_head.weight.dtype
        static = self._depth_decode_spec()
        num_layers = static.num_hidden_layers
        head_dim = static.head_dim
        max_cache_len = int(attention_bias.shape[-1])
        max_rope_len = max(int(text_config.max_position_embeddings or 0), max_cache_len)
        self.backbone.transformer.prepare_rope_cache(device=device, max_seq_len=max_rope_len)

        token_ids = torch.empty((1, 1), device=device, dtype=torch.long)
        cos = torch.empty((1, 1, head_dim), device=device, dtype=dtype)
        sin = torch.empty_like(cos)
        positions = torch.arange(max_cache_len, device=device, dtype=torch.long)
        context_shape = (1, 1, static.num_attention_heads, head_dim)

        token_ids.copy_(next_input_ids)
        self._select_depth_decode_rope(cos, sin, past_length=past_length)

        pre_graph, pre_output = _capture_cuda_graph(
            lambda: self._depth_decode_pre0(token_ids, cos, sin),
            device,
        )
        stages = [_DepthDecodeCudaGraphLayerStage(*pre_output)]
        post_graphs = []
        for layer_idx in range(num_layers - 1):
            stage = stages[-1]
            attn_context = torch.empty(context_shape, device=device, dtype=dtype)
            graph, output = _capture_cuda_graph(
                lambda layer_idx=layer_idx, stage=stage, attn_context=attn_context: (
                    self._depth_decode_post_and_pre_next(
                        layer_idx,
                        stage.residual,
                        attn_context,
                        cos,
                        sin,
                    )
                ),
                device,
            )
            post_graphs.append(_DepthDecodeCudaGraphPostStage(graph=graph, attn_context=attn_context))
            stages.append(_DepthDecodeCudaGraphLayerStage(*output))

        last_stage = stages[-1]
        last_attn_context = torch.empty(context_shape, device=device, dtype=dtype)
        last_graph, last_output = _capture_cuda_graph(
            lambda: self._depth_decode_last_post(
                num_layers - 1,
                last_stage.residual,
                last_attn_context,
            ),
            device,
        )
        post_graphs.append(_DepthDecodeCudaGraphPostStage(graph=last_graph, attn_context=last_attn_context))
        return _DepthDecodeCudaGraph(
            cache_key=self._depth_decode_key(next_input_ids, attention_bias),
            pre_graph=pre_graph,
            token_ids=token_ids,
            cos=cos,
            sin=sin,
            positions=positions,
            stages=tuple(stages),
            post_graphs=tuple(post_graphs),
            output=last_output,
        )

    def _get_depth_decode_graph(
        self,
        next_input_ids: torch.Tensor,
        *,
        past_length: int,
        attention_bias: torch.Tensor,
    ) -> _DepthDecodeCudaGraph:
        key = self._depth_decode_key(next_input_ids, attention_bias)
        decode_graph = self.graph
        if decode_graph is None or decode_graph.cache_key != key:
            decode_graph = self._build_depth_decode_graph(
                next_input_ids,
                past_length=past_length,
                attention_bias=attention_bias,
            )
            self.graph = decode_graph
        else:
            decode_graph.token_ids.copy_(next_input_ids)
            self._select_depth_decode_rope(decode_graph.cos, decode_graph.sin, past_length=past_length)
        return decode_graph

    def _run_depth_decode_attention_core(
        self,
        layer_idx: int,
        stage: _DepthDecodeCudaGraphLayerStage,
        *,
        past_key_values: Cache,
        attention_bias: torch.Tensor,
        cache_position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.backbone.transformer.blocks[layer_idx].self_attn
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            stage.key,
            stage.value,
            layer_idx,
            cache_kwargs,
        )
        key_states = _repeat_kv(key_states, attention.num_key_value_groups)
        value_states = _repeat_kv(value_states, attention.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            stage.query,
            key_states,
            value_states,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        return attn_output.transpose(1, 2)

    def run(
        self,
        next_input_ids: torch.Tensor,
        *,
        past_key_values: Cache,
        attention_bias: torch.Tensor,
        past_length: int,
    ) -> tuple[torch.Tensor, Cache]:
        end = past_length + 1
        decode_graph = self._get_depth_decode_graph(
            next_input_ids,
            past_length=past_length,
            attention_bias=attention_bias,
        )
        cache_position = decode_graph.positions[past_length:end]
        attention_bias_q = attention_bias[:, :, past_length:end, :end]

        decode_graph.pre_graph.replay()

        for layer_idx, post_graph in enumerate(decode_graph.post_graphs):
            attn_context = self._run_depth_decode_attention_core(
                layer_idx,
                decode_graph.stages[layer_idx],
                past_key_values=past_key_values,
                attention_bias=attention_bias_q,
                cache_position=cache_position,
                cos=decode_graph.cos,
                sin=decode_graph.sin,
            )
            post_graph.attn_context.copy_(attn_context)
            post_graph.graph.replay()

        return decode_graph.output, past_key_values


def _cuda_graph_tensor_signature(
    tensor: torch.Tensor | None,
) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _cuda_graph_context_signature(context: Any) -> tuple[Any, ...]:
    sig = _cuda_graph_tensor_signature
    return (
        tuple((sig(k), sig(v)) for k, v in context.kv_contexts),
        sig(context.cross_mask),
        sig(context.self_mask),
        sig(context.valid_action),
        None if context.rope_cache is None else tuple(sig(t) for t in context.rope_cache),
    )


def _cuda_graph_modulation_signature(modulations: Sequence[Any]) -> tuple[Any, ...]:
    sig = _cuda_graph_tensor_signature
    return tuple(
        (
            sig(step.conditioning),
            tuple(tuple(sig(t) for t in block_modulation) for block_modulation in step.block_modulations),
            tuple(sig(t) for t in step.final_modulation),
        )
        for step in modulations
    )


def _cuda_graph_key(inputs: _ActionFlowInputs, steps: int) -> tuple[Any, ...]:
    sig = _cuda_graph_tensor_signature
    return (
        sig(inputs.trajectory),
        _cuda_graph_context_signature(inputs.context),
        _cuda_graph_modulation_signature(inputs.modulations),
        sig(inputs.action_dim_is_pad),
        int(steps),
    )


def _clone_static_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    static = torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    static.copy_(tensor)
    return static


def _clone_static_context(context: Any) -> Any:
    rope_cache = None
    if context.rope_cache is not None:
        rope_cache = tuple(_clone_static_tensor(t) for t in context.rope_cache)
    return context.__class__(
        kv_contexts=tuple((_clone_static_tensor(k), _clone_static_tensor(v)) for k, v in context.kv_contexts),
        cross_mask=_clone_static_tensor(context.cross_mask),
        self_mask=_clone_static_tensor(context.self_mask),
        valid_action=_clone_static_tensor(context.valid_action),
        rope_cache=rope_cache,
    )


def _clone_static_modulations(modulations: Sequence[Any]) -> Sequence[Any]:
    return tuple(
        step.__class__(
            conditioning=_clone_static_tensor(step.conditioning),
            block_modulations=tuple(
                tuple(_clone_static_tensor(t) for t in block_modulation)
                for block_modulation in step.block_modulations
            ),
            final_modulation=tuple(_clone_static_tensor(t) for t in step.final_modulation),
        )
        for step in modulations
    )


def _clone_static_inputs(inputs: _ActionFlowInputs) -> _ActionFlowInputs:
    return _ActionFlowInputs(
        trajectory=_clone_static_tensor(inputs.trajectory),
        context=_clone_static_context(inputs.context),
        modulations=_clone_static_modulations(inputs.modulations),
        action_dim_is_pad=_clone_static_tensor(inputs.action_dim_is_pad),
    )


def _copy_context_(dst: Any, src: Any) -> None:
    for (dst_k, dst_v), (src_k, src_v) in zip(dst.kv_contexts, src.kv_contexts):
        dst_k.copy_(src_k)
        dst_v.copy_(src_v)
    if src.cross_mask is not None:
        dst.cross_mask.copy_(src.cross_mask)
    if src.self_mask is not None:
        dst.self_mask.copy_(src.self_mask)
    if src.valid_action is not None:
        dst.valid_action.copy_(src.valid_action)
    if src.rope_cache is not None:
        for dst_tensor, src_tensor in zip(dst.rope_cache, src.rope_cache):
            dst_tensor.copy_(src_tensor)


def _copy_inputs_(dst: _ActionFlowInputs, src: _ActionFlowInputs) -> None:
    dst.trajectory.copy_(src.trajectory)
    _copy_context_(dst.context, src.context)
    if src.action_dim_is_pad is not None:
        dst.action_dim_is_pad.copy_(src.action_dim_is_pad)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _capture_cuda_graph(
    fn,
    device: torch.device,
    *,
    after_warmup=None,
) -> tuple[torch.cuda.CUDAGraph, Any]:
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        fn()
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    if after_warmup is not None:
        after_warmup()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output
