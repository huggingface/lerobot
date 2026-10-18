# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/projections.py
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Deployment helpers extracted from the original model without changing weights."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask


@torch.library.custom_op("lerobot_longwam::cudnn_attention", mutates_args=())
def cudnn_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Force cuDNN with a stable sequence-major output layout for compilation."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return output.transpose(1, 2).contiguous().transpose(1, 2)


@cudnn_attention.register_fake
def _cudnn_attention_fake(q, k, v, mask):
    batch, heads, length, dim = q.shape
    return q.new_empty((batch, length, heads, dim)).transpose(1, 2)


class InferenceProjectionMixin:
    mixtures: nn.ModuleDict
    training: bool
    num_layers: int
    num_heads: int
    attn_head_dim: int
    expert_rope_compute_dtypes: dict[str, torch.dtype | None]
    expert_sdpa_backends: dict[str, str | None]
    expert_sdpa_block_masks: dict[str, BlockMask | None]
    expert_shared_self_qkv_quantization: dict[str, bool]
    expert_shared_cross_kv_quantization: dict[str, bool]
    expert_concatenated_self_qkv_gemm: dict[str, bool]
    expert_batched_self_qkv_gemm: dict[str, bool]

    if TYPE_CHECKING:
        # Implemented by the MoT host; declarations have no runtime effect.
        def _build_expert_attention_io(
            self,
            expert: nn.Module,
            block: nn.Module,
            x: torch.Tensor,
            freqs: torch.Tensor,
            t_mod: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            bool,
        ]: ...

        def _mixed_attention(
            self,
            q_cat: torch.Tensor,
            k_cat: torch.Tensor,
            v_cat: torch.Tensor,
            attention_mask: torch.Tensor,
            sdpa_backend: str | None = None,
            sdpa_block_mask: BlockMask | None = None,
        ) -> torch.Tensor: ...

        def _apply_post_with_optional_checkpoint(
            self,
            block: nn.Module,
            residual_x: torch.Tensor,
            gate_msa: torch.Tensor,
            shift_mlp: torch.Tensor,
            scale_mlp: torch.Tensor,
            gate_mlp: torch.Tensor,
            use_gradient_checkpointing: bool,
            mixed_slice: torch.Tensor,
            context_payload: dict | None,
        ) -> torch.Tensor: ...

    def configure_expert_rope_compute_dtype(
        self,
        expert_name: str,
        dtype: torch.dtype | None,
        *,
        direct_output: bool = False,
    ) -> None:
        """Select one expert's RoPE arithmetic without changing its peer."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError(f"Unsupported RoPE compute dtype: {dtype}")
        if direct_output and dtype != torch.float32:
            raise ValueError("Direct RoPE output requires float32 arithmetic.")
        self.expert_rope_compute_dtypes[expert_name] = dtype
        expert = self.mixtures[expert_name]
        for block in expert.blocks:
            block.self_attn.rope_compute_dtype = dtype
            block.self_attn.rope_direct_output = direct_output

    def configure_expert_self_sdpa_backend(
        self,
        expert_name: str,
        backend: str | None,
        *,
        block_mask: BlockMask | None = None,
    ) -> None:
        """Select self/mixed SDPA for one expert; keep cross-attention unchanged."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if backend not in (None, "cudnn", "flex-group-causal"):
            raise ValueError(f"Unsupported SDPA backend: {backend!r}")
        if backend == "flex-group-causal" and block_mask is None:
            raise ValueError("flex-group-causal SDPA requires a BlockMask")
        if backend != "flex-group-causal" and block_mask is not None:
            raise ValueError("BlockMask is only valid for flex-group-causal SDPA")
        self.expert_sdpa_backends[expert_name] = backend
        self.expert_sdpa_block_masks[expert_name] = block_mask
        for block in self.mixtures[expert_name].blocks:
            block.self_attn.sdpa_backend = backend
            block.self_attn.sdpa_block_mask = block_mask

    def configure_expert_packed_qkv(self, expert_name: str) -> int:
        """Prepack one inference expert's self-attention Q/K/V projections."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Packed QKV is inference-only")
        packed_layers = 0
        for block in self.mixtures[expert_name].blocks:
            attention = block.self_attn
            projections = (attention.q, attention.k, attention.v)
            if not all(isinstance(projection, nn.Linear) for projection in projections):
                raise TypeError("Packed QKV requires three torch.nn.Linear projections")
            if any(projection.bias is None for projection in projections):
                raise ValueError("Packed QKV requires Q/K/V biases")
            weight = torch.cat([projection.weight.detach() for projection in projections], dim=0).contiguous()
            bias = torch.cat([projection.bias.detach() for projection in projections], dim=0).contiguous()
            attention.register_buffer("_edge_packed_qkv_weight", weight, persistent=False)
            attention.register_buffer("_edge_packed_qkv_bias", bias, persistent=False)
            width = attention.attn_hidden_dim
            for index, projection in enumerate(projections):
                projection.weight = nn.Parameter(
                    weight[index * width : (index + 1) * width],
                    requires_grad=False,
                )
                projection.bias = nn.Parameter(
                    bias[index * width : (index + 1) * width],
                    requires_grad=False,
                )
            packed_layers += 1
        return packed_layers

    def configure_expert_packed_cross_kv(self, expert_name: str) -> int:
        """Prepack one inference expert's BF16 cross-attention K/V weights."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Packed cross K/V is inference-only")
        packed_layers = 0
        for block in self.mixtures[expert_name].blocks:
            attention = block.cross_attn
            projections = (attention.k, attention.v)
            if not all(isinstance(projection, nn.Linear) for projection in projections):
                raise TypeError("Packed cross K/V requires two torch.nn.Linear projections")
            if any(projection.bias is None for projection in projections):
                raise ValueError("Packed cross K/V requires K/V biases")
            weight = torch.cat([projection.weight.detach() for projection in projections], dim=0).contiguous()
            bias = torch.cat([projection.bias.detach() for projection in projections], dim=0).contiguous()
            attention.register_buffer("_edge_packed_cross_kv_weight", weight, persistent=False)
            attention.register_buffer("_edge_packed_cross_kv_bias", bias, persistent=False)
            width = attention.attn_hidden_dim
            for index, projection in enumerate(projections):
                projection.weight = nn.Parameter(
                    weight[index * width : (index + 1) * width],
                    requires_grad=False,
                )
                projection.bias = nn.Parameter(
                    bias[index * width : (index + 1) * width],
                    requires_grad=False,
                )
            packed_layers += 1
        return packed_layers

    def configure_expert_shared_self_qkv_quantization(
        self,
        expert_name: str,
    ) -> int:
        """Reuse one quantized activation across self-attention Q/K/V."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Shared self QKV quantization is inference-only")

        configured_layers = 0
        for block in self.mixtures[expert_name].blocks:
            projections = (
                block.self_attn.q,
                block.self_attn.k,
                block.self_attn.v,
            )
            activation_configs = []
            for projection in projections:
                if not callable(getattr(projection, "quantized_weight", None)):
                    raise TypeError("Shared self QKV quantization requires quantized Q/K/V projections")
                config = getattr(projection, "config", None)
                if config is None:
                    raise TypeError("Shared self QKV quantization requires projection config")
                activation_configs.append(config.get_activation_config())
            if any(config != activation_configs[0] for config in activation_configs[1:]):
                raise ValueError("Self Q/K/V projections must use one activation quantization configuration")
            block.self_attn._edge_shared_qkv_quantization = True
            configured_layers += 1

        self.expert_shared_self_qkv_quantization[expert_name] = True
        return configured_layers

    def configure_expert_shared_cross_kv_quantization(
        self,
        expert_name: str,
    ) -> int:
        """Reuse one quantized context across one expert's cross K/V projections."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Shared cross K/V quantization is inference-only")

        reference_config = None
        configured_layers = 0
        for block in self.mixtures[expert_name].blocks:
            for projection in (block.cross_attn.k, block.cross_attn.v):
                if not callable(getattr(projection, "quantized_weight", None)):
                    raise TypeError("Shared cross K/V quantization requires quantized K/V projections")
                config = getattr(projection, "config", None)
                if config is None:
                    raise TypeError("Shared cross K/V quantization requires projection config")
                activation_config = config.get_activation_config()
                if reference_config is None:
                    reference_config = activation_config
                elif activation_config != reference_config:
                    raise ValueError(
                        "Cross K/V projections must use one activation quantization configuration"
                    )
            configured_layers += 1

        self.expert_shared_cross_kv_quantization[expert_name] = True
        return configured_layers

    def configure_expert_direct_shared_self_qkv_quantization(
        self,
        expert_name: str,
    ) -> int:
        """Reuse one Q/K/V activation quantization in direct expert forwards."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Direct shared self QKV quantization is inference-only")
        if not self.expert_shared_self_qkv_quantization.get(expert_name, False):
            raise RuntimeError("Direct shared self QKV quantization requires shared QKV quantization")
        configured_layers = 0
        for block in self.mixtures[expert_name].blocks:
            block.self_attn._edge_direct_shared_qkv_quantization = True
            configured_layers += 1
        return configured_layers

    def configure_expert_nvfp4_bias_epilogue(self, expert_name: str) -> int:
        """Fuse bias into every quantized linear GEMM for one expert."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("NVFP4 bias epilogue is inference-only")

        configured_modules = 0
        for module_name, module in self.mixtures[expert_name].named_modules():
            if not callable(getattr(module, "quantized_weight", None)):
                continue
            enable = getattr(module, "enable_nvfp4_bias_epilogue", None)
            if not callable(enable):
                raise TypeError(
                    "NVFP4 bias epilogue requires the patched FourOverSix "
                    f"extension; module {module_name!r} has no enable method"
                )
            if getattr(module, "bias", None) is None:
                raise ValueError(f"NVFP4 bias epilogue requires bias on module {module_name!r}")
            enable()
            configured_modules += 1
        if configured_modules == 0:
            raise RuntimeError(f"No quantized linear modules found for expert {expert_name!r}")
        return configured_modules

    def configure_expert_concatenated_self_qkv_gemm(
        self,
        expert_name: str,
    ) -> int:
        """Pack three quantized self Q/K/V weights for one per-column GEMM."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Concatenated self QKV GEMM is inference-only")
        if not self.expert_shared_self_qkv_quantization.get(expert_name, False):
            raise RuntimeError("Concatenated self QKV GEMM requires shared QKV quantization")

        from fouroversix.utils import DataType

        configured_layers = 0
        for block in self.mixtures[expert_name].blocks:
            attention = block.self_attn
            projections = (attention.q, attention.k, attention.v)
            weights = tuple(projection.quantized_weight() for projection in projections)
            reference = weights[0]
            if any(weight.dtype != DataType.nvfp4 for weight in weights):
                raise TypeError("Concatenated self QKV GEMM requires NVFP4 weights")
            if any(
                weight.original_shape != reference.original_shape
                or weight.padded_shape != reference.padded_shape
                or weight.scale_rule != reference.scale_rule
                for weight in weights[1:]
            ):
                raise ValueError(
                    "Concatenated self QKV GEMM requires identical weight shapes and scale rules"
                )
            if any(projection.bias is None for projection in projections):
                raise ValueError("Concatenated self QKV GEMM requires Q/K/V biases")

            width = reference.original_shape[0]
            values = torch.cat([weight.values for weight in weights], dim=0).contiguous()
            scales = torch.cat([weight.scale_factors.reshape(-1) for weight in weights], dim=0).contiguous()
            weight_amax = torch.cat(
                [weight.amax.reshape(1).expand(width) for weight in weights], dim=0
            ).contiguous()
            output_bias = torch.cat(
                [projection.bias.detach() for projection in projections], dim=0
            ).contiguous()
            epilogue_zero = torch.zeros(
                3 * width,
                device=values.device,
                dtype=torch.float32,
            )
            attention.register_buffer("_edge_concat_qkv_values", values, persistent=False)
            attention.register_buffer("_edge_concat_qkv_scales", scales, persistent=False)
            attention.register_buffer("_edge_concat_qkv_weight_amax", weight_amax, persistent=False)
            attention.register_buffer("_edge_concat_qkv_output_bias", output_bias, persistent=False)
            attention.register_buffer("_edge_concat_qkv_epilogue_zero", epilogue_zero, persistent=False)
            attention._edge_concat_qkv_width = width
            attention._edge_concat_qkv_weight_scale_denominator = (
                reference.scale_rule.max_allowed_e2m1_value() * reference.scale_rule.max_allowed_e4m3_value()
            )
            attention._edge_concat_qkv_gemm = True
            configured_layers += 1

        self.expert_concatenated_self_qkv_gemm[expert_name] = True
        return configured_layers

    def configure_expert_batched_self_qkv_gemm(
        self,
        expert_name: str,
    ) -> int:
        """Stack separately quantized self Q/K/V for one fixed-batch GEMM."""
        if expert_name not in self.mixtures:
            raise ValueError(f"Unknown MoT expert: {expert_name!r}")
        if self.training:
            raise RuntimeError("Batched self QKV GEMM is inference-only")
        if not self.expert_shared_self_qkv_quantization.get(expert_name, False):
            raise RuntimeError("Batched self QKV GEMM requires shared QKV quantization")
        batched_op = getattr(
            getattr(torch.ops, "fouroversix", None),
            "qkv_gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt",
            None,
        )
        if batched_op is None:
            raise RuntimeError("Batched self QKV GEMM requires the Thor FourOverSix extension")

        from fouroversix.utils import DataType

        configured_layers = 0
        for block in self.mixtures[expert_name].blocks:
            attention = block.self_attn
            projections = (attention.q, attention.k, attention.v)
            weights = tuple(projection.quantized_weight() for projection in projections)
            reference = weights[0]
            if any(weight.dtype != DataType.nvfp4 for weight in weights):
                raise TypeError("Batched self QKV GEMM requires NVFP4 weights")
            if any(
                weight.original_shape != reference.original_shape
                or weight.padded_shape != reference.padded_shape
                or weight.scale_rule != reference.scale_rule
                for weight in weights[1:]
            ):
                raise ValueError("Batched self QKV GEMM requires identical weight shapes and scale rules")
            if any(projection.bias is None for projection in projections):
                raise ValueError("Batched self QKV GEMM requires Q/K/V biases")

            tensors = {
                "_edge_batched_qkv_values": torch.stack(
                    [weight.values.detach() for weight in weights]
                ).contiguous(),
                "_edge_batched_qkv_scales": torch.stack(
                    [weight.scale_factors.detach() for weight in weights]
                ).contiguous(),
                "_edge_batched_qkv_weight_amax": torch.stack([weight.amax.detach() for weight in weights])
                .reshape(-1)
                .contiguous(),
                "_edge_batched_qkv_output_bias": torch.stack(
                    [projection.bias.detach() for projection in projections]
                ).contiguous(),
            }
            for name, tensor in tensors.items():
                attention.register_buffer(name, tensor, persistent=False)
            for index, projection in enumerate(projections):
                projection.quantized_weight_values = tensors["_edge_batched_qkv_values"][index]
                projection.quantized_weight_scale_factors = tensors["_edge_batched_qkv_scales"][index]
                projection.quantized_weight_amax = tensors["_edge_batched_qkv_weight_amax"][index : index + 1]
                projection.bias = nn.Parameter(
                    tensors["_edge_batched_qkv_output_bias"][index],
                    requires_grad=False,
                )
                if hasattr(projection, "_quantized_weight"):
                    delattr(projection, "_quantized_weight")
                projection.quantized_weight()
            attention._edge_batched_qkv_gemm = True
            configured_layers += 1

        self.expert_batched_self_qkv_gemm[expert_name] = True
        return configured_layers

    def set_kv_cache_quantizer(self, quantizer) -> None:
        """Install an inference-only video KV cache codec."""
        if self.training and quantizer is not None:
            raise RuntimeError("KV cache quantization can only be enabled for eval().")
        self.kv_cache_quantizer = quantizer

    @staticmethod
    def _project_shared_quantized_linear(
        quantized_input: Any,
        input_shape: torch.Size,
        projection: nn.Module,
    ) -> torch.Tensor:
        config = projection.config
        weight = projection.quantized_weight()
        if getattr(projection, "_edge_nvfp4_bias_epilogue", False):
            from fouroversix.matmul.cutlass.ops import (
                gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_bias,
            )
            from fouroversix.utils import DataType, MatmulBackend

            if (
                weight.dtype != DataType.nvfp4
                or config.output_dtype != DataType.bfloat16
                or config.matmul_backend != MatmulBackend.cutlass
                or projection.bias is None
            ):
                raise RuntimeError("NVFP4 bias epilogue requires NVFP4/BF16 CUTLASS with bias")
            activation_config = config.get_activation_config()
            denominator = (
                activation_config.scale_rule.max_allowed_e2m1_value()
                * activation_config.scale_rule.max_allowed_e4m3_value()
                * weight.scale_rule.max_allowed_e2m1_value()
                * weight.scale_rule.max_allowed_e4m3_value()
            )
            alpha = (quantized_input.amax * weight.amax / denominator).to(torch.float32)
            out = gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_bias(
                quantized_input.values,
                weight.values,
                quantized_input.scale_factors,
                weight.scale_factors,
                alpha,
                projection.bias,
            )
            rows = quantized_input.original_shape[0]
            out = out[:rows, : weight.original_shape[0]].reshape(*input_shape[:-1], weight.original_shape[0])
        else:
            from fouroversix.matmul import fp4_matmul

            out = fp4_matmul(
                quantized_input,
                weight,
                backend=config.matmul_backend,
                out_dtype=config.output_dtype,
            ).reshape(*input_shape[:-1], weight.original_shape[0])
            if projection.bias is not None:
                out = out + projection.bias
        return out

    def prefill_video_cross_kv_cache(
        self,
        context: torch.Tensor,
    ) -> dict[str, Any]:
        """Project invariant video context and per-layer cross-attention K/V."""
        expert = self.mixtures["video"]
        projected_context = expert.text_embedding(context)
        if self.expert_shared_cross_kv_quantization["video"]:
            if projected_context.numel() == 0:
                layers = [
                    {
                        "k": projected_context.new_empty(
                            *projected_context.shape[:-1],
                            block.cross_attn.k.out_features,
                        ),
                        "v": projected_context.new_empty(
                            *projected_context.shape[:-1],
                            block.cross_attn.v.out_features,
                        ),
                    }
                    for block in expert.blocks
                ]
            else:
                from fouroversix.quantize import quantize_to_fp4

                activation_config = expert.blocks[0].cross_attn.k.config.get_activation_config()
                quantized_context = quantize_to_fp4(
                    projected_context.reshape(-1, projected_context.shape[-1]),
                    activation_config,
                )
                layers = []
                for block in expert.blocks:
                    attention = block.cross_attn
                    k = self._project_shared_quantized_linear(
                        quantized_context,
                        projected_context.shape,
                        attention.k,
                    )
                    v = self._project_shared_quantized_linear(
                        quantized_context,
                        projected_context.shape,
                        attention.v,
                    )
                    layers.append({"k": attention.norm_k(k), "v": v})
        else:
            layers = [
                {
                    "k": block.cross_attn.norm_k(block.cross_attn.k(projected_context)),
                    "v": block.cross_attn.v(projected_context),
                }
                for block in expert.blocks
            ]
        return {
            "context": projected_context,
            "layers": layers,
        }

    @torch.no_grad()
    def install_video_cross_kv_cache(
        self,
        cache: dict[str, Any],
    ) -> torch.Tensor:
        """Copy video cross K/V into a fixed-address module-owned bank.

        The compiled prefill CUDA Graph owns its output addresses.  Rebinding those
        outputs on every plan both shortens their lifetime and changes the module
        buffer addresses seen by the downstream video CUDA Graphs.  A resident bank
        preserves the cache ABI and downstream capture across plans.
        """
        expert = self.mixtures["video"]
        if set(cache) != {"context", "layers"}:
            raise ValueError("Video cross K/V cache must contain context and layers")
        layers = cache["layers"]
        if len(layers) != len(expert.blocks):
            raise ValueError(f"Video cross K/V cache has {len(layers)} layers; expected {len(expert.blocks)}")

        copy_calls = 0
        copy_bytes = 0

        def install_tensor(
            owner: torch.nn.Module,
            name: str,
            source: torch.Tensor,
            *,
            label: str,
        ) -> torch.Tensor:
            nonlocal copy_calls, copy_bytes
            if not torch.is_tensor(source) or source.layout != torch.strided:
                raise TypeError(f"{label} must be a strided tensor")
            destination = owner._buffers.get(name)
            if destination is None:
                destination = torch.empty_like(
                    source,
                    memory_format=torch.preserve_format,
                )
                owner.register_buffer(name, destination, persistent=False)
            source_abi = (
                tuple(source.shape),
                tuple(source.stride()),
                source.dtype,
                source.device,
                source.layout,
            )
            destination_abi = (
                tuple(destination.shape),
                tuple(destination.stride()),
                destination.dtype,
                destination.device,
                destination.layout,
            )
            if destination_abi != source_abi:
                raise RuntimeError(
                    f"Resident {label} ABI changed: source={source_abi}, destination={destination_abi}"
                )
            if destination is not source:
                destination.copy_(source)
                copy_calls += 1
                copy_bytes += int(source.numel() * source.element_size())
            return destination

        resident_context = install_tensor(
            expert,
            "_edge_video_cross_context",
            cache["context"],
            label="video cross context",
        )
        for layer_index, (block, layer_cache) in enumerate(zip(expert.blocks, layers, strict=False)):
            if set(layer_cache) != {"k", "v"}:
                raise ValueError(f"Video cross K/V cache layer {layer_index} must contain exactly k and v")
            attention = block.cross_attn
            for name, key in (("_edge_cached_k", "k"), ("_edge_cached_v", "v")):
                install_tensor(
                    attention,
                    name,
                    layer_cache[key],
                    label=f"video cross layer {layer_index} {key}",
                )

        self._edge_video_cross_kv_install_calls = (
            int(getattr(self, "_edge_video_cross_kv_install_calls", 0)) + 1
        )
        self._edge_video_cross_kv_copy_calls = (
            int(getattr(self, "_edge_video_cross_kv_copy_calls", 0)) + copy_calls
        )
        self._edge_video_cross_kv_copy_bytes = (
            int(getattr(self, "_edge_video_cross_kv_copy_bytes", 0)) + copy_bytes
        )
        return resident_context

    def video_cross_kv_cache(self) -> dict[str, Any]:
        """Return the fixed-address video cross-attention cache tree."""
        expert = self.mixtures["video"]
        context = expert._buffers.get("_edge_video_cross_context")
        if context is None:
            raise RuntimeError("Resident video cross context is not installed")
        layers = []
        for layer_index, block in enumerate(expert.blocks):
            k = block.cross_attn._buffers.get("_edge_cached_k")
            v = block.cross_attn._buffers.get("_edge_cached_v")
            if k is None or v is None:
                raise RuntimeError(f"Resident video cross K/V is not installed for layer {layer_index}")
            layers.append({"k": k, "v": v})
        return {"context": context, "layers": layers}

    def prefill_action_cross_kv_cache(
        self,
        context: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Project invariant action cross-attention K/V once per inference."""
        expert = self.mixtures["action"]
        projected_context = expert.text_embedding(context)
        cache = []
        for block in expert.blocks:
            attention = block.cross_attn
            packed_weight = getattr(attention, "_edge_packed_cross_kv_weight", None)
            packed_bias = getattr(attention, "_edge_packed_cross_kv_bias", None)
            if packed_weight is None or packed_bias is None:
                k = attention.k(projected_context)
                v = attention.v(projected_context)
            else:
                packed_kv = F.linear(
                    projected_context,
                    packed_weight,
                    packed_bias,
                )
                k, v = packed_kv.chunk(2, dim=-1)
                k = k.contiguous()
                v = v.contiguous()
            cache.append({"k": attention.norm_k(k), "v": v})
        return cache

    @torch.no_grad()
    def install_action_cross_kv_cache(
        self,
        cache: list[dict[str, torch.Tensor]],
    ) -> int:
        """Copy action cross K/V into a fixed-address module-owned bank.

        The action prefill may itself be a CUDA Graph.  Binding its graph-owned
        outputs directly makes the downstream action graph depend on addresses
        that are overwritten by a later replay.  Stable module buffers preserve
        both output lifetime and the action-denoiser capture ABI.
        """
        expert = self.mixtures["action"]
        if len(cache) != len(expert.blocks):
            raise ValueError(f"Action cross K/V cache has {len(cache)} layers; expected {len(expert.blocks)}")

        copy_calls = 0
        copy_bytes = 0
        for layer_index, (block, layer_cache) in enumerate(zip(expert.blocks, cache, strict=False)):
            if set(layer_cache) != {"k", "v"}:
                raise ValueError(f"Action cross K/V cache layer {layer_index} must contain exactly k and v")
            attention = block.cross_attn
            for name, key in (("_edge_cached_k", "k"), ("_edge_cached_v", "v")):
                source = layer_cache[key]
                if not torch.is_tensor(source) or source.layout != torch.strided:
                    raise TypeError(f"action cross layer {layer_index} {key} must be a strided tensor")
                destination = attention._buffers.get(name)
                if destination is None:
                    destination = torch.empty_like(
                        source,
                        memory_format=torch.preserve_format,
                    )
                    attention.register_buffer(name, destination, persistent=False)
                source_abi = (
                    tuple(source.shape),
                    tuple(source.stride()),
                    source.dtype,
                    source.device,
                    source.layout,
                )
                destination_abi = (
                    tuple(destination.shape),
                    tuple(destination.stride()),
                    destination.dtype,
                    destination.device,
                    destination.layout,
                )
                if destination_abi != source_abi:
                    raise RuntimeError(
                        f"Resident action cross layer {layer_index} {key} ABI changed: "
                        f"source={source_abi}, destination={destination_abi}"
                    )
                if destination is not source:
                    destination.copy_(source)
                    copy_calls += 1
                    copy_bytes += int(source.numel() * source.element_size())

        self._edge_action_cross_kv_install_calls = (
            int(getattr(self, "_edge_action_cross_kv_install_calls", 0)) + 1
        )
        self._edge_action_cross_kv_copy_calls = (
            int(getattr(self, "_edge_action_cross_kv_copy_calls", 0)) + copy_calls
        )
        self._edge_action_cross_kv_copy_bytes = (
            int(getattr(self, "_edge_action_cross_kv_copy_bytes", 0)) + copy_bytes
        )
        return len(cache)

    def action_cross_kv_cache(self) -> list[dict[str, torch.Tensor]]:
        """Return the fixed-address action cross-attention cache tree."""
        expert = self.mixtures["action"]
        layers = []
        for layer_index, block in enumerate(expert.blocks):
            k = block.cross_attn._buffers.get("_edge_cached_k")
            v = block.cross_attn._buffers.get("_edge_cached_v")
            if k is None or v is None:
                raise RuntimeError(f"Resident action cross K/V is not installed for layer {layer_index}")
            layers.append({"k": k, "v": v})
        return layers

    @torch.no_grad()
    def install_action_video_kv_cache(
        self,
        cache: list[dict[str, torch.Tensor]],
    ) -> int:
        """Copy final-video K/V into stable buffers consumed by action graphs.

        The video-prefill CUDA Graph owns its output addresses and may overwrite
        them on a later replay.  These module buffers provide a distinct,
        fixed-address lifetime while requiring only one update per world plan.
        """
        expert = self.mixtures["action"]
        if self.kv_cache_quantizer is not None:
            raise RuntimeError("Resident action video K/V currently supports only BF16 cache entries")
        if len(cache) != len(expert.blocks):
            raise ValueError(f"Video K/V cache has {len(cache)} layers; expected {len(expert.blocks)}")

        copy_calls = 0
        copy_bytes = 0
        for layer_index, (block, layer_cache) in enumerate(zip(expert.blocks, cache, strict=False)):
            if set(layer_cache) != {"k", "v"}:
                raise ValueError(f"Video K/V cache layer {layer_index} must contain exactly k and v")
            attention = block.self_attn
            for name, key in (
                ("_edge_action_video_k", "k"),
                ("_edge_action_video_v", "v"),
            ):
                source = layer_cache[key]
                if not torch.is_tensor(source) or source.layout != torch.strided:
                    raise TypeError(f"Video K/V cache layer {layer_index} {key} must be a strided tensor")
                destination = attention._buffers.get(name)
                if destination is None:
                    destination = torch.empty_like(
                        source,
                        memory_format=torch.preserve_format,
                    )
                    attention.register_buffer(name, destination, persistent=False)
                source_abi = (
                    tuple(source.shape),
                    tuple(source.stride()),
                    source.dtype,
                    source.device,
                    source.layout,
                )
                destination_abi = (
                    tuple(destination.shape),
                    tuple(destination.stride()),
                    destination.dtype,
                    destination.device,
                    destination.layout,
                )
                if destination_abi != source_abi:
                    raise RuntimeError(
                        f"Resident video K/V ABI changed at layer {layer_index} {key}: "
                        f"source={source_abi}, destination={destination_abi}"
                    )
                destination.copy_(source)
                copy_calls += 1
                copy_bytes += int(source.numel() * source.element_size())

        self._edge_action_video_kv_install_calls = (
            int(getattr(self, "_edge_action_video_kv_install_calls", 0)) + 1
        )
        self._edge_action_video_kv_copy_calls = (
            int(getattr(self, "_edge_action_video_kv_copy_calls", 0)) + copy_calls
        )
        self._edge_action_video_kv_copy_bytes = (
            int(getattr(self, "_edge_action_video_kv_copy_bytes", 0)) + copy_bytes
        )
        return len(cache)

    def action_video_kv_cache(self) -> list[dict[str, torch.Tensor]]:
        """Return the fixed-address K/V tree installed for action denoising."""
        expert = self.mixtures["action"]
        cache: list[dict[str, torch.Tensor]] = []
        for layer_index, block in enumerate(expert.blocks):
            attention = block.self_attn
            k = attention._buffers.get("_edge_action_video_k")
            v = attention._buffers.get("_edge_action_video_v")
            if k is None or v is None:
                raise RuntimeError(f"Resident action video K/V is not installed for layer {layer_index}")
            cache.append({"k": k, "v": v})
        return cache

    def owns_action_video_kv_cache(
        self,
        cache: list[dict[str, torch.Tensor]],
    ) -> bool:
        """Return whether a cache tree is the module-owned action bank."""
        try:
            resident = self.action_video_kv_cache()
        except RuntimeError:
            return False
        return len(cache) == len(resident) and all(
            set(source) == {"k", "v"} and source["k"] is destination["k"] and source["v"] is destination["v"]
            for source, destination in zip(cache, resident, strict=False)
        )

    def _prefill_video_cache_impl(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
        sdpa_block_mask: BlockMask | None,
    ) -> list[Any]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE real/imag pairs, shape [Sv, 1, rope_dim, 2].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[Any] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
                sdpa_backend=self.expert_sdpa_backends["video"],
                sdpa_block_mask=sdpa_block_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            if self.kv_cache_quantizer is None:
                kv_cache.append({"k": k, "v": v})
            else:
                kv_cache.append(
                    self.kv_cache_quantizer.quantize(
                        k,
                        v,
                        num_heads=self.num_heads,
                        head_dim=self.attn_head_dim,
                    )
                )
        return kv_cache

    def prefill_video_clean_prefix_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> list[Any]:
        """Evaluate the causally closed observation prefix once."""
        if self.kv_cache_quantizer is not None:
            raise RuntimeError("clean-prefix wavefront currently requires BF16 video KV storage")
        block_mask = self.expert_sdpa_block_masks["video"]
        if self.expert_sdpa_backends["video"] == "flex-group-causal":
            block_mask = getattr(self, "_video_clean_prefix_sdpa_block_mask", None)
            if block_mask is None:
                raise RuntimeError("clean-prefix FlexAttention mask is not installed")
        return self._prefill_video_cache_impl(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            sdpa_block_mask=block_mask,
        )

    @torch.no_grad()
    def install_video_clean_prefix_cache(
        self,
        cache: list[dict[str, torch.Tensor]],
        *,
        total_sequence_length: int,
    ) -> int:
        """Copy clean K/V once into stable full-sequence retained arenas."""
        if len(cache) != self.num_layers:
            raise ValueError(f"clean-prefix cache must contain {self.num_layers} layers, got {len(cache)}")
        total_sequence_length = int(total_sequence_length)
        if total_sequence_length <= 0:
            raise ValueError("total video sequence length must be positive")
        expert = self.mixtures["video"]
        sequence_length: int | None = None
        arena_bytes = 0
        for layer_index, (block, entry) in enumerate(zip(expert.blocks, cache, strict=False)):
            if set(entry) != {"k", "v"}:
                raise ValueError(f"clean-prefix cache layer {layer_index} must contain only k/v")
            for name, key in (
                ("_edge_video_k_arena", "k"),
                ("_edge_video_v_arena", "v"),
            ):
                value = entry[key]
                if value.ndim != 3:
                    raise ValueError(f"clean-prefix {key} must be [B,S,D], got {tuple(value.shape)}")
                if sequence_length is None:
                    sequence_length = int(value.shape[1])
                elif int(value.shape[1]) != sequence_length:
                    raise ValueError("clean-prefix sequence length changed across layers")
                if int(value.shape[1]) >= total_sequence_length:
                    raise ValueError("clean-prefix sequence must be shorter than the complete video sequence")
                arena_shape = (
                    int(value.shape[0]),
                    total_sequence_length,
                    int(value.shape[2]),
                )
                existing = getattr(block.self_attn, name, None)
                if existing is None:
                    arena = torch.empty(
                        arena_shape,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    block.self_attn.register_buffer(name, arena, persistent=False)
                    existing = getattr(block.self_attn, name)
                    if existing.device.type == "cuda":
                        torch._dynamo.mark_static_address(existing)
                elif (
                    tuple(existing.shape) != arena_shape
                    or existing.dtype != value.dtype
                    or existing.device != value.device
                ):
                    raise RuntimeError(f"retained video K/V arena ABI changed at layer {layer_index} {key}")
                existing[:, : int(value.shape[1])].copy_(value)
                arena_bytes += int(existing.numel() * existing.element_size())
        if sequence_length is None or sequence_length <= 0:
            raise RuntimeError("clean-prefix cache is empty")
        self._video_clean_prefix_seq_len = sequence_length
        self._video_clean_prefix_total_seq_len = total_sequence_length
        self._video_retained_kv_arena_bytes = arena_bytes
        return len(cache)

    def _forward_video_future_with_clean_prefix_impl(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
        *,
        return_combined_cache: bool,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Execute future queries against ``[retained clean | live future]``."""
        if self.kv_cache_quantizer is not None:
            raise RuntimeError("clean-prefix wavefront currently requires BF16 video KV storage")
        if "video" not in self.mixtures:
            raise ValueError("MoT requires a video expert for clean-prefix wavefront")
        clean_seq_len = int(getattr(self, "_video_clean_prefix_seq_len", 0))
        total_seq_len = int(video_tokens.shape[1])
        if not 0 < clean_seq_len < total_seq_len:
            raise RuntimeError(f"invalid clean-prefix split {clean_seq_len}/{total_seq_len}")
        if tuple(video_attention_mask.shape) != (total_seq_len, total_seq_len):
            raise ValueError("video attention mask must be square and match the full video sequence")
        if int(video_freqs.shape[0]) != total_seq_len:
            raise ValueError("video RoPE sequence length does not match video tokens")
        if int(video_t_mod.shape[1]) != total_seq_len:
            raise ValueError("video modulation sequence length does not match video tokens")

        expert = self.mixtures["video"]
        x = video_tokens[:, clean_seq_len:]
        future_freqs = video_freqs[clean_seq_len:]
        future_t_mod = video_t_mod[:, clean_seq_len:]
        future_mask = video_attention_mask[clean_seq_len:, :]
        future_context_payload = video_context_payload
        if video_context_payload is not None:
            future_context_payload = dict(video_context_payload)
            context_mask = future_context_payload.get("mask")
            if context_mask is not None:
                if context_mask.dim() == 3:
                    future_context_payload["mask"] = context_mask[:, clean_seq_len:, :]
                elif context_mask.dim() == 4:
                    future_context_payload["mask"] = context_mask[:, :, clean_seq_len:, :]
                else:
                    raise ValueError(f"video context mask must be 3D/4D, got {context_mask.dim()}D")

        block_mask = self.expert_sdpa_block_masks["video"]
        if self.expert_sdpa_backends["video"] == "flex-group-causal":
            block_mask = getattr(self, "_video_future_sdpa_block_mask", None)
            if block_mask is None:
                raise RuntimeError("future-query FlexAttention mask is not installed")

        combined_cache: list[dict[str, torch.Tensor]] = []
        for layer_index, block in enumerate(expert.blocks):
            (
                q,
                k_future,
                v_future,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=future_freqs,
                t_mod=future_t_mod,
            )
            k_arena = getattr(block.self_attn, "_edge_video_k_arena", None)
            v_arena = getattr(block.self_attn, "_edge_video_v_arena", None)
            expected_shape = (
                int(k_future.shape[0]),
                total_seq_len,
                int(k_future.shape[2]),
            )
            if k_arena is None or v_arena is None:
                raise RuntimeError(f"retained video K/V arena is not installed at video layer {layer_index}")
            if tuple(k_arena.shape) != expected_shape or tuple(v_arena.shape) != expected_shape:
                raise RuntimeError(f"retained video K/V arena ABI changed at layer {layer_index}")
            k_arena[:, clean_seq_len:].copy_(k_future)
            v_arena[:, clean_seq_len:].copy_(v_future)
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k_arena,
                v_cat=v_arena,
                attention_mask=future_mask,
                sdpa_backend=self.expert_sdpa_backends["video"],
                sdpa_block_mask=block_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=future_context_payload,
            )
            if return_combined_cache:
                combined_cache.append({"k": k_arena, "v": v_arena})
        return x, combined_cache

    def forward_video_future_with_clean_prefix(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self._forward_video_future_with_clean_prefix_impl(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            return_combined_cache=False,
        )
        return x

    def prefill_video_cache_with_clean_prefix(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        _x, cache = self._forward_video_future_with_clean_prefix_impl(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            return_combined_cache=True,
        )
        return cache
