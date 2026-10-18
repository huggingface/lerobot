# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/models/wan22/mot.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/models/wan22/mot.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

from __future__ import annotations

from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from lerobot.policies.longwam._core.projections import InferenceProjectionMixin

from .wan_video_dit import flash_attention, modulate, rope_apply

logger = getLogger(__name__)


class MoT(InferenceProjectionMixin, nn.Module):
    def __init__(
        self,
        mixtures: dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        self.kv_cache_quantizer = None
        self.action_segmented_attention = False
        self.expert_rope_compute_dtypes: dict[str, torch.dtype | None] = dict.fromkeys(self.expert_order)
        self.expert_sdpa_backends: dict[str, str | None] = dict.fromkeys(self.expert_order)
        self.expert_sdpa_block_masks: dict[str, BlockMask | None] = dict.fromkeys(self.expert_order)
        self.expert_shared_self_qkv_quantization: dict[str, bool] = dict.fromkeys(self.expert_order, False)
        self.expert_concatenated_self_qkv_gemm: dict[str, bool] = dict.fromkeys(self.expert_order, False)
        self.expert_batched_self_qkv_gemm: dict[str, bool] = dict.fromkeys(self.expert_order, False)
        self.expert_shared_cross_kv_quantization: dict[str, bool] = dict.fromkeys(self.expert_order, False)

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(
            6, dim=chunk_dim
        )
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        sdpa_backend: str | None = None,
        sdpa_block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q,
                k=k,
                v=v,
                num_heads=self.num_heads,
                ctx_mask=attn_mask,
                sdpa_backend=sdpa_backend,
                sdpa_block_mask=sdpa_block_mask,
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: dict | None,
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                normalized = block.norm3(x)
                if context_payload.get("use_cached_kv", False):
                    cached_kv = context_payload.get("cached_kv")
                    if cached_kv is None:
                        cross = block.cross_attn.forward_with_cached_kv(
                            normalized,
                            ctx_mask=context_mask,
                        )
                    elif set(cached_kv) != {"k", "v"}:
                        raise ValueError("Cached cross-attention payload must contain exactly k and v")
                    else:
                        q = block.cross_attn.norm_q(block.cross_attn.q(normalized))
                        cross = block.cross_attn.forward_projected(
                            q,
                            cached_kv["k"],
                            cached_kv["v"],
                            ctx_mask=context_mask,
                        )
                else:
                    cross = block.cross_attn(
                        normalized,
                        context,
                        ctx_mask=context_mask,
                    )
                x = x + cross

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
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
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: FP64 RoPE real/imag pairs aligned with the token sequence,
                shape [S, 1, rope_dim, 2].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        packed_weight = getattr(block.self_attn, "_edge_packed_qkv_weight", None)
        if packed_weight is None:
            if getattr(block.self_attn, "_edge_batched_qkv_gemm", False):
                q, k, v = block.self_attn.project_qkv(attn_input)
            elif getattr(
                block.self_attn,
                "_edge_shared_qkv_quantization",
                False,
            ):
                if attn_input.numel() == 0:
                    q, k, v = (
                        attn_input.new_empty(
                            *attn_input.shape[:-1],
                            projection.out_features,
                        )
                        for projection in (
                            block.self_attn.q,
                            block.self_attn.k,
                            block.self_attn.v,
                        )
                    )
                else:
                    from fouroversix.quantize import quantize_to_fp4

                    activation_config = block.self_attn.q.config.get_activation_config()
                    quantized_attn_input = quantize_to_fp4(
                        attn_input.reshape(-1, attn_input.shape[-1]),
                        activation_config,
                    )
                    if getattr(
                        block.self_attn,
                        "_edge_concat_qkv_gemm",
                        False,
                    ):
                        from fouroversix.matmul.cutlass.ops import (
                            gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_per_col,
                        )

                        activation_denominator = (
                            activation_config.scale_rule.max_allowed_e2m1_value()
                            * activation_config.scale_rule.max_allowed_e4m3_value()
                        )
                        alpha = (
                            quantized_attn_input.amax
                            * block.self_attn._edge_concat_qkv_weight_amax
                            / (
                                activation_denominator
                                * block.self_attn._edge_concat_qkv_weight_scale_denominator
                            )
                        ).to(torch.float32)
                        packed = gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_per_col(
                            quantized_attn_input.values,
                            block.self_attn._edge_concat_qkv_values,
                            quantized_attn_input.scale_factors,
                            block.self_attn._edge_concat_qkv_scales,
                            alpha,
                            block.self_attn._edge_concat_qkv_epilogue_zero,
                        )
                        rows = quantized_attn_input.original_shape[0]
                        width = block.self_attn._edge_concat_qkv_width
                        packed = packed[:rows, : 3 * width].reshape(
                            *attn_input.shape[:-1],
                            3 * width,
                        )
                        packed = packed + block.self_attn._edge_concat_qkv_output_bias
                        q, k, v = packed.chunk(3, dim=-1)
                    else:
                        q = self._project_shared_quantized_linear(
                            quantized_attn_input,
                            attn_input.shape,
                            block.self_attn.q,
                        )
                        k = self._project_shared_quantized_linear(
                            quantized_attn_input,
                            attn_input.shape,
                            block.self_attn.k,
                        )
                        v = self._project_shared_quantized_linear(
                            quantized_attn_input,
                            attn_input.shape,
                            block.self_attn.v,
                        )
            else:
                q = block.self_attn.q(attn_input)
                k = block.self_attn.k(attn_input)
                v = block.self_attn.v(attn_input)
        else:
            packed = F.linear(
                attn_input,
                packed_weight,
                block.self_attn._edge_packed_qkv_bias,
            )
            q, k, v = packed.chunk(3, dim=-1)
        q = block.self_attn.norm_q(q)
        k = block.self_attn.norm_k(k)

        q = rope_apply(
            q,
            freqs,
            block.num_heads,
            block.self_attn.rope_compute_dtype,
            block.self_attn.rope_direct_output,
        )
        k = rope_apply(
            k,
            freqs,
            block.num_heads,
            block.self_attn.rope_compute_dtype,
            block.self_attn.rope_direct_output,
        )

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: dict | None,
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """

        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> list[Any]:
        return self._prefill_video_cache_impl(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            sdpa_block_mask=self.expert_sdpa_block_masks["video"],
        )

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: dict | None,
        video_kv_cache: list[Any],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_cross_kv_cache: list[Any] | None = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE real/imag pairs, shape [Sa, 1, rope_dim, 2].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if action_cross_kv_cache is not None and len(action_cross_kv_cache) != self.num_layers:
            raise ValueError(
                "`action_cross_kv_cache` must contain "
                f"{self.num_layers} layers, got {len(action_cross_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
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
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if self.kv_cache_quantizer is None:
                if "k" not in layer_cache or "v" not in layer_cache:
                    raise ValueError(f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`.")
                k_video = layer_cache["k"]
                v_video = layer_cache["v"]
            else:
                k_video, v_video = self.kv_cache_quantizer.dequantize(layer_cache)
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}.")

            # Action queries attend to cached video K/V plus current action K/V.
            if self.action_segmented_attention:
                raise NotImplementedError("Segmented hardware kernels are not part of this reference port")

            else:
                k_cat = torch.cat([k_video, k_action], dim=1)
                v_cat = torch.cat([v_video, v_action], dim=1)
                mixed = self._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=action_attention_mask,
                )
            layer_context_payload = action_context_payload
            if action_cross_kv_cache is not None:
                layer_context_payload = dict(action_context_payload or {})
                layer_context_payload["use_cached_kv"] = True
                layer_context_payload["cached_kv"] = action_cross_kv_cache[layer_idx]
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=layer_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        context_all: dict[str, dict | None],
        t_mod_all: dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = dict(embeds_all.items())

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

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
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens, strict=False):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
