# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

from __future__ import annotations

import math

from transformers import PretrainedConfig


class G05ActionCodecConfig(PretrainedConfig):
    """Configuration for the ActionCodec shipped with G0.5 checkpoints."""

    model_type = "g05_actioncodec"

    def __init__(
        self,
        max_component_dim: int = 9,
        horizon: int = 32,
        horizon_patch_size: int = 8,
        conv_in_action_kernel: int = 2,
        encoder_channels: int = 256,
        latent_dim: int = 128,
        c_mults: list[int] | None = None,
        strides: list[list[int]] | None = None,
        transformer_depths: list[int] | None = None,
        num_heads: int = 8,
        dim_heads: int = 64,
        ffn_mult: float = 4.0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 0.01,
        use_qk_norm: bool = True,
        rope_base: int = 10_000,
        use_block_dct: bool = True,
        block_dct_block_size: int = 8,
        n_codebooks: int = 4,
        codebook_size: int = 4096,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        ema_decay: float = 0.95,
        threshold_ema_dead: float = 2.0,
        use_rotation_trick: bool = False,
        commitment_loss_weight: float = 0.25,
        reconstruction_loss_weight: float = 1.0,
        parts_meta: dict[str, int] | None = None,
        rule_based_key_patterns: list[str] | None = None,
        rule_based_min_block_len: int = 1,
        rule_based_binarize_threshold: float = 0.0,
        num_residuals: int | None = None,
        use_group_markers: bool = True,
        absent_key_fill_value: float = -100.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_component_dim = max_component_dim
        self.horizon = horizon
        self.horizon_patch_size = horizon_patch_size
        self.conv_in_action_kernel = conv_in_action_kernel
        self.encoder_channels = encoder_channels
        self.latent_dim = latent_dim
        self.c_mults = c_mults or [1, 2, 2]
        self.strides = strides or [[1, 1], [2, 1], [2, 1]]
        self.transformer_depths = transformer_depths or [2, 2, 2]
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.ffn_mult = ffn_mult
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base
        self.use_block_dct = use_block_dct
        self.block_dct_block_size = block_dct_block_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.ema_decay = ema_decay
        self.threshold_ema_dead = threshold_ema_dead
        self.use_rotation_trick = use_rotation_trick
        self.commitment_loss_weight = commitment_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.parts_meta = parts_meta or {}
        self.rule_based_key_patterns = rule_based_key_patterns or ["gripper"]
        self.rule_based_min_block_len = rule_based_min_block_len
        self.rule_based_binarize_threshold = rule_based_binarize_threshold
        self.num_residuals = num_residuals or n_codebooks
        self.use_group_markers = use_group_markers
        self.absent_key_fill_value = absent_key_fill_value

        if horizon % horizon_patch_size:
            raise ValueError("horizon must be divisible by horizon_patch_size")
        if len(self.c_mults) != len(self.strides) or len(self.strides) != len(self.transformer_depths):
            raise ValueError("c_mults, strides, and transformer_depths must have equal lengths")
        if not 1 <= self.num_residuals <= n_codebooks:
            raise ValueError("num_residuals must be between 1 and n_codebooks")
        if any(width > max_component_dim for width in self.parts_meta.values()):
            raise ValueError("released G0.5 ActionCodec does not support parts wider than max_component_dim")

    @property
    def code_height(self) -> int:
        height = self.horizon // self.horizon_patch_size
        for stride_height, _ in self.strides:
            height //= stride_height
        return height

    @property
    def code_width(self) -> int:
        return self.max_component_dim - self.conv_in_action_kernel + 1

    @property
    def code_length(self) -> int:
        return self.code_height * self.code_width

    @property
    def rule_tokens_per_key(self) -> int:
        counts = [0] * (self.horizon + 1)
        for index in range(1, min(self.rule_based_min_block_len + 2, self.horizon + 1)):
            counts[index] = 2 * index
        for index in range(self.rule_based_min_block_len + 2, self.horizon + 1):
            counts[index] = counts[index - 1] + counts[index - self.rule_based_min_block_len - 1]
        valid_sequences = counts[self.horizon]
        return max(1, math.ceil(math.log(valid_sequences, self.codebook_size)))
