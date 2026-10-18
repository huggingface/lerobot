# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/models/wan22/action_dit.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/models/wan22/action_dit.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

import os
from logging import getLogger
from typing import Any

import torch
import torch.nn as nn

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)

logger = getLogger(__name__)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_ROPE_H32 = 32
    ACTION_ROPE_MODES = frozenset({"cpu_per_step", "gpu_resident"})
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)
        self._action_rope_h32_cache: torch.Tensor | None
        self.register_buffer("_action_rope_h32_cache", None, persistent=False)
        self._action_rope_mode = "cpu_per_step"
        self._action_rope_cache_generation = 0
        self._action_rope_cache_build_count = 0
        self._action_rope_cache_dtype = torch.float64
        self._action_rope_source_identity: tuple[int, int] | None = None
        self._action_rope_cache_identity: tuple[int, int, int] | None = None

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @property
    def action_rope_mode(self) -> str:
        return self._action_rope_mode

    @property
    def action_rope_cache(self) -> torch.Tensor | None:
        return self._action_rope_h32_cache

    def configure_action_rope_mode(self, mode: str) -> None:
        normalized = str(mode)
        if normalized not in self.ACTION_ROPE_MODES:
            raise ValueError(
                f"unsupported Action RoPE mode {normalized!r}; "
                f"expected one of {sorted(self.ACTION_ROPE_MODES)}"
            )
        if normalized == self._action_rope_mode:
            return
        if self._action_rope_h32_cache is not None:
            raise RuntimeError("Action RoPE mode cannot change while a resident cache exists.")
        self._action_rope_mode = normalized
        self._action_rope_cache_generation += 1

    def invalidate_action_rope_cache(self) -> None:
        had_cache = self._action_rope_h32_cache is not None
        self._action_rope_h32_cache = None
        self._action_rope_source_identity = None
        self._action_rope_cache_identity = None
        if had_cache:
            self._action_rope_cache_generation += 1

    def _apply(self, fn, recurse: bool = True):
        # A later model.to()/dtype conversion must not silently migrate a
        # buffer whose address was frozen before compilation/capture.
        if hasattr(self, "_action_rope_h32_cache"):
            self.invalidate_action_rope_cache()
        return super()._apply(fn, recurse=recurse)

    def prepare_action_rope_cache(
        self,
        device: torch.device | str,
        *,
        storage_dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        if self._action_rope_mode != "gpu_resident":
            raise RuntimeError("prepare_action_rope_cache requires action_rope_mode='gpu_resident'.")
        if self._action_rope_h32_cache is not None or self._action_rope_cache_build_count != 0:
            raise RuntimeError("Action H32 RoPE cache may be built only once per model setup.")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError(f"Action H32 RoPE residency requires CUDA, got {target}.")
        if storage_dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Action RoPE resident storage must be float32 or float64, got {storage_dtype}.")
        source = self.freqs
        source_view = source[: self.ACTION_ROPE_H32].view(
            self.ACTION_ROPE_H32,
            1,
            -1,
            2,
        )
        cache = source_view.to(
            device=target,
            dtype=storage_dtype,
            non_blocking=True,
        ).detach()
        expected_shape = (
            self.ACTION_ROPE_H32,
            1,
            self.attn_head_dim // 2,
            2,
        )
        if (
            tuple(cache.shape) != expected_shape
            or cache.dtype != storage_dtype
            or cache.device != target
            or cache.requires_grad
        ):
            raise RuntimeError("Action H32 RoPE cache does not preserve the frozen eager contract.")
        self._action_rope_h32_cache = cache
        self._action_rope_cache_dtype = storage_dtype
        self._action_rope_source_identity = (id(source), int(source._version))
        self._action_rope_cache_identity = (
            id(cache),
            int(cache.data_ptr()),
            int(cache._version),
        )
        self._action_rope_cache_generation += 1
        self._action_rope_cache_build_count += 1
        return cache

    def select_action_rope_freqs(
        self,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")
        if self._action_rope_mode == "cpu_per_step":
            return self.freqs[:seq_len].view(seq_len, 1, -1, 2).to(device)
        if self._action_rope_mode != "gpu_resident":
            raise RuntimeError(f"invalid Action RoPE mode: {self._action_rope_mode!r}")
        if seq_len != self.ACTION_ROPE_H32:
            raise RuntimeError(f"gpu_resident Action RoPE is frozen for exact H32; got H{seq_len}.")
        if hasattr(torch, "compiler") and torch.compiler.is_compiling():
            # The setup path already froze and validated this buffer.  Keep
            # Python object/version auditing outside the Dynamo graph.
            return self._action_rope_h32_cache
        source = self.freqs
        cache = self._action_rope_h32_cache
        if (
            self._action_rope_source_identity != (id(source), int(source._version))
            or cache is None
            or self._action_rope_cache_identity != (id(cache), int(cache.data_ptr()), int(cache._version))
            or cache.device != device
            or cache.dtype != self._action_rope_cache_dtype
            or cache.requires_grad
        ):
            raise RuntimeError("Action H32 RoPE resident cache identity changed.")
        return cache

    def action_rope_descriptor(self) -> dict[str, Any]:
        cache = self._action_rope_h32_cache
        return {
            "mode": self._action_rope_mode,
            "cache_generation": int(self._action_rope_cache_generation),
            "cache_build_count": int(self._action_rope_cache_build_count),
            "storage_dtype": str(self._action_rope_cache_dtype),
            "grid_size": [self.ACTION_ROPE_H32],
            "cache": (
                None
                if cache is None
                else {
                    "data_ptr": int(cache.data_ptr()),
                    "shape": [int(item) for item in cache.shape],
                    "stride": [int(item) for item in cache.stride()],
                    "dtype": str(cache.dtype),
                    "device": str(cache.device),
                    "payload_bytes": int(cache.numel() * cache.element_size()),
                }
            ),
        }

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info(
                "No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path

        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(
            action_dit_pretrained_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {action_dit_pretrained_path}, got {type(meta)}")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        load_result = action_expert.load_state_dict(merged_state, strict=True)
        action_expert._longwam_strict_backbone_load_report = {
            "source_path": str(Path(action_dit_pretrained_path).resolve()),
            "state_container": "backbone_state_dict",
            "strict": True,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "source_tensors": [
                {
                    "name": key,
                    "shape": [int(dimension) for dimension in value.shape],
                    "dtype": str(value.dtype),
                    "numel": int(value.numel()),
                }
                for key, value in sorted(backbone_state_dict.items())
            ],
            "excluded_seed42_tensors": sorted(set(action_state).difference(expected_backbone_keys)),
        }
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.select_action_rope_freqs(
            seq_len=seq_len,
            device=tokens.device,
        )

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
