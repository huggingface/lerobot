#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

"""XVLA-RMoE: X-VLA with Cross-Step Routing Memory dense soft Mixture-of-Experts.

Ports the Cross-Step Routing Memory MoE idea from `lerobot.policies.smolvla_rmoe` onto
X-VLA's own architecture (`lerobot.policies.xvla`) instead of copying its implementation:
X-VLA is a single homogeneous Transformer over `[action | VLM | aux-view | soft-prompt]`
tokens with **no** KV cache / prefix-suffix split and **no** velocity target -- it predicts
the clean action directly at every denoising step (see `XVLAModel.forward`/`generate_actions`
in `lerobot.policies.xvla.modeling_xvla`). See `moe_soft_transformer.py` for the
architectural adaptation this implies for `MoEFFN` and the routing summaries.

Baseline modes (share the exact same code path, only `XVLARMoEConfig` flags differ, so
comparisons between them are apples-to-apples given equal expert count / MoE layers /
training budget):
    original-compatible : use_moe=False
                           (byte-identical code path to plain `xvla`, given equal weights/seed)
    stateless MoE        : use_moe=True, use_routing_memory=False, use_timestep_router=False
    timestep-only MoE    : use_moe=True, use_routing_memory=False, use_timestep_router=True
    proposed RMoE        : use_moe=True, use_routing_memory=True, use_timestep_router=True,
                            use_recurrent_routing_training=True   (all defaults)
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import tempfile
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
import torch
import torch.nn.functional as functional
from huggingface_hub.constants import CONFIG_NAME
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy, T
from ..utils import log_model_loading_keys
from ..xvla.action_hub import build_action_space
from ..xvla.modeling_xvla import XVLAPolicy, pad_tensor_along_dim
from .configuration_xvla_rmoe import XVLARMoEConfig
from .moe_soft_transformer import (
    MoEFFN,
    RoutingMemoryCell,
    SoftPromptedTransformerRMoE,
    compute_routing_delta_t,
    timestep_embedding,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import Florence2Config, Florence2Model
else:
    Florence2Config = None
    Florence2Model = None


@dataclass
class RoutingInfo:
    """Routing summary produced by one denoising / recurrent-training step.

    Shapes:
        routing_summary: (B, num_moe_layers * num_experts) -- concatenation of every
            MoEFFN layer's mean gating weights (over valid action tokens) for this step.
        hidden_summary: (B, hidden_size) -- masked-mean-pooled final action-token hidden
            state.
        next_routing_state: (B, routing_hidden_dim) -- the GRU state produced by this step,
            i.e. the state the *next* step's routers will read.
        layer_routing_weights: analysis/eval only. List of raw (B, L, num_experts) per-token
            routing weights, one per MoE layer. Always None on the training hot path -- a
            (B, num_moe_layers, seq_len, num_experts) tensor is far too large to keep around
            every step (see `XVLARMoEConfig.return_full_routing_weights`).
    """

    routing_summary: Tensor
    hidden_summary: Tensor
    next_routing_state: Tensor
    layer_routing_weights: list[Tensor] | None = None


class XVLARMoEModel(nn.Module):
    """XVLA backbone with `SoftPromptedTransformerRMoE` in place of `SoftPromptedTransformer`.

    Mirrors `lerobot.policies.xvla.modeling_xvla.XVLAModel` (VLM setup, dtype/freezing,
    `forward_vlm`, action-space handling are copied unchanged -- `xvla/modeling_xvla.py`
    itself is never imported-and-subclassed at the model level, since its `__init__`
    hardcodes the non-MoE transformer) and adds:
      * a `RoutingMemoryCell` (only constructed when `use_moe and use_routing_memory`),
      * the single-t `forward` (h_0 = 0 every call, exactly mirroring the original
        single-timestep training -- with `use_moe=False` this is numerically identical to
        `XVLAModel.forward` given equal weights/seed),
      * `forward_recurrent` for truncated cross-step recurrent training,
      * `generate_actions` with cross-step routing memory threaded through the denoising loop
        (reset to zero at the start of every call, i.e. every new action chunk).

    Observation encoding (`forward_vlm`, the Florence-2 pass) has no timestep dependence in
    the original X-VLA and is therefore computed exactly once per call in every method below
    and reused across every denoising / recurrent step, exactly as `XVLAModel` already does.
    The Transformer itself cannot be cached across steps: action tokens change every step and
    attend densely with every other token (no KV cache in X-VLA's design), so it is fully
    recomputed each step, by design (see `moe_soft_transformer.py` module docstring).
    """

    def __init__(self, config: XVLARMoEConfig, florence_config: Florence2Config, proprio_dim: int) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        if config.action_mode.lower() == "auto":
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        self.vlm = Florence2Model(florence_config)
        # `xvla_rmoe` only uses the encoder-side path of Florence-2; drop the text decoder
        # entirely to save memory. Identical to `XVLAModel.__init__` -- keeping `self.vlm` a
        # bare `Florence2Model` (not `Florence2ForConditionalGeneration`) keeps every
        # parameter name identical to plain XVLA, which `use_moe=False` compatibility mode
        # relies on for 1:1 checkpoint loading.
        del self.vlm.language_model.decoder

        projection_dim = getattr(florence_config.vision_config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = SoftPromptedTransformerRMoE(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            moe_layer_indices=config.moe_layer_indices if config.use_moe else [],
            num_moe_experts=config.num_moe_experts,
            routing_hidden_dim=config.routing_hidden_dim,
            routing_timestep_dim=config.routing_timestep_dim,
            chunk_pos_emb_dim=config.chunk_pos_emb_dim,
            use_routing_memory=config.use_routing_memory,
            use_timestep_router=config.use_timestep_router,
            use_chunk_position_embedding=config.use_chunk_position_embedding,
            expert_symmetry_breaking_std=config.expert_symmetry_breaking_std,
        )

        self.routing_memory = (
            RoutingMemoryCell(
                num_moe_layers=self.transformer.num_moe_layers,
                num_experts=config.num_moe_experts,
                routing_hidden_dim=config.routing_hidden_dim,
                routing_timestep_dim=config.routing_timestep_dim,
                hidden_size=config.hidden_size,
                use_delta_t_conditioning=config.use_delta_t_conditioning,
            )
            if (config.use_moe and config.use_routing_memory)
            else None
        )

        self._apply_freezing()
        self._apply_dtype()

    def _get_target_dtype(self) -> torch.dtype:
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        self.to(dtype=self._get_target_dtype())

    def _apply_freezing(self) -> None:
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "encoder"):
                for param in lm.encoder.parameters():
                    param.requires_grad = False
            if hasattr(lm, "shared"):
                for param in lm.shared.parameters():
                    param.requires_grad = False

        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompts" not in name:
                    param.requires_grad = False

        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode text and multi-view images via the Florence-2 encoder. Identical to
        `XVLAModel.forward_vlm`: has no timestep dependence, so callers compute it once and
        reuse the result across every denoising / recurrent step."""
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm.get_image_features(valid_images).pooler_output
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        # XVLA prepends the primary view's image tokens to the text embeddings and attends to everything.
        merged_embeds = torch.cat([image_features[:, 0], inputs_embeds], dim=1)
        attention_mask = torch.ones(merged_embeds.shape[:2], dtype=torch.long, device=merged_embeds.device)

        enc_out = self.vlm.language_model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def _routing_timestep_emb(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of a scalar-per-batch timestep -> (B, routing_timestep_dim)
        float32. Reuses X-VLA's own stateless sinusoidal primitive (`timestep_embedding`,
        `..xvla.soft_transformer`) but through its own learned router/GRU projections --
        distinct from the action model's own `dim_time` embedding (`action_encoder`)."""
        return timestep_embedding(t.to(dtype=torch.float32), self.config.routing_timestep_dim).to(
            dtype=torch.float32
        )

    def _compute_action_loss(
        self,
        predicted_action: Tensor,
        target_action: Tensor,
        action_valid_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        """Compute the X-VLA action loss over valid episode timesteps only."""
        if action_valid_mask is not None:
            valid = action_valid_mask.to(device=predicted_action.device, dtype=torch.bool)
            if valid.shape != predicted_action.shape[:2]:
                raise ValueError(
                    f"action_valid_mask shape {tuple(valid.shape)} does not match "
                    f"action horizon {tuple(predicted_action.shape[:2])}."
                )
            if not valid.any():
                raise ValueError("Action chunk contains no valid timesteps.")
            # Collapse valid tokens into one synthetic sequence so padded
            # end-of-episode targets contribute no gradient.
            predicted_action = predicted_action[valid].unsqueeze(0)
            target_action = target_action[valid].unsqueeze(0)

        if self.config.action_mode.lower() == "ee6d" and self.config.single_arm_ee6d_loss:
            if predicted_action.shape[-1] < 10 or target_action.shape[-1] < 10:
                raise ValueError("Single-arm EE6D loss requires at least 10 action channels.")
            # LIBERO owns only the first X-VLA arm slot:
            # xyz [0:3], rotation-6D [3:9], gripper [9]. Channels [10:20]
            # are structural padding and must not supervise the model.
            return {
                "position_loss": functional.mse_loss(predicted_action[..., :3], target_action[..., :3])
                * self.action_space.XYZ_SCALE,
                "rotate6D_loss": functional.mse_loss(predicted_action[..., 3:9], target_action[..., 3:9])
                * self.action_space.ROT_SCALE,
                "gripper_loss": functional.binary_cross_entropy_with_logits(
                    predicted_action[..., 9], target_action[..., 9]
                )
                * self.action_space.GRIPPER_SCALE,
            }

        return self.action_space.compute_loss(predicted_action, target_action)

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        action_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-timestep training forward. Preserves the original X-VLA single-t
        flow-matching training exactly: `routing_state` starts at zero on *every* call (no
        cross-call persistence), so with `use_moe=False` this reduces to `XVLAModel.forward`
        bit-for-bit given the same weights and RNG state."""
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        routing_state = (
            self.routing_memory.initial_state(batch_size, input_ids.device)
            if self.routing_memory is not None
            else None
        )
        timestep_emb = (
            self._routing_timestep_emb(t)
            if (self.config.use_moe and self.config.use_timestep_router)
            else None
        )

        pred_action, _routing_decisions, _hidden_summary, _full_weights = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            routing_state=routing_state,
            timestep_emb=timestep_emb,
            action_padding_mask=action_padding_mask,
            **enc,
        )
        return self._compute_action_loss(pred_action, action, action_padding_mask)

    def _sample_recurrent_timesteps(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """K consecutive, strictly decreasing timesteps on X-VLA's own inference grid
        `t_s = 1 - s/S` (see `generate_actions` below), shared by the whole batch."""
        num_steps = self.config.num_denoising_steps
        k = self.config.recurrent_unroll_steps
        max_start = num_steps - k
        start_idx = int(torch.randint(0, max_start + 1, (1,)).item())
        indices = torch.arange(start_idx, start_idx + k, device=device, dtype=torch.float32)
        timesteps = 1.0 - indices / num_steps
        return timesteps.to(dtype=dtype)

    def forward_recurrent(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        action_padding_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, Tensor]], list[RoutingInfo], dict[str, float]]:
        """K-step truncated recurrent training.

        Shares one noise sample and one ground-truth action chunk across K consecutive
        denoising timesteps. Every `x_t_q` is built directly from X-VLA's own exact
        conditional path `t*noise + (1-t)*action` -- never from a previous step's predicted
        action -- so the only thing coupling the K steps is `routing_state` flowing through
        the GRU, with no solver error leaking into the supervised target. `forward_vlm` is
        computed once and reused for every step (see class docstring); the Transformer itself
        is re-run in full every step (X-VLA has no KV cache to reuse there).

        Returns:
            step_losses: K loss dicts, one per step, each in `action_space.compute_loss`'s own
                format (e.g. `{"position_loss": ..., "gripper_loss": ...}` for `ee6d`).
            routing_infos: K `RoutingInfo`, one per step (post-GRU-update).
            extra: scalar logging stats (routing memory norm / delta norm, router entropy,
                   expert utilization) aggregated over the K steps.
        """
        if self.routing_memory is None:
            raise RuntimeError("forward_recurrent() requires `use_moe=True` and `use_routing_memory=True`.")

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        if noise is None:
            noise = torch.randn_like(action)

        timesteps = self._sample_recurrent_timesteps(device, target_dtype)  # (K,)

        routing_state = self.routing_memory.initial_state(batch_size, device)
        previous_t: Tensor | None = None
        step_losses: list[dict[str, Tensor]] = []
        routing_infos: list[RoutingInfo] = []
        memory_norms: list[Tensor] = []
        memory_delta_norms: list[Tensor] = []
        last_routing_decisions: list[Tensor] = []

        for step_idx in range(timesteps.shape[0]):
            t = timesteps[step_idx].expand(batch_size)
            t_expanded = t.view(-1, 1, 1)
            # Ground-truth conditional flow path -- never an Euler update of a previous
            # prediction, so the recurrent dependency lives only in routing_state.
            action_noisy = noise * t_expanded + action * (1 - t_expanded)
            proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

            timestep_emb = self._routing_timestep_emb(t) if self.config.use_timestep_router else None

            pred_action, routing_decisions, hidden_summary, _full_weights = self.transformer(
                domain_id=domain_id,
                action_with_noise=action_noisy_m,
                t=t,
                proprio=proprio_m,
                routing_state=routing_state,
                timestep_emb=timestep_emb,
                action_padding_mask=action_padding_mask,
                **enc,
            )
            step_losses.append(self._compute_action_loss(pred_action, action, action_padding_mask))
            last_routing_decisions = routing_decisions

            delta_t = torch.zeros_like(t) if previous_t is None else compute_routing_delta_t(t, previous_t)
            previous_t = t

            routing_t_emb = self._routing_timestep_emb(t)
            routing_dt_emb = (
                self._routing_timestep_emb(delta_t) if self.config.use_delta_t_conditioning else None
            )
            prev_state = routing_state
            # h_{q+1} = GRU([routing_decisions | Pool(H_q) | e(t_q) | e(Δt_q)], h_q). The
            # router at step q only ever saw `prev_state` (h_q); h_{q+1} is computed here,
            # after step q's loss inputs are already fixed, and only affects step q+1 onward.
            routing_state = self.routing_memory(
                routing_decisions, hidden_summary, routing_t_emb, routing_dt_emb, routing_state
            )
            routing_infos.append(
                RoutingInfo(
                    routing_summary=torch.cat(routing_decisions, dim=-1),
                    hidden_summary=hidden_summary,
                    next_routing_state=routing_state,
                )
            )
            memory_norms.append(routing_state.detach().norm(dim=-1).mean())
            memory_delta_norms.append((routing_state - prev_state).detach().norm(dim=-1).mean())

            if self.config.recurrent_detach_state:
                routing_state = routing_state.detach()

        extra: dict[str, float] = {
            "routing_memory_norm": torch.stack(memory_norms).mean().item(),
            "routing_memory_delta_norm": torch.stack(memory_delta_norms).mean().item(),
        }
        if last_routing_decisions:
            mean_weights = torch.stack(last_routing_decisions, dim=0).mean(dim=(0, 1))
            probs = mean_weights.clamp_min(1e-8)
            extra["router_entropy"] = (-(probs * probs.log()).sum()).item()
            extra["expert_utilization_min"] = mean_weights.min().item()
            extra["expert_utilization_max"] = mean_weights.max().item()

        return step_losses, routing_infos, extra

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
        action_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference denoising loop with cross-step routing memory. Preserves X-VLA's
        original Euler/x0-prediction loop and timestep schedule exactly (`t = i/steps`,
        `x_t = x1*t + action*(1-t)`); `routing_state` is reset to zero at the *start* of this
        call, i.e. every new action-chunk generation, and never persists across calls."""
        self.eval()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action
        device = proprio.device

        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        routing_state = (
            self.routing_memory.initial_state(batch_size, device) if self.routing_memory is not None else None
        )
        prev_time = 1.0

        for i in range(steps, 0, -1):
            time = i / steps
            t = torch.full((batch_size,), time, device=device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)

            timestep_emb = (
                self._routing_timestep_emb(t)
                if (self.config.use_moe and self.config.use_timestep_router)
                else None
            )

            action, routing_decisions, hidden_summary, _full_weights = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                routing_state=routing_state,
                timestep_emb=timestep_emb,
                action_padding_mask=action_padding_mask,
                **enc,
            )

            if self.routing_memory is not None and routing_decisions:
                delta_t = compute_routing_delta_t(time, prev_time)
                dt_tensor = torch.full((batch_size,), delta_t, device=device, dtype=target_dtype)
                routing_t_emb = self._routing_timestep_emb(t)
                routing_dt_emb = (
                    self._routing_timestep_emb(dt_tensor) if self.config.use_delta_t_conditioning else None
                )
                routing_state = self.routing_memory(
                    routing_decisions, hidden_summary, routing_t_emb, routing_dt_emb, routing_state
                )
            prev_time = time

        return self.action_space.postprocess(action)


def _load_xvla_rmoe_config(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> XVLARMoEConfig:
    """Load a `config.json` (from a plain X-VLA checkpoint, an X-VLA-RMoE checkpoint, or any
    other `XVLAConfig`-schema-compatible checkpoint) as an `XVLARMoEConfig`, regardless of what
    `"type"` the checkpoint itself declares.

    `PreTrainedConfig.from_pretrained` (the generic loader) determines *which dataclass* to
    decode the JSON as from the checkpoint's own `"type"` field, not from the class it's called
    on -- see its "HACK" docstring in `lerobot/configs/policies.py`. That is exactly right for
    loading a same-typed checkpoint, but means `XVLARMoEPolicy.from_pretrained("plain/xvla-ckpt")`
    would silently come back as a plain `XVLAConfig` (missing every `xvla_rmoe`-only field), and
    `lerobot-train --policy.type=xvla_rmoe --policy.path=<a plain xvla checkpoint>` hits the same
    thing one layer up in `TrainPipelineConfig` -- worse, there the type-changing re-decode drops
    even shared nested fields like `florence_config`, since draccus rebuilds the CLI-requested
    type from scratch rather than merging it with a JSON checkpoint of a different declared type.

    This instead always decodes straight into `XVLARMoEConfig` (a strict field superset of
    `XVLAConfig`): fields present in the checkpoint's JSON (including `florence_config`) are
    loaded, and any `xvla_rmoe`-only fields absent from a plain-`xvla` checkpoint fall back to
    `XVLARMoEConfig`'s own defaults -- exactly the "load a plain X-VLA checkpoint into RMoE with
    default RMoE settings" behavior `_remap_xvla_mlp_weights_to_moe` is built to support.
    """
    model_id = str(pretrained_name_or_path)
    if os.path.isdir(model_id):
        config_file = os.path.join(model_id, CONFIG_NAME)
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError

        try:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except HfHubHTTPError as e:
            raise FileNotFoundError(f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}") from e

    with open(config_file) as f:
        raw_config = json.load(f)
    raw_config.pop("type", None)

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(raw_config, f)
        stripped_config_file = f.name
    with draccus.config_type("json"):
        return draccus.parse(XVLARMoEConfig, stripped_config_file, args=[])


def _merge_pretrained_xvla_rmoe_config(
    pretrained_config: XVLARMoEConfig,
    requested_config: PreTrainedConfig,
) -> XVLARMoEConfig:
    """Merge CLI/runtime overrides onto the config stored in an X-VLA checkpoint."""
    if not isinstance(requested_config, XVLARMoEConfig):
        raise TypeError(
            "XVLARMoEPolicy requires an XVLARMoEConfig when `config` is supplied, "
            f"got {type(requested_config).__name__}."
        )

    checkpoint_contract_fields = {"input_features", "output_features", "florence_config"}
    runtime_fields = {
        "device",
        "pretrained_path",
        "pretrained_revision",
        "repo_id",
        "push_to_hub",
        "private",
        "tags",
        "license",
        "use_peft",
    }
    for config_field in fields(XVLARMoEConfig):
        name = config_field.name
        requested_value = getattr(requested_config, name)
        if name in checkpoint_contract_fields:
            continue
        if config_field.default is not MISSING:
            default_value = config_field.default
        elif config_field.default_factory is not MISSING:
            default_value = config_field.default_factory()
        else:
            continue
        if name in runtime_fields or requested_value != default_value:
            setattr(pretrained_config, name, requested_value)

    return pretrained_config


def _remap_xvla_mlp_weights_to_moe(
    state_dict: dict[str, Tensor],
    model: nn.Module,
    config: XVLARMoEConfig,
) -> dict[str, Tensor]:
    """Broadcast a plain X-VLA checkpoint's per-block FFN (`transformer.blocks.{l}.mlp.*`,
    i.e. `fc1`/`fc2` weights of the original `Mlp`) into every `MoEFFN` expert.

    Plain X-VLA has no `experts.{e}.` / `router.` keys, so it is loaded on top of a freshly
    constructed `XVLARMoEModel` (whose `MoEFFN` layers already deep-copied and
    symmetry-broke their own random-init FFN at construction time -- see
    `SoftPromptedTransformerRMoE.__init__`): every MoE layer's *source* `fc1`/`fc2` weight is
    broadcast into all of its experts (with the same symmetry-breaking noise applied as at
    construction, so fine-tuning from a plain checkpoint doesn't silently reintroduce the
    permutation-symmetry trap -- see `expert_symmetry_breaking_std`), and the router / GRU /
    hidden_proj stay at their fresh random init (there is nothing to load for them). Loading
    an already-RMoE checkpoint (state dict already has `experts.`/`router.` keys under every
    MoE layer) is a no-op here.
    """
    moe_modules = {name: module for name, module in model.named_modules() if isinstance(module, MoEFFN)}
    if not moe_modules:
        return state_dict

    already_moe = any(
        any(key.startswith(f"{name}.experts.") or key.startswith(f"{name}.router.") for name in moe_modules)
        for key in state_dict
    )
    if already_moe:
        return state_dict

    remapped = dict(state_dict)
    std = config.expert_symmetry_breaking_std
    for moe_name, moe_module in moe_modules.items():
        source_prefix = f"{moe_name}."
        source_keys = [key for key in state_dict if key.startswith(source_prefix)]
        for key in source_keys:
            suffix = key[len(source_prefix) :]
            value = remapped.pop(key)
            for expert_idx in range(len(moe_module.experts)):
                broadcast_value = value.clone()
                if std > 0.0 and broadcast_value.is_floating_point():
                    broadcast_value = broadcast_value + torch.randn_like(broadcast_value) * std
                remapped[f"{source_prefix}experts.{expert_idx}.{suffix}"] = broadcast_value
    return remapped


class XVLARMoEPolicy(XVLAPolicy):
    """LeRobot-compliant wrapper around `XVLARMoEModel`.

    Subclasses `XVLAPolicy` to reuse its batch-prep plumbing (`_prepare_state`,
    `_prepare_images`, `_get_domain_id`, `_prepare_action_targets`, `reset`,
    `get_optim_params`, `predict_action_chunk`, `select_action`, `_get_action_chunk`) as-is
    -- those only touch `self.config` / `self.model.{dim_proprio,dim_action}` /
    `self.model.generate_actions(...)`, all of which `XVLARMoEModel` also provides with a
    compatible signature. Only `__init__` (builds `XVLARMoEModel` instead of `XVLAModel`),
    `_build_model_inputs` (adds `action_padding_mask`), `forward` (adds the truncated
    recurrent-training branch) and `from_pretrained` (adds MoE checkpoint remapping) differ.
    """

    config_class = XVLARMoEConfig
    name = "xvla_rmoe"

    def __init__(self, config: XVLARMoEConfig, **kwargs):
        require_package("transformers", extra="xvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLARMoEModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        # Deterministic, DDP-safe counter used to pick single-t vs. recurrent training mode
        # (see `_should_use_recurrent_step`). Not persisted in the checkpoint: the schedule
        # simply restarts on resume, which has no effect on correctness.
        self._train_call_counter = 0
        self.reset()

    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        inputs = super()._build_model_inputs(batch)
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            inputs["action_padding_mask"] = None
        else:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"action_is_pad must have shape (batch, horizon), got {tuple(action_is_pad.shape)}."
                )
            # The dataset can be constructed from the CLI default horizon before a pretrained
            # checkpoint overrides chunk_size. Keep the mask aligned with
            # _prepare_action_targets, which pads/truncates actions to the checkpoint horizon.
            action_valid = ~action_is_pad.to(dtype=torch.bool)
            inputs["action_padding_mask"] = pad_tensor_along_dim(action_valid, self.config.chunk_size, dim=1)
        return inputs

    def _should_use_recurrent_step(self) -> bool:
        """Deterministic (call-count based, not `torch.rand`) single-t vs. truncated
        recurrent-training branch selection -- every DDP rank calls `forward()` the same
        number of times per optimizer step, so the counter (and therefore the branch) stays
        in lockstep across ranks by construction. See `smolvla_rmoe`'s
        `should_use_recurrent_step` for the identical reasoning this mirrors."""
        if not (self.training and self.config.use_recurrent_routing_training):
            return False
        prob = self.config.recurrent_training_probability
        if prob <= 0.0:
            return False
        interval = max(1, round(1.0 / prob))
        step = self._train_call_counter
        self._train_call_counter += 1
        return step % interval == 0

    def _expert_pairwise_output_distance(self) -> float:
        """Mean pairwise L2 distance between MoE experts' own parameters (cheap, forward-only
        proxy for "have the experts diverged from their bit-identical init"), averaged over
        every expert pair in every MoE layer. 0.0 if MoE is disabled."""
        distances = []
        for module in self.model.transformer.blocks:
            if not isinstance(module.mlp, MoEFFN):
                continue
            experts = module.mlp.experts
            for i in range(len(experts)):
                for j in range(i + 1, len(experts)):
                    sq_dist = sum(
                        (p_i.detach() - p_j.detach()).pow(2).sum()
                        for p_i, p_j in zip(experts[i].parameters(), experts[j].parameters(), strict=True)
                    )
                    distances.append(torch.sqrt(sq_dist).item())
        return sum(distances) / len(distances) if distances else 0.0

    def _forward_recurrent(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        step_losses, _routing_infos, extra = self.model.forward_recurrent(action=targets, **inputs)

        per_step_totals = torch.stack([sum(losses.values()) for losses in step_losses])
        loss = (
            per_step_totals.mean()
            if self.config.recurrent_loss_reduction == "mean"
            else per_step_totals.sum()
        )

        log_dict = {
            "train/used_recurrent_training": True,
            "train/recurrent_unroll_steps": self.config.recurrent_unroll_steps,
            "train/recurrent_flow_loss": loss.detach().item(),
            "train/routing_memory_norm": extra["routing_memory_norm"],
            "train/routing_memory_delta_norm": extra["routing_memory_delta_norm"],
        }
        for key in ("router_entropy", "expert_utilization_min", "expert_utilization_max"):
            if key in extra:
                log_dict[f"train/{key}"] = extra[key]
        log_dict["train/expert_pairwise_output_distance"] = self._expert_pairwise_output_distance()
        log_dict["loss"] = loss.detach().item()
        return loss, log_dict

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self._should_use_recurrent_step():
            return self._forward_recurrent(batch)

        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())

        log_dict = {k: v.detach().item() for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        log_dict["train/used_recurrent_training"] = False
        log_dict["train/single_flow_loss"] = log_dict["loss"]
        if self.config.use_moe:
            log_dict["train/expert_pairwise_output_distance"] = self._expert_pairwise_output_distance()
        return total_loss, log_dict

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,  # noqa: ARG003 -- always loaded with strict=False, see below
        **kwargs,
    ):
        """Load either a plain X-VLA checkpoint (broadcasting its FFN weights into every
        `MoEFFN` expert, see `_remap_xvla_mlp_weights_to_moe`) or an already-RMoE checkpoint
        (loaded as-is). Missing keys after remapping are expected to be exactly the freshly
        initialized router / GRU / hidden_proj parameters; anything else is logged."""
        import safetensors.torch

        pretrained_config = _load_xvla_rmoe_config(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        if config is None:
            config = pretrained_config
        else:
            config = _merge_pretrained_xvla_rmoe_config(pretrained_config, config)

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e

        logging.info(f"Loading checkpoint from {model_file}")
        state_dict = safetensors.torch.load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]

        state_dict = _remap_xvla_mlp_weights_to_moe(state_dict, instance, instance.config)

        missing_keys, unexpected_keys = instance.load_state_dict(state_dict, strict=False)
        expected_missing_substrings = ("router.", "gru_cell.", "hidden_proj.", "routing_memory.")
        unexpected_missing = [
            key for key in missing_keys if not any(sub in key for sub in expected_missing_substrings)
        ]
        if unexpected_missing or unexpected_keys:
            log_model_loading_keys(unexpected_missing, list(unexpected_keys))
        logging.info("Loaded XVLA-RMoE checkpoint")

        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance


__all__ = [
    "RoutingInfo",
    "XVLARMoEModel",
    "XVLARMoEPolicy",
    "compute_routing_delta_t",
]
