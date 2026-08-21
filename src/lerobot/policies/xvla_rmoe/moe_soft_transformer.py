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

"""X-VLA Transformer with a Cross-Step Routing Memory dense soft Mixture-of-Experts FFN.

This mirrors the role `smolvlm_with_expert.py`'s `MoEFFN` plays for `smolvla_rmoe`, but is
built directly against X-VLA's own `SoftPromptedTransformer` (see
`lerobot.policies.xvla.soft_transformer`) instead of copying SmolVLA's gated-MLP /
prefix-suffix-with-KV-cache design, which does not apply here:

  * X-VLA's FFN is a plain 2-layer `Mlp` (`fc1 -> GELU -> fc2`), not a Llama-style
    gated `{gate,up,down}_proj` MLP.
  * X-VLA runs ONE homogeneous Transformer over a single concatenated sequence
    `[action tokens | VLM tokens | auxiliary-view tokens | soft prompts]` (action tokens
    are always the first `chunk_size` positions -- see `SoftPromptedTransformer.forward`)
    rather than two separate VLM/expert token streams attending cross-attention-style with
    a KV cache. So there is no prefix/suffix split to reuse here: every MoE layer sees
    both context tokens and action tokens in the same call, and the router/summary/position
    embedding must explicitly restrict themselves to the action-token sub-range.
  * X-VLA has no attention/padding mask convention at all (full dense self-attention;
    missing camera views are zero-filled, not masked) -- this file does not add one, only
    an optional *action-token* validity mask for MoE routing/hidden summaries and for
    truncated recurrent training with padded action chunks.

Dense soft MoE only: every expert runs on every token and outputs are combined with
softmax routing weights (no top-K / sparse routing, no conditional compute).
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ..xvla.soft_transformer import (
    DomainAwareLinear,
    TransformerBlock,
    basic_init,
    timestep_embedding,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    """Mean over `dim`, ignoring positions where `mask` is False (`True` == valid token).

    Falls back to a plain mean when `mask` is None, so every call site that doesn't pass a
    mask (e.g. inference, where every action-chunk position is a real prediction) keeps its
    original, unmasked numerics unchanged.
    """
    if mask is None:
        return values.mean(dim=dim)
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    masked_sum = (values * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return masked_sum / denom


def compute_routing_delta_t(current_t, previous_t):
    """Shared Δt definition (|t_current - t_previous|) used identically by the inference
    denoising loop and truncated recurrent training, for both Python floats (inference
    tracks `time` as a scalar) and tensors (recurrent training tracks `t` per-batch)."""
    return abs(current_t - previous_t)


def chunk_position_embedding(num_actions: int, dim: int, device: torch.device) -> Tensor:
    """Normalised action-chunk position embedding `e(i / max(L-1, 1))`, shape (num_actions, dim)."""
    positions = torch.arange(num_actions, device=device, dtype=torch.float32) / max(num_actions - 1, 1)
    return timestep_embedding(positions, dim)


class MoEFFN(nn.Module):
    """Dense soft-MoE replacement for one X-VLA Transformer block's FFN (`Mlp`).

    All `num_experts` experts run on every token; outputs are combined with softmax routing
    weights -- `y = sum_e g_e * Expert_e(x)`, never top-K / sparse.

    The router optionally sees (all toggles are static per-instance, driven by
    `XVLARMoEConfig`):
      * the current token hidden state `x`                                (always)
      * the cross-step GRU routing memory                                  (`use_routing_memory`)
      * a sinusoidal embedding of the current denoising timestep           (`use_timestep_router`)
      * a sinusoidal action-chunk position embedding (zero for non-action
        tokens -- see `SoftPromptedTransformerRMoE.forward`)               (`use_chunk_position_embedding`)

    Initialisation: every expert is a deep copy of `original_ffn` and the router is
    zero-initialised, so softmax produces uniform weights and the initial output equals
    `original_ffn(x)` exactly when `expert_symmetry_breaking_std == 0`.

    Symmetry breaking: with bit-identical experts and a zero-init router, every expert gets
    the same input, the same (uniform) routing weight, and therefore the same gradient every
    step -- they never diverge. Because the mixture output is then exactly invariant to the
    routing weights themselves (`sum_e w_e * f(x) = f(x)`), the router (and anything feeding
    it, including the cross-step GRU) never receives a real gradient either. A tiny
    independent noise draw per expert at init breaks this permutation symmetry while leaving
    step-0 output effectively unchanged (see `_break_expert_symmetry`).
    """

    def __init__(
        self,
        original_ffn: nn.Module,
        num_experts: int,
        routing_hidden_dim: int,
        routing_timestep_dim: int,
        chunk_pos_emb_dim: int,
        use_routing_memory: bool,
        use_timestep_router: bool,
        use_chunk_position_embedding: bool,
        expert_symmetry_breaking_std: float = 1e-5,
    ):
        super().__init__()
        hidden_dim = original_ffn.fc1.in_features
        self.num_experts = num_experts
        self.use_routing_memory = use_routing_memory
        self.use_timestep_router = use_timestep_router
        self.use_chunk_position_embedding = use_chunk_position_embedding

        self.experts = nn.ModuleList([copy.deepcopy(original_ffn) for _ in range(num_experts)])
        self._break_expert_symmetry(expert_symmetry_breaking_std)

        router_in = hidden_dim
        if use_routing_memory:
            router_in += routing_hidden_dim
        if use_timestep_router:
            router_in += routing_timestep_dim
        if use_chunk_position_embedding:
            router_in += chunk_pos_emb_dim
        # Zero-init -> uniform routing (= original FFN output) at symmetry_breaking_std=0.
        self.router = nn.Linear(router_in, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)

    @torch.no_grad()
    def _break_expert_symmetry(self, std: float) -> None:
        """Nudge each expert's parameters by independent tiny Gaussian noise so they stop
        being bit-identical copies of `original_ffn`. `std<=0` disables it (ablation/debug
        only -- see `expert_symmetry_breaking_std` docs in `configuration_xvla_rmoe.py`)."""
        if std <= 0.0:
            return
        for expert in self.experts:
            for param in expert.parameters():
                if param.is_floating_point():
                    param.add_(torch.randn_like(param) * std)

    def forward(
        self,
        x: torch.Tensor,
        routing_state: torch.Tensor | None,
        timestep_emb: torch.Tensor | None,
        chunk_pos_emb: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
        return_full_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (B, L, D) hidden states for the *full* sequence (context + action tokens).
            routing_state: (B, routing_hidden_dim) GRU state, or None if disabled.
            timestep_emb: (B, routing_timestep_dim) sinusoidal embedding of the current
                denoising timestep, or None if disabled.
            chunk_pos_emb: (B, L, chunk_pos_emb_dim) action-chunk position embedding, zero
                for non-action tokens, or None if disabled.
            token_mask: optional (B, L) mask (`True` == valid action token, `False` for both
                padded action tokens and non-action context tokens) used to restrict the
                returned routing decision to valid action tokens only. None reproduces the
                original unmasked mean (used at inference, where every action-chunk position
                is a real prediction).
            return_full_weights: if True, also return the raw per-token (B, L, E) routing
                weights -- analysis/eval only, never set on the training hot path (see
                `RoutingInfo.layer_routing_weights`).

        Returns:
            output: (B, L, D) weighted mixture of expert outputs for the full sequence.
            routing_decision: (B, num_experts) mean routing weights over valid action tokens.
            full_weights: (B, L, num_experts) raw per-token routing weights, or None.
        """
        batch_size, seq_len, _ = x.shape
        router_dtype = self.router.weight.dtype
        router_inputs = [x.to(router_dtype)]
        if self.use_routing_memory:
            router_inputs.append(routing_state.to(router_dtype)[:, None, :].expand(batch_size, seq_len, -1))
        if self.use_timestep_router:
            router_inputs.append(timestep_emb.to(router_dtype)[:, None, :].expand(batch_size, seq_len, -1))
        if self.use_chunk_position_embedding:
            router_inputs.append(chunk_pos_emb.to(router_dtype))

        logits = self.router(torch.cat(router_inputs, dim=-1))  # (B, L, E)
        weights = logits.softmax(dim=-1)

        expert_dtype = self.experts[0].fc1.weight.dtype
        expert_outs = torch.stack([e(x.to(expert_dtype)) for e in self.experts], dim=2)  # (B, L, E, D)
        output = (weights.to(expert_dtype).unsqueeze(-1) * expert_outs).sum(dim=2).to(x.dtype)

        routing_decision = masked_mean(weights, token_mask, dim=1)  # (B, E)
        full_weights = weights.detach() if return_full_weights else None
        return output, routing_decision, full_weights


class RoutingMemoryCell(nn.Module):
    """GRU cell carrying the cross-step routing state across denoising steps.

    GRU input each step:
      [routing_decisions | hidden_proj(masked_mean(z_final)) | e(t) | e(Δt)]
        - routing_decisions: mean gating weights from all MoEFFN layers (num_moe_layers * num_experts)
        - hidden_proj output: projected masked-mean action-token hidden state (routing_hidden_dim)
        - e(t):  sinusoidal timestep embedding                                  (routing_timestep_dim)
        - e(Δt): sinusoidal timestep-gap embedding, only if `use_delta_t_conditioning`

    Output: updated state (B, routing_hidden_dim) fed into the next step's MoEFFN routers.
    """

    def __init__(
        self,
        num_moe_layers: int,
        num_experts: int,
        routing_hidden_dim: int,
        routing_timestep_dim: int,
        hidden_size: int,
        use_delta_t_conditioning: bool = True,
    ):
        super().__init__()
        self.routing_hidden_dim = routing_hidden_dim
        self.use_delta_t_conditioning = use_delta_t_conditioning
        self.hidden_proj = nn.Linear(hidden_size, routing_hidden_dim)
        gru_input_dim = (
            num_moe_layers * num_experts
            + routing_hidden_dim
            + routing_timestep_dim * (2 if use_delta_t_conditioning else 1)
        )
        self.gru_cell = nn.GRUCell(gru_input_dim, routing_hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.routing_hidden_dim, device=device)

    def forward(
        self,
        routing_decisions: list[Tensor],
        hidden_summary: Tensor,
        timestep_emb: Tensor,
        dt_emb: Tensor | None,
        prev_state: Tensor,
    ) -> Tensor:
        hp_dtype = self.hidden_proj.weight.dtype
        hidden_part = self.hidden_proj(hidden_summary.to(dtype=hp_dtype))

        gru_dtype = next(self.gru_cell.parameters()).dtype
        parts = [
            torch.cat(routing_decisions, dim=-1).to(dtype=gru_dtype),
            hidden_part.to(dtype=gru_dtype),
            timestep_emb.to(dtype=gru_dtype),
        ]
        if self.use_delta_t_conditioning:
            parts.append(dt_emb.to(dtype=gru_dtype))
        gru_input = torch.cat(parts, dim=-1)
        return self.gru_cell(gru_input, prev_state.to(dtype=gru_dtype))


class SoftPromptedTransformerRMoE(nn.Module):
    """`SoftPromptedTransformer` (see `lerobot.policies.xvla.soft_transformer`) with a
    configurable subset of Transformer-block FFNs replaced by `MoEFFN`.

    Attention, LayerNorm placement, and residual ordering are untouched (both are reused
    unmodified from `..xvla.soft_transformer`) -- only the FFN call inside the selected
    blocks is replaced. Non-MoE blocks run exactly `TransformerBlock.forward` unchanged, so
    with `moe_layer_indices=[]` this class is numerically identical to the original
    `SoftPromptedTransformer` given the same weights and inputs.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        moe_layer_indices: list[int] | None = None,
        num_moe_experts: int = 4,
        routing_hidden_dim: int = 64,
        routing_timestep_dim: int = 64,
        chunk_pos_emb_dim: int = 32,
        use_routing_memory: bool = True,
        use_timestep_router: bool = True,
        use_chunk_position_embedding: bool = True,
        expert_symmetry_breaking_std: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj
        self.chunk_pos_emb_dim = chunk_pos_emb_dim
        self.use_routing_memory = use_routing_memory
        self.use_timestep_router = use_timestep_router
        self.use_chunk_position_embedding = use_chunk_position_embedding

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(
                multi_modal_input_size, hidden_size, num_domains=num_domains
            )
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        # Original X-VLA init first (Xavier on every plain nn.Linear, including each
        # block's still-plain `Mlp`), *then* wrap the selected FFNs into MoEFFN below so
        # every expert deep-copies a properly (not default-random) initialized FFN --
        # required for the near-function-preserving init test.
        self.apply(basic_init)

        moe_layer_indices = moe_layer_indices or []
        bad = [i for i in moe_layer_indices if i < 0 or i >= depth]
        if bad:
            raise ValueError(f"`moe_layer_indices` contains out-of-range layer indices {bad}.")
        self.moe_layer_indices = list(moe_layer_indices)
        self.num_moe_layers = len(self.moe_layer_indices)
        for layer_idx in self.moe_layer_indices:
            original_ffn = self.blocks[layer_idx].mlp
            self.blocks[layer_idx].mlp = MoEFFN(
                original_ffn,
                num_experts=num_moe_experts,
                routing_hidden_dim=routing_hidden_dim,
                routing_timestep_dim=routing_timestep_dim,
                chunk_pos_emb_dim=chunk_pos_emb_dim,
                use_routing_memory=use_routing_memory,
                use_timestep_router=use_timestep_router,
                use_chunk_position_embedding=use_chunk_position_embedding,
                expert_symmetry_breaking_std=expert_symmetry_breaking_std,
            )

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        routing_state: torch.Tensor | None = None,
        timestep_emb: torch.Tensor | None = None,
        action_padding_mask: torch.Tensor | None = None,
        return_full_routing_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[torch.Tensor] | None]:
        """
        Same inputs/output-shape contract as `SoftPromptedTransformer.forward` (see that
        class for the full docstring), plus:
            routing_state: (B, routing_hidden_dim) or None -- forwarded to every MoEFFN.
            timestep_emb: (B, routing_timestep_dim) or None -- forwarded to every MoEFFN.
            action_padding_mask: optional (B, num_actions) bool mask, `True` == valid
                (non-padding) action token. Only meaningful during truncated recurrent
                training (some chunk positions can be past the episode end); None (default,
                used by inference) reproduces the original unmasked pooling exactly.
            return_full_routing_weights: analysis/eval only -- see `RoutingInfo`. Never set
                on the training hot path.

        Returns:
            pred_action: (B, num_actions, dim_action), identical contract to the original.
            routing_decisions: list of (B, num_experts) mean gating weights, one per MoE
                layer (empty if this transformer has no MoE layers).
            hidden_summary: (B, hidden_size) masked-mean-pooled action-token hidden state
                from the final block output (float32), used by `RoutingMemoryCell`.
            layer_routing_weights: list of raw (B, L, num_experts) per-token routing
                weights, one per MoE layer, or None unless `return_full_routing_weights`.
        """
        batch_size, num_actions = action_with_noise.shape[:2]

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(batch_size, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(batch_size, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)

        if self.use_hetero_proj:
            x = torch.cat(
                [
                    x,
                    self.vlm_proj(vlm_features, domain_id),
                    self.aux_visual_proj(aux_visual_inputs, domain_id),
                ],
                dim=1,
            )
        else:
            x = torch.cat([x, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]

        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(
                batch_size, self.len_soft_prompts, self.hidden_size
            )
            x = torch.cat([x, soft_prompts], dim=1)

        full_seq_len = x.shape[1]

        # Action tokens are always the first `num_actions` positions (see the concatenation
        # order above). Build the MoE routing helpers once, for the whole sequence, shared
        # by every MoE layer below.
        token_mask = None
        chunk_pos_emb = None
        if self.num_moe_layers > 0:
            action_valid = (
                action_padding_mask
                if action_padding_mask is not None
                else torch.ones(batch_size, num_actions, dtype=torch.bool, device=x.device)
            )
            token_mask = x.new_zeros((batch_size, full_seq_len), dtype=torch.bool)
            token_mask[:, :num_actions] = action_valid

            if self.use_chunk_position_embedding and self.chunk_pos_emb_dim > 0:
                action_pe = chunk_position_embedding(num_actions, self.chunk_pos_emb_dim, x.device)
                pe_full = x.new_zeros((full_seq_len, self.chunk_pos_emb_dim))
                pe_full[:num_actions] = action_pe.to(dtype=pe_full.dtype)
                chunk_pos_emb = pe_full.unsqueeze(0).expand(batch_size, -1, -1)

        routing_decisions: list[torch.Tensor] = []
        layer_routing_weights: list[torch.Tensor] | None = [] if return_full_routing_weights else None
        for block in self.blocks:
            x = x + block.attn(block.norm1(x))
            normed = block.norm2(x)
            if isinstance(block.mlp, MoEFFN):
                mlp_out, routing_decision, full_weights = block.mlp(
                    normed,
                    routing_state=routing_state,
                    timestep_emb=timestep_emb,
                    chunk_pos_emb=chunk_pos_emb,
                    token_mask=token_mask,
                    return_full_weights=return_full_routing_weights,
                )
                routing_decisions.append(routing_decision)
                if layer_routing_weights is not None:
                    layer_routing_weights.append(full_weights)
            else:
                mlp_out = block.mlp(normed)
            x = x + mlp_out

        final_action_hidden = self.norm(x[:, :num_actions])
        action_valid_for_summary = token_mask[:, :num_actions] if token_mask is not None else None
        hidden_summary = masked_mean(final_action_hidden, action_valid_for_summary, dim=1).to(
            dtype=torch.float32
        )

        pred_action = self.action_decoder(final_action_hidden, domain_id)
        return pred_action, routing_decisions, hidden_summary, layer_routing_weights
