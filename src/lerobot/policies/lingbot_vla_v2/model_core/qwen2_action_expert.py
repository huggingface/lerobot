import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Tuple
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import Cache, SlidingWindowCache, StaticCache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
    can_return_tuple,
    auto_docstring,
)
from transformers.utils.deprecation import deprecate_kwarg

logger = logging.get_logger(__name__)  # module logger (missing in upstream vendored file)
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack

try:
    from dinov3.hub.backbones import (
        dinov3_vits16,
        dinov3_vits16plus,
        dinov3_vitb16,
    )
except ImportError:
    dinov3_vits16 = dinov3_vits16plus = dinov3_vitb16 = None


def _update_moe_runtime_stats(block, routing_weights, selected_experts):
    """Update MoE runtime buffers outside torch.compile graphs."""
    with torch.no_grad():
        if routing_weights is not None and hasattr(block, "avg_topk_sigmoid_score"):
            avg_score = routing_weights.detach().float().mean()
            block.avg_topk_sigmoid_score.copy_(
                avg_score.reshape_as(block.avg_topk_sigmoid_score).to(
                    device=block.avg_topk_sigmoid_score.device,
                    dtype=block.avg_topk_sigmoid_score.dtype,
                )
            )

        if hasattr(block, "tokens_per_expert"):
            counts = F.one_hot(
                selected_experts.detach().reshape(-1),
                num_classes=block.num_experts,
            ).sum(dim=0)
            block.tokens_per_expert.add_(
                counts.to(
                    device=block.tokens_per_expert.device,
                    dtype=block.tokens_per_expert.dtype,
                )
            )


from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    Qwen2Attention,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    PreTrainedModel,
)

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model as _Qwen2Model,
    Qwen2ForCausalLM as _Qwen2ForCausalLM,
)

# from transformers.models.mistral.modeling_mistral import MistralMLP


# Modified from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2Moe
class Qwen2MoeRoutedExpertMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeSharedExpertMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2FusedExperts(nn.Module):
    """Fused expert module: stores E experts' weights as 3D tensors for group_gemm.

    Shape convention matches nn.Linear(in, out).weight = [out, in]:
      gate_proj: [E, intermediate_size, hidden_size]
      up_proj:   [E, intermediate_size, hidden_size]
      down_proj:  [E, hidden_size, intermediate_size]

    The forward() method runs the full fused_moe computation. This is critical
    for FSDP2: calling self.experts(...) triggers FSDP2's forward pre-hook to
    unshard the expert params on ep_fsdp_mesh BEFORE they are used by kernels.
    """

    def __init__(self, num_experts, hidden_size, intermediate_size, initializer_range=0.02):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.register_buffer("_gate_up_proj_cache", None, persistent=False)
        self._gate_up_proj_cache_key = None
        self.register_buffer("_dense_w1_cache", None, persistent=False)
        self.register_buffer("_dense_w2_cache", None, persistent=False)
        self._dense_cache_key = None
        self.register_buffer("_sparse_wd_cache", None, persistent=False)
        self._sparse_cache_key = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.up_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=self.initializer_range)
        self.clear_inference_cache()

    def clear_inference_cache(self):
        self._gate_up_proj_cache = None
        self._gate_up_proj_cache_key = None
        self._dense_w1_cache = None
        self._dense_w2_cache = None
        self._dense_cache_key = None
        self._sparse_wd_cache = None
        self._sparse_cache_key = None

    def _load_from_state_dict(self, *args, **kwargs):
        # Loading weights invalidates the packed inference caches.
        super()._load_from_state_dict(*args, **kwargs)
        self.clear_inference_cache()

    def _dense_packed_weights(self):
        """E-major repacked weights for the two-GEMM dense path.

        w1: [H, E*2I] — gate and up concatenated along the output dim, so a single
            ``x @ w1`` computes every expert's gate and up projections at once.
        w2: [E*I, H] — down projections concatenated along the input dim, so the
            second GEMM's reduction over E*I performs the (weight-folded) expert
            combine for free.

        Cached only when gradients are disabled (inference); under autograd the
        repack is rebuilt each call so gradients flow into the Parameters.
        """
        E, inter_dim, H = self.gate_proj.shape
        if torch.is_grad_enabled():
            w1 = torch.cat([self.gate_proj, self.up_proj], dim=1).reshape(E * 2 * inter_dim, H).t()
            w2 = self.down_proj.permute(0, 2, 1).reshape(E * inter_dim, H)
            return w1, w2
        # Key on the parameter versions: in-place updates (optimizer.step,
        # load_state_dict copy_) bump _version, so stale packed weights are
        # rebuilt instead of silently serving the old values.
        cache_key = (self.gate_proj._version, self.up_proj._version, self.down_proj._version)
        if self._dense_w1_cache is None or self._dense_cache_key != cache_key:
            with torch.no_grad():
                self._dense_w1_cache = (
                    torch.cat([self.gate_proj, self.up_proj], dim=1)
                    .reshape(E * 2 * inter_dim, H)
                    .t()
                    .contiguous()
                )
                self._dense_w2_cache = self.down_proj.permute(0, 2, 1).reshape(E * inter_dim, H).contiguous()
            self._dense_cache_key = cache_key
        return self._dense_w1_cache, self._dense_w2_cache

    def _dense_forward(self, routing_weights, selected_experts, hidden_states):
        """Dense two-GEMM MoE: compute ALL experts for ALL tokens, fold the top-k
        routing weights into the intermediate, and let the down GEMM's reduction
        perform the weighted expert combine.

        Algebraically identical to the grouped/eager paths (unselected experts get
        an exact 0 weight), differing only in floating-point reassociation. At the
        tiny token counts of flow-matching inference (T ~= chunk+1 = 51) the 8x
        extra FLOPs of computing every expert cost less than the routing machinery
        (argsort / gather / scatter / per-expert launches) they replace — and being
        two plain matmuls with static shapes, the whole block stays inside a single
        torch.compile graph instead of forcing a graph break per MoE layer.

        Routing weights are combined in fp32, matching the eager path's fp32
        accumulation; cast back to the input dtype at the end.
        """
        T, H = hidden_states.shape
        E, inter_dim = self.gate_proj.shape[0], self.gate_proj.shape[1]
        w1, w2 = self._dense_packed_weights()  # [H, E*2I], [E*I, H]

        gu = (hidden_states @ w1).view(T, E, 2 * inter_dim)
        inter = F.silu(gu[..., :inter_dim]) * gu[..., inter_dim:]  # [T, E, I]

        # One-hot the top-k routing weights back to a dense [T, E] table (0 for
        # unselected experts) and fold them into the intermediate activations.
        w = torch.zeros(T, E, dtype=torch.float32, device=hidden_states.device)
        w.scatter_(1, selected_experts, routing_weights.to(torch.float32))
        inter = inter * w.unsqueeze(-1).to(inter.dtype)

        return (inter.reshape(T, E * inter_dim) @ w2).to(hidden_states.dtype)

    def _sparse_packed_weights(self):
        """Per-expert packed weights for the padded sparse path.

        wgu: [E, H, 2I] — gate/up concatenated, transposed for bmm/grouped_mm.
        wd:  [E, I, H] — down projection transposed.
        Cached inference-only, keyed on parameter versions (same discipline as
        :meth:`_dense_packed_weights`); rebuilt each call under autograd.
        """
        if torch.is_grad_enabled():
            return (
                torch.cat([self.gate_proj, self.up_proj], dim=1).transpose(1, 2),
                self.down_proj.transpose(1, 2),
            )
        cache_key = (self.gate_proj._version, self.up_proj._version, self.down_proj._version)
        if self._gate_up_proj_cache is None or self._sparse_cache_key != cache_key:
            with torch.no_grad():
                self._gate_up_proj_cache = (
                    torch.cat([self.gate_proj, self.up_proj], dim=1).transpose(1, 2).contiguous()
                )
                self._sparse_wd_cache = self.down_proj.transpose(1, 2).contiguous()
            self._sparse_cache_key = cache_key
        return self._gate_up_proj_cache, self._sparse_wd_cache

    def _sparse_forward(self, routing_weights, selected_experts, hidden_states, use_grouped_mm=False, static_capacity=False):
        """Padded sparse MoE: real per-token expert activation with two fixed
        batched-GEMM launches.

        Sort the (token, expert) routing pairs by expert, scatter the routed
        token rows into a padded [E, Tm, H] tensor (Tm = max tokens routed to
        any single expert), run gate/up and down as one bmm — or grouped_mm 3D
        static where the running torch dispatches it to a real kernel — then
        gather the live rows back and accumulate the top-k weighted outputs in
        fp32 (matching the eager path's combine).

        Only the routed top-k rows ever touch expert weights, so FLOPs scale
        with top-k/E instead of E (unlike the dense path). The dynamic Tm
        needs one host sync per call (counts.max()), so this backend is
        eager-mode; CUDA-graph capture requires a static capacity.

        bmm is the universal GEMM backend (any arch, any torch); grouped_mm
        3D static is an opt-in (~2.11+ on sm89/sm121) that shares this padding.
        """
        T, H = hidden_states.shape
        E, inter_dim = self.gate_proj.shape[0], self.gate_proj.shape[1]
        top_k = selected_experts.shape[-1]

        flat_expert = selected_experts.reshape(-1)  # [T * top_k]
        flat_token = torch.arange(T, device=hidden_states.device).repeat_interleave(top_k)
        flat_weight = routing_weights.reshape(-1).to(torch.float32).unsqueeze(-1)

        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        sorted_token = flat_token[order]
        # Capture-safe expert histogram: torch.bincount internally does
        # input.max().item() (a host sync) to size its output, which aborts
        # CUDA-graph capture; scatter_add_ has no host round-trip.
        counts = torch.zeros(E, dtype=torch.long, device=hidden_states.device)
        counts.scatter_add_(0, flat_expert, torch.ones_like(flat_expert))
        if static_capacity:
            # Graph-safe upper bound: each token routes to top_k DISTINCT
            # experts, so no expert sees more than T pairs. Costs padded FLOPs
            # (up to dense-level at T ~= 51) but removes the host sync and the
            # dynamic shape, so the whole block is CUDA-graph capturable.
            Tm = T
        else:
            Tm = int(counts.max().item())  # dynamic capacity; provably <= T
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(T * top_k, device=hidden_states.device) - starts[sorted_expert]

        ap = torch.zeros(E, Tm, H, dtype=hidden_states.dtype, device=hidden_states.device)
        ap[sorted_expert, slot] = hidden_states[sorted_token]
        wgu, wd = self._sparse_packed_weights()  # [E,H,2I], [E,I,H]

        if use_grouped_mm:
            gu = torch.nn.functional.grouped_mm(ap, wgu)  # [E, Tm, 2I]
        else:
            gu = torch.bmm(ap, wgu)
        inter = F.silu(gu[..., :inter_dim]) * gu[..., inter_dim:]
        if use_grouped_mm:
            d = torch.nn.functional.grouped_mm(inter, wd)  # [E, Tm, H]
        else:
            d = torch.bmm(inter, wd)

        dp = d[sorted_expert, slot]  # [T * top_k, H] live rows only
        out = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
        out.index_add_(0, sorted_token, flat_weight[order] * dp.to(torch.float32))
        return out.to(hidden_states.dtype)

    def forward(self, module, num_experts, routing_weights, selected_experts, hidden_states):
        """Run the fused experts with FSDP2-managed weights.

        Must be called via self.experts(...) so FSDP2 unshards params first. Backends
        (module._moe_backend, numerically equivalent up to floating-point /
        tensor-core reassociation):
          - "sparse_static" (shipped default): :meth:`_sparse_forward` with the
            padded capacity pinned to T — real per-token expert activation with
            static shapes (CUDA-graph capturable, no host sync, fastest
            training step time);
          - "sparse": same, but with dynamic capacity (one .item() sync —
            breaks CUDA graph capture and stalls the training pipeline);
          - "sparse_gmm" / "sparse_static_gmm": same padding with grouped_mm
            3D as the GEMM (requires a torch that dispatches it on this arch;
            sm89/sm121 ~2.11+; measured e2e-equal to bmm);
          - "auto": dense two-GEMM for small token counts (flow-matching
            denoise: T ~= 51), grouped-by-expert eager for everything else;
          - "dense": force the dense two-GEMM path (-1.5~2.7% model-only
            inference time vs sparse_static, +2.7% training step time);
          - "eager": grouped-by-expert eager fallback (CPU / large T / training).
        """
        backend = getattr(module, "_moe_backend", "auto")
        if backend == "sparse":
            return self._sparse_forward(routing_weights, selected_experts, hidden_states)
        if backend == "sparse_static":
            return self._sparse_forward(routing_weights, selected_experts, hidden_states, static_capacity=True)
        if backend == "sparse_static_gmm":
            return self._sparse_forward(
                routing_weights, selected_experts, hidden_states, use_grouped_mm=True, static_capacity=True
            )
        if backend == "sparse_gmm":
            return self._sparse_forward(routing_weights, selected_experts, hidden_states, use_grouped_mm=True)

        # dense two-GEMM path for small token counts (flow-matching denoise:
        # T ~= 51). Pure torch, static shapes, no graph breaks under torch.compile.
        dense_max_tokens = getattr(module, "_dense_max_tokens", 512)
        if backend in ("auto", "dense") and dense_max_tokens > 0 and hidden_states.shape[0] <= dense_max_tokens:
            return self._dense_forward(routing_weights, selected_experts, hidden_states)

        # pure-torch grouped-by-expert eager fallback (CPU / large T / training).
        return self._eager_forward(routing_weights, selected_experts, hidden_states)

    def _eager_forward(self, routing_weights, selected_experts, hidden_states):
        """Grouped-by-EXPERT eager MoE over the stacked (num_experts, ...) weights.

        For each expert we gather the tokens routed to it and run a single dense matmul
        (gate/up: ``[I, H]``; down: ``[H, I]``) — the expert weight is loaded once and reused
        across its tokens. This replaces the naive per-token form (which materialized a
        ``[T, I, H]`` weight copy per route, i.e. O(T) activation memory and O(T) matmuls).
        It is algebraically identical (the same per-(token, route) SwiGLU terms, only
        reordered/reassociated); cost is O(num_experts) matmuls with O(T) activations. Pure
        torch, works on any backend. gate/up: ``[E, I, H]``; down: ``[E, H, I]``.
        """
        T, H = hidden_states.shape
        num_experts = self.gate_proj.shape[0]
        top_k = selected_experts.shape[-1]
        out = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)

        # Flatten the (token, route) routing table so each expert can pull its rows.
        flat_expert = selected_experts.reshape(-1)  # [T * top_k]
        flat_token = torch.arange(T, device=hidden_states.device).repeat_interleave(top_k)
        flat_weight = routing_weights.reshape(-1).to(torch.float32).unsqueeze(-1)  # [T*top_k, 1]

        # Sort routes by expert once — a single host sync for the split sizes —
        # instead of a per-expert torch.nonzero (one device sync per expert, i.e.
        # num_experts syncs per MoE layer per forward).
        order = torch.argsort(flat_expert)
        counts = torch.bincount(flat_expert, minlength=num_experts).tolist()
        sorted_token = flat_token[order]
        sorted_weight = flat_weight[order]

        offset = 0
        for e in range(num_experts):
            n_e = counts[e]
            if n_e == 0:
                continue
            tok = sorted_token[offset : offset + n_e]
            xe = hidden_states[tok]  # [n_e, H] — tokens routed to expert e
            gate = xe @ self.gate_proj[e].t()  # [n_e, I]
            up = xe @ self.up_proj[e].t()  # [n_e, I]
            ye = (F.silu(gate) * up) @ self.down_proj[e].t()  # [n_e, H]
            out.index_add_(0, tok, sorted_weight[offset : offset + n_e] * ye.to(torch.float32))
            offset += n_e
        return out.to(hidden_states.dtype)


class FixQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        FixQwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2TokenMoeBlock(nn.Module):
    """Token-level routing MoE block with all-to-all computation for torch.compile compatibility."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Loss-free balancing support. With zero correction bias this is
        # equivalent to unbiased top-k selection; the optimizer pre-hook updates
        # the bias when bias_update_speed > 0.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts),
            persistent=True,
        )
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "avg_topk_sigmoid_score",
            torch.zeros(1, dtype=torch.float32),
            persistent=False,
        )

        # gating (per-token)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # Token-count ceiling for the dense two-GEMM MoE path (0 disables it and
        # falls through to the grouped-eager backend).
        self._dense_max_tokens = getattr(config, "moe_dense_max_tokens", 512)
        # MoE execution backend override: "sparse_static" (shipped default,
        # from LingBotVLAV2Config.moe_backend) | "auto" | "sparse" (padded
        # bmm) | "sparse_gmm" (padded + grouped_mm 3D) | "dense" | "eager".
        self._moe_backend = getattr(config, "moe_backend", "sparse_static")

        # EP/fused support: choose expert storage based on moe_implementation
        self._moe_implementation = getattr(config, "_moe_implementation", None) or "eager"
        if self._moe_implementation == "fused":
            self.experts = Qwen2FusedExperts(
                self.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
                initializer_range=getattr(config, "initializer_range", 0.02),
            )
        else:
            self.experts = nn.ModuleList(
                [
                    Qwen2MoeRoutedExpertMLP(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        self.shared_expert = Qwen2MoeSharedExpertMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self._router_activation = getattr(config, "router_activation", "softmax")
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self._use_shared_expert_gate = getattr(config, "use_shared_expert_gate", True)
        if self._use_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Token-level routing with all-to-all computation for torch.compile compatibility."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length

        # Token-level routing: each token individually
        hidden_flat = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)
        # Gate in true fp32 (autocast disabled): bf16 gate logits can flip top-k
        # selection on near-equal scores -> routing jitter / rotating dead experts.
        # cf. VideoPretrain lumos/moe/router.py TokenChoiceTopKRouter.
        with torch.amp.autocast(hidden_flat.device.type, enabled=False):
            router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())  # (B*T, num_experts)

        if self._router_activation == "sigmoid":
            routing_scores = router_logits.sigmoid()
        else:
            routing_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

        scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(1, selected_experts)
        if self.training:
            _update_moe_runtime_stats(self, routing_weights, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor != 1.0:
            routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Expert computation: dense two-GEMM (small T) / grouped eager (fallback)
        if self._moe_implementation == "fused":
            final_hidden_states = self.experts(
                module=self,
                num_experts=self.num_experts,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                hidden_states=hidden_flat,
            )
        else:
            # Original eager path: every expert processes all tokens
            expert_outputs = torch.stack(
                [expert(hidden_flat) for expert in self.experts], dim=0
            )  # (num_experts, B*T, D)
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts
            ).float()  # (B*T, top_k, num_experts)
            weights = (
                (expert_mask * routing_weights.unsqueeze(-1).float()).sum(dim=1).to(hidden_states.dtype)
            )  # (B*T, num_experts)
            final_hidden_states = torch.einsum("ebd,be->bd", expert_outputs, weights)  # (B*T, D)

        # Shared expert: applied to all tokens (fixed shape)
        if final_hidden_states.dtype != hidden_flat.dtype:
            final_hidden_states = final_hidden_states.to(hidden_flat.dtype)
        shared_expert_output = self.shared_expert(hidden_flat)
        if self._use_shared_expert_gate:
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_flat)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor | None = None,
        start: int | None = 0,
        end: int | None = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        ada_cond: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # Ensure input dtypes match weight dtype (needed for gradient checkpointing
        # recomputation where autocast context is lost)
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)
        if ada_cond is not None:
            ada_cond = ada_cond.to(param_dtype)

        if compute_kqv:
            if ada_cond is not None:
                hidden_states = self.input_layernorm(hidden_states, ada_cond)
            else:
                hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

            query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)

            return query_state, key_state, value_state

        elif output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])

            # first residual
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            if ada_cond is not None:
                out_emb = self.post_attention_layernorm(out_emb, ada_cond)
            else:
                out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            # Handle MoE block returning (hidden_states, router_logits)
            router_logits = None
            if isinstance(out_emb, tuple):
                out_emb, router_logits = out_emb
            # second residual
            out_emb += after_first_residual

            return out_emb, router_logits

        else:
            raise ValueError(
                f"Invalid Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2DecoderLayer in LingBot-VLA"
            )


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2FusedExperts):
            module.initializer_range = std
            module.reset_parameters()


class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    get_input_embeddings = _Qwen2Model.get_input_embeddings
    set_input_embeddings = _Qwen2Model.set_input_embeddings

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This vendored Qwen2Model is only driven by LingBot-VLA internals "
            "(QwenvlWithExpert calls its layers with the custom compute_kqv/"
            "output_atten signature); the HF-native forward is intentionally "
            "not supported."
        )

    def __init__(self, config: Qwen2Config, eval=False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = FixQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        if eval:
            self._init_weights = lambda module: None
        self.post_init()


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    # transformers>=5.5 expects a dict {tied_key: source_key} (was a list in 4.57).
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    get_input_embeddings = _Qwen2ForCausalLM.get_input_embeddings
    set_input_embeddings = _Qwen2ForCausalLM.set_input_embeddings
    get_output_embeddings = _Qwen2ForCausalLM.get_output_embeddings
    set_output_embeddings = _Qwen2ForCausalLM.set_output_embeddings
    set_decoder = _Qwen2ForCausalLM.set_decoder
    get_decoder = _Qwen2ForCausalLM.get_decoder

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This vendored Qwen2ForCausalLM is only driven by LingBot-VLA internals "
            "(QwenvlWithExpert calls its layers with the custom compute_kqv/"
            "output_atten signature); the HF-native forward is intentionally "
            "not supported."
        )

    def __init__(self, config, eval):
        super().__init__(config)
        self.model = Qwen2Model(config, eval)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
