# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Optional in-tree Triton grouped-GEMM backend for the lingbot_vla_v2 sparse-MoE expert.

Sorts the ``(token, route)`` pairs by expert so each expert's tokens are contiguous, then runs
a grouped GEMM (one dense matmul per expert over its token block, weight reused) for the gate,
up and down projections, and scatters the routed-weighted result back. This is the same
algorithm the original Triton ``fused_moe`` uses, and matches the pure-torch grouped-by-expert
eager path up to tensor-core (bf16 / tf32) MMA reassociation.

Fully guarded: ``triton_moe_available()`` is False when Triton is missing (e.g. CPU / macOS /
ARM wheels without it), so callers fall back to the eager path. Triton ships transitively with
CUDA PyTorch on Linux x86_64, so this needs no extra dependency or build step.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - any import failure -> feature simply unavailable
    _HAS_TRITON = False


def triton_moe_available() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _grouped_gemm_kernel(
        x_ptr,  # [M, K]  tokens sorted by expert
        w_ptr,  # [E, N, K]  stacked nn.Linear weights (out=N, in=K)
        out_ptr,  # [M, N]
        cumsum_ptr,  # [E + 1] int32 prefix sums of tokens-per-expert
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        start = tl.load(cumsum_ptr + e).to(tl.int64)
        end = tl.load(cumsum_ptr + e + 1).to(tl.int64)
        m_size = end - start
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # within-expert row offset
        row_valid = offs_m < m_size
        rows = start + offs_m.to(tl.int64)  # global sorted-row index

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            a = tl.load(
                x_ptr + rows[:, None] * K + kk[None, :],
                mask=row_valid[:, None] & (kk[None, :] < K),
                other=0.0,
            )  # [BLOCK_M, BLOCK_K]
            w = tl.load(
                w_ptr + e * N * K + offs_n[:, None] * K + kk[None, :],
                mask=(offs_n[:, None] < N) & (kk[None, :] < K),
                other=0.0,
            )  # [BLOCK_N, BLOCK_K]
            acc += tl.dot(a, tl.trans(w))  # [BLOCK_M, BLOCK_N]

        out = acc.to(out_ptr.dtype.element_ty)
        tl.store(
            out_ptr + rows[:, None] * N + offs_n[None, :],
            out,
            mask=row_valid[:, None] & (offs_n[None, :] < N),
        )

    def _grouped_gemm(x, weight, cumsum, max_m, N, K):
        """x: [M, K] sorted by expert; weight: [E, N, K]; cumsum: [E+1] int32. -> [M, N]."""
        M = x.shape[0]
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
        if M == 0 or max_m == 0:
            return out
        E = weight.shape[0]
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (E, triton.cdiv(max_m, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _grouped_gemm_kernel[grid](
            x, weight, out, cumsum, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out


def triton_grouped_moe(experts, routing_weights, selected_experts, hidden_states):
    """Grouped-GEMM MoE over the stacked weights (gate/up: [E, I, H]; down: [E, H, I]).

    Returns the routed-weighted expert output ``[T, H]`` in ``hidden_states.dtype``.
    """
    import torch.nn.functional as F

    T, H = hidden_states.shape
    E, inter_dim, _ = experts.gate_proj.shape
    top_k = selected_experts.shape[-1]

    flat_expert = selected_experts.reshape(-1)
    flat_token = torch.arange(T, device=hidden_states.device).repeat_interleave(top_k)
    flat_weight = routing_weights.reshape(-1).to(torch.float32)

    order = torch.argsort(flat_expert)
    sorted_token = flat_token[order]
    x_sorted = hidden_states[sorted_token].contiguous()  # [M, H]

    counts = torch.bincount(flat_expert, minlength=E)
    cumsum = torch.zeros(E + 1, dtype=torch.int32, device=hidden_states.device)
    cumsum[1:] = counts.cumsum(0).to(torch.int32)
    max_m = int(counts.max().item())

    gate = _grouped_gemm(x_sorted, experts.gate_proj.contiguous(), cumsum, max_m, inter_dim, H)
    up = _grouped_gemm(x_sorted, experts.up_proj.contiguous(), cumsum, max_m, inter_dim, H)
    inter = (F.silu(gate) * up).contiguous()
    down = _grouped_gemm(inter, experts.down_proj.contiguous(), cumsum, max_m, H, inter_dim)  # [M, H]

    out = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    out.index_add_(0, sorted_token, flat_weight[order].unsqueeze(-1) * down.to(torch.float32))
    return out.to(hidden_states.dtype)
