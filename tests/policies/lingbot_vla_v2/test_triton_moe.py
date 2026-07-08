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

"""The optional in-tree Triton grouped-GEMM MoE backend must match the pure-torch
grouped-by-expert eager path up to tensor-core (bf16 / tf32) MMA reassociation. Requires a
CUDA GPU with Triton; skips otherwise (CPU CI falls back to the eager path automatically)."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton grouped-MoE requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_grouped_moe_matches_eager(dtype):
    pytest.importorskip("triton")
    from lerobot.policies.lingbot_vla_v2.qwen2_action_expert import Qwen2FusedExperts
    from lerobot.policies.lingbot_vla_v2.triton_moe import triton_grouped_moe, triton_moe_available

    if not triton_moe_available():
        pytest.skip("triton unavailable")

    torch.manual_seed(0)
    num_experts, hidden, inter, top_k = 32, 128, 64, 4
    experts = Qwen2FusedExperts(num_experts, hidden, inter).to("cuda", dtype)

    T = 256
    hidden_states = torch.randn(T, hidden, device="cuda", dtype=dtype)
    scores = torch.randn(T, num_experts, device="cuda").sigmoid()
    routing_weights, selected_experts = torch.topk(scores, top_k, dim=-1)
    routing_weights = routing_weights.to(dtype)

    with torch.no_grad():
        fast = triton_grouped_moe(experts, routing_weights, selected_experts, hidden_states)
        ref = experts._eager_forward(routing_weights, selected_experts, hidden_states)

    assert fast.shape == ref.shape == (T, hidden)
    assert torch.isfinite(fast).all()
    # Tensor-core (bf16 / tf32) MMA precision — not bit-exact vs full-precision eager, but the
    # same class as the original triton kernel the checkpoint was benchmarked at.
    torch.testing.assert_close(fast.float(), ref.float(), atol=5e-2, rtol=5e-2)
