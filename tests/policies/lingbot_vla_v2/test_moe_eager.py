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

"""CPU test that the grouped-by-expert eager MoE is algebraically equivalent to the naive
per-token reference (so swapping in the faster, O(1)-activation-memory expert loop does not
change the pretrained model's outputs beyond floating-point reassociation)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


def _per_token_reference(experts, routing_weights, selected_experts, hidden_states):
    """The previous per-token eager MoE (materializes [T, I, H] weights per route)."""
    T, H = hidden_states.shape
    top_k = selected_experts.shape[-1]
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    x = hidden_states.unsqueeze(1)
    for k in range(top_k):
        eidx = selected_experts[:, k]
        w = routing_weights[:, k].to(torch.float32).unsqueeze(-1)
        g = experts.gate_proj[eidx]
        u = experts.up_proj[eidx]
        d = experts.down_proj[eidx]
        go = torch.bmm(x, g.transpose(1, 2)).squeeze(1)
        uo = torch.bmm(x, u.transpose(1, 2)).squeeze(1)
        inter = (F.silu(go) * uo).unsqueeze(1)
        y = torch.bmm(inter, d.transpose(1, 2)).squeeze(1)
        out = out + w * y.to(torch.float32)
    return out.to(hidden_states.dtype)


def test_grouped_expert_eager_matches_per_token_reference():
    from lerobot.policies.lingbot_vla_v2.qwen2_action_expert import Qwen2FusedExperts

    torch.manual_seed(0)
    num_experts, hidden, inter, top_k = 6, 16, 8, 2
    experts = Qwen2FusedExperts(num_experts, hidden, inter)

    T = 40
    hidden_states = torch.randn(T, hidden)
    scores = torch.randn(T, num_experts).sigmoid()
    routing_weights, selected_experts = torch.topk(scores, top_k, dim=-1)

    grouped = experts._eager_forward(routing_weights, selected_experts, hidden_states)
    reference = _per_token_reference(experts, routing_weights, selected_experts, hidden_states)

    assert grouped.shape == (T, hidden)
    # Algebraically identical up to float32 reassociation.
    torch.testing.assert_close(grouped, reference, atol=1e-5, rtol=1e-4)


def test_grouped_expert_eager_handles_unused_experts():
    """Experts with no routed tokens must be skipped without error and not contribute."""
    from lerobot.policies.lingbot_vla_v2.qwen2_action_expert import Qwen2FusedExperts

    torch.manual_seed(1)
    num_experts, hidden, inter, top_k = 8, 16, 8, 1
    experts = Qwen2FusedExperts(num_experts, hidden, inter)

    T = 5
    hidden_states = torch.randn(T, hidden)
    # Route every token to expert 0 only -> experts 1..7 are unused.
    selected_experts = torch.zeros(T, top_k, dtype=torch.long)
    routing_weights = torch.ones(T, top_k)

    out = experts._eager_forward(routing_weights, selected_experts, hidden_states)
    reference = _per_token_reference(experts, routing_weights, selected_experts, hidden_states)
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-4)
