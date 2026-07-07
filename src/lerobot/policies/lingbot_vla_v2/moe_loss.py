# Copyright 2025 Ant Group Co., Ltd. All Rights Reserved.
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

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Filter out None entries (from non-MoE layers)
    gate_logits = tuple(g for g in gate_logits if g is not None)
    if len(gate_logits) == 0:
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(device=compute_device, dtype=torch.float32) for layer_gate in gate_logits], dim=0
    )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / (
            torch.sum(expert_attention_mask, dim=0) + 1e-8
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / (
            torch.sum(router_per_expert_attention_mask, dim=0) + 1e-8
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def sequence_wise_balance_loss(
    router_logits_list: tuple,
    top_k: int,
    seq_lengths: Optional[List[int]] = None,
    padding_len: int = 0,
    score_func: str = "softmax",
):
    if router_logits_list is None or not isinstance(router_logits_list, (tuple, list)):
        return []

    # Filter out None entries (from non-MoE layers)
    router_logits_list = [rl for rl in router_logits_list if rl is not None]
    if len(router_logits_list) == 0:
        return []

    layer_loss_list = []

    for logits in router_logits_list:
        # Cast to float32 for numerical stability
        logits = logits.to(dtype=torch.float32)
        N, E = logits.shape

        # Remove padding tokens
        if padding_len > 0:
            logits = logits[: N - padding_len]

        if logits.shape[0] == 0:
            continue

        if seq_lengths is not None and len(seq_lengths) > 0:
            # Split by sequence and compute per-sequence loss
            seq_logits_list = torch.split(logits, seq_lengths, dim=0)

            loss_per_seq = []
            for seq_logits in seq_logits_list:
                T_s = seq_logits.shape[0]
                if T_s == 0:
                    continue

                # P_i: mean routing probability per expert within each sequence
                if score_func == "sigmoid":
                    scores = seq_logits.sigmoid()
                    probs = scores / scores.sum(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(seq_logits, dim=-1)
                P_i = torch.mean(probs, dim=0)  # [E]

                # f_i: per-expert assignment frequency within each sequence (normalized)
                _, topk_indices = torch.topk(seq_logits, k=top_k, dim=-1)  # [T_s, top_k]
                mask = torch.zeros(T_s, E, device=seq_logits.device, dtype=torch.float32)
                mask.scatter_(1, topk_indices, 1.0)
                f_i = (E / top_k) * torch.mean(mask, dim=0)  # [E]

                # f_i.detach() stops gradients, backprop only flows through P_i
                loss_per_seq.append(torch.sum(f_i.detach() * P_i))

            if len(loss_per_seq) == 0:
                continue
            layer_loss_scalar = torch.stack(loss_per_seq).mean()
        else:
            # Treat all valid tokens as one sequence
            if score_func == "sigmoid":
                scores = logits.sigmoid()
                probs = scores / scores.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
            P_i = torch.mean(probs, dim=0)  # [E]

            _, topk_indices = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, topk_indices, 1.0)
            f_i = (E / top_k) * torch.mean(mask, dim=0)  # [E]

            layer_loss_scalar = torch.sum(f_i.detach() * P_i)

        layer_loss_list.append(layer_loss_scalar)

    return layer_loss_list
