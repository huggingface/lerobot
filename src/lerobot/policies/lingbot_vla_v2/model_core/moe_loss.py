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


def sequence_wise_balance_loss(
    router_logits_list: tuple,
    top_k: int,
    seq_lengths: list[int] | None = None,
    padding_len: int = 0,
    score_func: str = "softmax",
    e_score_correction_bias_list: list | tuple | None = None,
):
    if router_logits_list is None or not isinstance(router_logits_list, (tuple, list)):
        return []

    # Filter out None entries (from non-MoE layers)
    if e_score_correction_bias_list is not None:
        paired = [
            (rl, bias)
            for rl, bias in zip(router_logits_list, e_score_correction_bias_list, strict=False)
            if rl is not None
        ]
        router_logits_list = [rl for rl, _ in paired]
        bias_list = [bias for _, bias in paired]
    else:
        router_logits_list = [rl for rl in router_logits_list if rl is not None]
        bias_list = [None] * len(router_logits_list)
    if len(router_logits_list) == 0:
        return []

    layer_loss_list = []

    for logits, bias in zip(router_logits_list, bias_list, strict=False):
        # Cast to float32 for numerical stability
        logits = logits.to(dtype=torch.float32)
        if bias is not None:
            bias = bias.to(dtype=torch.float32, device=logits.device)
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
                    scores = F.softmax(seq_logits, dim=-1)
                    probs = scores
                P_i = torch.mean(probs, dim=0)  # [E]

                # f_i: per-expert assignment frequency within each sequence (normalized).
                # Top-k on scores + e_score_correction_bias, matching the router's
                # actual selection in Qwen2TokenMoeBlock (loss-free balancing).
                choice_scores = scores if bias is None else scores + bias
                _, topk_indices = torch.topk(choice_scores, k=top_k, dim=-1)  # [T_s, top_k]
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
                scores = F.softmax(logits, dim=-1)
                probs = scores
            P_i = torch.mean(probs, dim=0)  # [E]

            choice_scores = scores if bias is None else scores + bias
            _, topk_indices = torch.topk(choice_scores, k=top_k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, topk_indices, 1.0)
            f_i = (E / top_k) * torch.mean(mask, dim=0)  # [E]

            layer_loss_scalar = torch.sum(f_i.detach() * P_i)

        layer_loss_list.append(layer_loss_scalar)

    return layer_loss_list
