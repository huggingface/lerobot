# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Custom eager attention with prediction-slot isolation."""

import torch
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def pred_slot_isolated_eager(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    is_pred_key: torch.Tensor | None = None,
    pred_slot_id: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    batch_size, query_length, key_length = query.shape[0], query.shape[2], key_states.shape[2]
    neg_inf = torch.finfo(query.dtype).min
    if attention_mask is not None:
        add_mask = attention_mask[:, :, :, :key_length].to(query.dtype).clone()
    else:
        add_mask = torch.zeros(
            batch_size, 1, query_length, key_length, dtype=query.dtype, device=query.device
        )
    if is_pred_key is not None and is_pred_key.any():
        if pred_slot_id is not None:
            slot_q, slot_k = pred_slot_id[:, :, None], pred_slot_id[:, None, :]
            same_slot = (slot_q == slot_k) & (slot_q >= 0)
            add_mask = add_mask.masked_fill(same_slot[:, None], 0.0)
            pred_mask = is_pred_key[:, None, :] & ~same_slot
        else:
            pred_mask = is_pred_key[:, None, :].expand(batch_size, query_length, key_length)
        add_mask = add_mask.masked_fill(pred_mask[:, None], neg_inf)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling + add_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return output, attn_weights


ALL_ATTENTION_FUNCTIONS["pred_slot_isolated_eager"] = pred_slot_isolated_eager
ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["pred_slot_isolated_eager"] = eager_mask
