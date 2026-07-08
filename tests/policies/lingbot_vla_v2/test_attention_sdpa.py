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

"""The SDPA attention path must match the eager path (same softmax attention up to
floating-point reassociation), so selecting attention_implementation='sdpa' does not change
a pretrained model's outputs. Runs on CPU via the SDPA math backend."""

import pytest

torch = pytest.importorskip("torch")


def _valid_mask(bsize, seq):
    # bool mask, True = attend; guarantee each query attends to >=1 key (its own position)
    # so both the eager (big-neg) and SDPA (-inf) softmax rows are well-defined.
    torch.manual_seed(3)
    mask = torch.rand(bsize, seq, seq) > 0.3
    return mask | torch.eye(seq, dtype=torch.bool).unsqueeze(0)


@pytest.mark.parametrize("num_kv_heads", [2, 4])  # GQA (2) and MHA (4)
def test_sdpa_matches_eager(num_kv_heads):
    from lerobot.policies.lingbot_vla_v2.utils import (
        our_eager_attention_forward,
        our_sdpa_attention_forward,
    )

    torch.manual_seed(0)
    bsize, seq, num_att_heads, head_dim = 2, 8, 4, 8
    q = torch.randn(bsize, seq, num_att_heads, head_dim)
    k = torch.randn(bsize, seq, num_kv_heads, head_dim)
    v = torch.randn(bsize, seq, num_kv_heads, head_dim)
    mask = _valid_mask(bsize, seq)

    eager = our_eager_attention_forward(q, k, v, mask)
    sdpa = our_sdpa_attention_forward(q, k, v, mask)

    assert sdpa.shape == eager.shape == (bsize, seq, num_att_heads * head_dim)
    torch.testing.assert_close(sdpa, eager, atol=1e-4, rtol=1e-4)
