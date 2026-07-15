#!/usr/bin/env python

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

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi052.modeling_pi052 import _lin_ce_flat


@pytest.mark.parametrize("z_loss_weight", [0.0, 1e-4])
@pytest.mark.parametrize("rows,valid_rows", [(24, 9), (48, 25)])
def test_bucketed_ce_matches_dense_loss_and_gradients(z_loss_weight, rows, valid_rows):
    generator = torch.Generator().manual_seed(23)
    hidden_size, vocab_size = 7, 19
    hidden_ref = torch.randn(rows, hidden_size, generator=generator, dtype=torch.float64, requires_grad=True)
    weight_ref = torch.randn(
        vocab_size, hidden_size, generator=generator, dtype=torch.float64, requires_grad=True
    )
    labels = torch.full((rows,), -100, dtype=torch.long)
    valid_indices = torch.randperm(rows, generator=generator)[:valid_rows]
    labels[valid_indices] = torch.randint(0, vocab_size, (valid_rows,), generator=generator)
    hidden_bucketed = hidden_ref.detach().clone().requires_grad_(True)
    weight_bucketed = weight_ref.detach().clone().requires_grad_(True)

    import lerobot.policies.pi052.modeling_pi052 as modeling_pi052

    loss_ref = _lin_ce_flat(hidden_ref, weight_ref, labels, z_loss_weight=z_loss_weight)
    old_limit = modeling_pi052._LOGITS_CE_MAX_POSITIONS
    modeling_pi052._LOGITS_CE_MAX_POSITIONS = 16
    try:
        loss_bucketed = _lin_ce_flat(
            hidden_bucketed,
            weight_bucketed,
            labels,
            z_loss_weight=z_loss_weight,
        )
    finally:
        modeling_pi052._LOGITS_CE_MAX_POSITIONS = old_limit

    loss_ref.backward()
    loss_bucketed.backward()

    torch.testing.assert_close(loss_bucketed, loss_ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hidden_bucketed.grad, hidden_ref.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weight_bucketed.grad, weight_ref.grad, rtol=1e-12, atol=1e-12)


def test_bucketed_ce_all_ignored_preserves_zero_gradients():
    hidden = torch.randn(24, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(19, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.full((24,), -100, dtype=torch.long)

    import lerobot.policies.pi052.modeling_pi052 as modeling_pi052

    old_limit = modeling_pi052._LOGITS_CE_MAX_POSITIONS
    modeling_pi052._LOGITS_CE_MAX_POSITIONS = 16
    try:
        loss = _lin_ce_flat(hidden, weight, labels)
    finally:
        modeling_pi052._LOGITS_CE_MAX_POSITIONS = old_limit
    loss.backward()

    assert loss.item() == 0.0
    assert hidden.grad is not None
    assert weight.grad is not None
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0
