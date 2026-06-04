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

"""Regression tests for PI052 FAST action-code supervision."""

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("transformers")
pytest.importorskip("liger_kernel")

from lerobot.policies.pi052.modeling_pi052 import _fast_lin_ce  # noqa: E402


def _fast_ce(logits, action_tokens, action_code_mask, predict_actions_t):
    """Adapter: ``_fast_lin_ce`` is Liger-fused (hidden @ lm_head_weightᵀ).

    Feeding an identity ``lm_head_weight`` makes the computed logits equal the
    provided ``logits``, so these regression tests exercise the masking/gating
    logic exactly as before the fused-CE refactor. Liger's Triton kernel is
    GPU-only, so inputs are moved to CUDA and the loss is returned on CPU
    (keeping grad flowing back to the CPU ``logits`` leaf).
    """
    if not torch.cuda.is_available():
        pytest.skip("Liger fused CE requires CUDA")
    vocab_size = logits.shape[-1]
    eye = torch.eye(vocab_size, dtype=logits.dtype, device="cuda")
    predict = predict_actions_t.cuda() if predict_actions_t is not None else None
    loss = _fast_lin_ce(
        logits.cuda(), eye, action_tokens.cuda(), action_code_mask.cuda(), predict
    )
    return loss.cpu()


def test_fast_ce_supervises_only_discrete_action_codes():
    """Wrapper tokens can be wrong without affecting the FAST action-code loss."""
    vocab_size = 8
    action_tokens = torch.tensor([[1, 2, 3, 4, 5, 0]])
    action_code_mask = torch.tensor([[False, False, True, True, False, False]])

    logits = torch.zeros(1, action_tokens.shape[1], vocab_size)
    # Deliberately bad wrapper-token predictions. These should be ignored.
    logits[0, 0, 7] = 10.0  # target would be token 2
    logits[0, 3, 7] = 10.0  # target would be delimiter token 5
    # Correct action-code predictions: hidden t predicts target t + 1.
    logits[0, 1, 3] = 10.0
    logits[0, 2, 4] = 10.0

    loss = _fast_ce(logits, action_tokens, action_code_mask, predict_actions_t=None)
    expected = F.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2]]),
        torch.tensor([3, 4]),
        reduction="mean",
    )

    # Looser tolerance: the fused Triton kernel (GPU) differs from CPU eager
    # F.cross_entropy at the ~1e-7 level, which exceeds the default rtol on
    # these very small (~1e-4) losses.
    assert torch.allclose(loss, expected, atol=1e-5, rtol=1e-3)


def test_fast_ce_masks_non_action_samples():
    """Recipe samples with predict_actions=False do not contribute FAST loss."""
    vocab_size = 8
    action_tokens = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
    action_code_mask = torch.tensor(
        [[False, False, True, True], [False, False, True, True]]
    )
    predict_actions = torch.tensor([True, False])

    logits = torch.zeros(2, action_tokens.shape[1], vocab_size)
    logits[0, 1, 3] = 10.0
    logits[0, 2, 4] = 10.0
    # Bad predictions in the masked sample should not matter.
    logits[1, 1, 7] = 10.0
    logits[1, 2, 7] = 10.0

    loss = _fast_ce(logits, action_tokens, action_code_mask, predict_actions)
    expected = F.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2]]),
        torch.tensor([3, 4]),
        reduction="mean",
    )

    # Looser tolerance: the fused Triton kernel (GPU) differs from CPU eager
    # F.cross_entropy at the ~1e-7 level, which exceeds the default rtol on
    # these very small (~1e-4) losses.
    assert torch.allclose(loss, expected, atol=1e-5, rtol=1e-3)


def test_fast_ce_returns_zero_when_no_action_code_positions_are_valid():
    logits = torch.randn(2, 4, 8, requires_grad=True)
    action_tokens = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
    action_code_mask = torch.zeros_like(action_tokens, dtype=torch.bool)

    loss = _fast_ce(logits, action_tokens, action_code_mask, predict_actions_t=None)

    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None
