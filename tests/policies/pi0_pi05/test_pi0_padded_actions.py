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

"""pi0 loss reduction with ``action_is_pad``, run through the real ``PI0Policy.forward``."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

BATCH, CHUNK, ACTION_DIM, MAX_ACTION_DIM = 2, 4, 2, 3


class _LossHarness:
    """The attributes ``PI0Policy.forward`` reads, with the model replaced by fixed per-element losses."""

    forward = PI0Policy.forward

    def __init__(self, losses: Tensor):
        self.config = SimpleNamespace(
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
        )
        self.model = SimpleNamespace(
            sample_noise=lambda shape, device: torch.zeros(shape, device=device),
            sample_time=lambda batch_size, device: torch.zeros(batch_size, device=device),
            forward=lambda *_args: losses,
        )

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        return [], []

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.zeros(BATCH, 1)

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        return batch[ACTION]


def _losses() -> Tensor:
    """Per-element losses of ``t + 1`` at step ``t``; the padded action dimension is 100 and gets truncated."""
    losses = torch.arange(1, CHUNK + 1, dtype=torch.float32).view(1, CHUNK, 1)
    losses = losses.expand(BATCH, CHUNK, MAX_ACTION_DIM).clone()
    losses[..., ACTION_DIM:] = 100.0
    return losses.requires_grad_()


def _batch(action_is_pad: Tensor | None) -> dict[str, Tensor]:
    batch = {
        ACTION: torch.zeros(BATCH, CHUNK, MAX_ACTION_DIM),
        OBS_LANGUAGE_TOKENS: torch.zeros(BATCH, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(BATCH, 1, dtype=torch.bool),
    }
    if action_is_pad is not None:
        batch["action_is_pad"] = action_is_pad
    return batch


def _tail_padded() -> Tensor:
    """Sample 0 has its last two steps padded; sample 1 has none."""
    action_is_pad = torch.zeros(BATCH, CHUNK, dtype=torch.bool)
    action_is_pad[0, CHUNK // 2 :] = True
    return action_is_pad


@pytest.mark.parametrize("action_is_pad", [None, torch.zeros(BATCH, CHUNK, dtype=torch.bool)])
def test_unpadded_batch_keeps_the_plain_mean(action_is_pad: Tensor | None):
    # Mean of [1, 2, 3, 4], with or without an all-False padding mask.
    policy = _LossHarness(_losses())

    loss, _ = policy.forward(_batch(action_is_pad))
    per_sample_loss, _ = policy.forward(_batch(action_is_pad), reduction="none")

    torch.testing.assert_close(loss, torch.tensor(2.5))
    torch.testing.assert_close(per_sample_loss, torch.tensor([2.5, 2.5]))


def test_padded_steps_are_excluded_from_the_loss():
    policy = _LossHarness(_losses())

    loss, _ = policy.forward(_batch(_tail_padded()))
    per_sample_loss, _ = policy.forward(_batch(_tail_padded()), reduction="none")

    # Valid elements: sample 0 steps [1, 2] and sample 1 steps [1, 2, 3, 4], each over ACTION_DIM dims.
    torch.testing.assert_close(loss, torch.tensor(26.0 / 12.0))
    torch.testing.assert_close(per_sample_loss, torch.tensor([1.5, 2.5]))


def test_padded_steps_get_no_gradient():
    losses = _losses()
    policy = _LossHarness(losses)

    loss, _ = policy.forward(_batch(_tail_padded()))
    loss.backward()

    grad = losses.grad[..., :ACTION_DIM]
    num_valid = 12
    torch.testing.assert_close(grad[0, CHUNK // 2 :], torch.zeros(CHUNK // 2, ACTION_DIM))
    torch.testing.assert_close(grad[0, : CHUNK // 2], torch.full((CHUNK // 2, ACTION_DIM), 1 / num_valid))
    torch.testing.assert_close(grad[1], torch.full((CHUNK, ACTION_DIM), 1 / num_valid))


def test_fully_padded_batch_gives_zero_loss():
    # No valid step left: the loss is 0, not NaN from dividing by zero.
    policy = _LossHarness(_losses())
    action_is_pad = torch.ones(BATCH, CHUNK, dtype=torch.bool)

    loss, _ = policy.forward(_batch(action_is_pad))
    per_sample_loss, _ = policy.forward(_batch(action_is_pad), reduction="none")

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(per_sample_loss, torch.zeros(BATCH))


def test_float_padding_mask_is_treated_as_bool():
    # A 0/1 float mask gives the same loss as the bool one.
    policy = _LossHarness(_losses())

    loss, _ = policy.forward(_batch(_tail_padded().float()))

    torch.testing.assert_close(loss, torch.tensor(26.0 / 12.0))


# (CHUNK,) would otherwise broadcast over the batch without any error.
@pytest.mark.parametrize("shape", [(CHUNK,), (BATCH,), (BATCH, CHUNK + 1), (BATCH, CHUNK, 1)])
def test_padding_mask_with_wrong_shape_raises(shape: tuple[int, ...]):
    policy = _LossHarness(_losses())

    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        policy.forward(_batch(torch.zeros(shape, dtype=torch.bool)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_padding_mask_with_cuda_losses():
    policy = _LossHarness(_losses().cuda())

    loss, _ = policy.forward(_batch(_tail_padded()))

    torch.testing.assert_close(loss.cpu(), torch.tensor(26.0 / 12.0))
