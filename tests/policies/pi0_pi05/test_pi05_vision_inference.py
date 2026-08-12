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

"""Tests for the pi0.5 vision-path inference optimizations.

Cameras are embedded as one batched vision-tower call at inference and one call each during
training, and the tower runs under bfloat16 autocast at inference only. Both are behavior
that must not leak into the training path, so that is what these assert.
"""

import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, PI05Pytorch
from tests.utils import require_cuda

TOKENS_PER_IMAGE = 4
EMBED_DIM = 8


class _RecordingVisionTower:
    """Stands in for ``paligemma_with_expert``, recording the batch size of every call."""

    def __init__(self):
        self.call_batch_sizes: list[int] = []

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.call_batch_sizes.append(image.shape[0])
        # Make the output a recognizable function of the input so a mis-split would show up.
        marker = image.flatten(1).mean(dim=1)
        return marker[:, None, None].expand(image.shape[0], TOKENS_PER_IMAGE, EMBED_DIM).clone()


class _Harness(torch.nn.Module):
    """Exercises ``PI05Pytorch._embed_images`` without building a 3B backbone."""

    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _RecordingVisionTower()
        self.gradient_checkpointing_enabled = False

    _apply_checkpoint = PI05Pytorch._apply_checkpoint
    _embed_images = PI05Pytorch._embed_images


def _distinct_images(count: int, batch: int = 1) -> list[torch.Tensor]:
    return [torch.full((batch, 3, 4, 4), float(i + 1)) for i in range(count)]


def test_inference_embeds_all_cameras_in_one_call():
    harness = _Harness().eval()
    images = _distinct_images(3)

    harness._embed_images(images)  # noqa: SLF001

    assert harness.paligemma_with_expert.call_batch_sizes == [3]


def test_training_keeps_one_call_per_camera():
    # The per-camera loop is what bounds peak activation memory when gradient checkpointing
    # recomputes the vision tower during the backward pass.
    harness = _Harness().train()
    images = _distinct_images(3)

    harness._embed_images(images)  # noqa: SLF001

    assert harness.paligemma_with_expert.call_batch_sizes == [1, 1, 1]


def test_batching_returns_the_same_embeddings_in_the_same_order():
    images = _distinct_images(3, batch=2)

    looped = _Harness().train()._embed_images(images)  # noqa: SLF001
    batched = _Harness().eval()._embed_images(images)  # noqa: SLF001

    assert len(batched) == len(looped) == 3
    for one, many in zip(looped, batched, strict=True):
        assert one.shape == many.shape == (2, TOKENS_PER_IMAGE, EMBED_DIM)
        torch.testing.assert_close(one, many)


def test_a_single_camera_is_not_needlessly_reshaped():
    harness = _Harness().eval()

    harness._embed_images(_distinct_images(1))  # noqa: SLF001

    assert harness.paligemma_with_expert.call_batch_sizes == [1]


def _autocast_stub(precision: str, *, training: bool) -> torch.nn.Module:
    stub = torch.nn.Module()
    stub.precision = precision
    stub.train(training)
    return stub


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_autocast_is_off_during_training(precision):
    # Training numerics must not change: the float32 pin on the vision path exists so optimizer
    # state never sees a parameter dtype change.
    stub = _autocast_stub(precision, training=True)
    image = torch.zeros(1, 3, 4, 4)

    assert PaliGemmaWithExpertModel._vision_autocast(stub, image) is False  # noqa: SLF001


def test_autocast_is_off_for_a_float32_checkpoint():
    # Asking for float32 should get float32, not a silently faster and less exact vision tower.
    stub = _autocast_stub("float32", training=False)
    image = torch.zeros(1, 3, 4, 4)

    assert PaliGemmaWithExpertModel._vision_autocast(stub, image) is False  # noqa: SLF001


def test_autocast_is_off_on_cpu():
    stub = _autocast_stub("bfloat16", training=False)
    image = torch.zeros(1, 3, 4, 4)

    assert PaliGemmaWithExpertModel._vision_autocast(stub, image) is False  # noqa: SLF001


@require_cuda
def test_autocast_is_enabled_for_bfloat16_inference_on_cuda():
    image = torch.zeros(1, 3, 4, 4, device="cuda")

    inference = _autocast_stub("bfloat16", training=False)
    assert PaliGemmaWithExpertModel._vision_autocast(inference, image) is True  # noqa: SLF001

    training = _autocast_stub("bfloat16", training=True)
    assert PaliGemmaWithExpertModel._vision_autocast(training, image) is False  # noqa: SLF001


@require_cuda
def test_batched_and_looped_embeddings_match_on_the_real_vision_tower():
    torch.manual_seed(0)
    model = PI05Pytorch(PI05Config(paligemma_variant="gemma_300m")).to("cuda").eval()
    images = [torch.rand(1, 3, 224, 224, device="cuda") for _ in range(3)]

    with torch.no_grad():
        batched = model._embed_images(images)  # noqa: SLF001
        model.train()
        looped = model._embed_images(images)  # noqa: SLF001

    for one, many in zip(looped, batched, strict=True):
        # Autocast is on for the eval pass, so this is a bfloat16-vs-float32 comparison.
        torch.testing.assert_close(one, many, rtol=2e-2, atol=2e-2)
