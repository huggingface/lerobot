#!/usr/bin/env python

# Regression tests for image scale handling in replay-buffer transitions.
#
# Datasets created with `return_uint8=True` yield 0-255 uint8 images, while
# online observations arrive as 0-1 float32 after the observation processor.
# Mixing both in one batch feeds the policy inconsistent inputs, and writing
# 0-255 floats back through the image writer (float expected in [0, 1] or
# uint8) fails. `_to_transition_tensor` normalizes image keys to 0-1 float32.

import torch

from lerobot.rl.buffer import _to_transition_tensor
from lerobot.utils.constants import OBS_IMAGE


def test_uint8_image_scaled_to_unit_range():
    image = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)

    out = _to_transition_tensor(image, f"{OBS_IMAGE}.top")

    assert out.dtype == torch.float32
    assert out.shape == (1, 3, 8, 8)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert torch.allclose(out.squeeze(0) * 255.0, image.to(torch.float32), atol=1e-4)


def test_float_image_in_unit_range_untouched():
    image = torch.rand(3, 8, 8)

    out = _to_transition_tensor(image, f"{OBS_IMAGE}.wrist")

    assert out.dtype == torch.float32
    assert torch.equal(out, image.unsqueeze(0))


def test_float_image_in_byte_range_is_scaled():
    image = torch.randint(0, 256, (3, 8, 8)).to(torch.float32)

    out = _to_transition_tensor(image, f"{OBS_IMAGE}.top")

    assert out.max() <= 1.0
    assert torch.allclose(out.squeeze(0) * 255.0, image, atol=1e-3)


def test_non_image_keys_keep_their_scale():
    state = torch.arange(12, dtype=torch.uint8).reshape(12)

    out = _to_transition_tensor(state, "observation.state")

    assert out.dtype == torch.uint8
    assert out.shape == (1, 12)
