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

from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("transformers")

from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

CAMERA_KEYS = tuple(f"observation.images.{name}" for name in ("front", "wrist", "overhead"))
PreprocessImages = Callable[[dict[str, Tensor]], tuple[list[Tensor], list[Tensor]]]


@pytest.fixture(params=["pi0", "pi05", "smolvla"])
def preprocess_images(request: pytest.FixtureRequest) -> PreprocessImages:
    # Call the real preprocessing methods without constructing or downloading a backbone.
    policy = nn.Module()
    policy.register_parameter("device_anchor", nn.Parameter(torch.zeros(1)))
    policy.config = SimpleNamespace(
        image_features=dict.fromkeys(CAMERA_KEYS),
        image_resolution=(8, 8),
        resize_imgs_with_padding=(8, 8),
        empty_cameras=0,
    )
    method = {
        "pi0": PI0Policy._preprocess_images,
        "pi05": PI05Policy._preprocess_images,
        "smolvla": SmolVLAPolicy.prepare_images,
    }[request.param]
    return partial(method, policy)


@pytest.mark.parametrize("present_slots", [(0, 1, 2), (0, 1), (0, 2), (1, 2), (0,), (1,), (2,)])
def test_missing_cameras_preserve_configured_slots(
    preprocess_images: PreprocessImages, present_slots: tuple[int, ...]
) -> None:
    full_batch = {key: torch.full((2, 3, 4, 6), (i + 1) / 4) for i, key in enumerate(CAMERA_KEYS)}
    expected_images, _ = preprocess_images(full_batch)
    # Batch insertion order must not affect the configured camera order either.
    batch = {CAMERA_KEYS[i]: full_batch[CAMERA_KEYS[i]] for i in reversed(present_slots)}

    images, masks = preprocess_images(batch)

    assert len(images) == len(masks) == len(CAMERA_KEYS)
    for i, (image, mask) in enumerate(zip(images, masks, strict=True)):
        assert image.shape == (2, 3, 8, 8)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.bool
        assert mask.device == image.device
        if i in present_slots:
            torch.testing.assert_close(image, expected_images[i])
            assert mask.tolist() == [True, True]
        else:
            torch.testing.assert_close(image, torch.full_like(image, -1))
            assert mask.tolist() == [False, False]


def test_all_configured_cameras_missing_raises(preprocess_images: PreprocessImages) -> None:
    with pytest.raises(ValueError, match="All image features are missing"):
        preprocess_images({"observation.images.unknown": torch.zeros(2, 3, 4, 6)})


def test_smolvla_preserves_padding_mask_and_uses_latest_frame() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features=dict.fromkeys(CAMERA_KEYS), resize_imgs_with_padding=None, empty_cameras=1
        )
    )
    frames = torch.zeros(2, 3, 3, 4, 6)
    frames[:, -1] = 0.75
    batch = {CAMERA_KEYS[1]: frames, f"{CAMERA_KEYS[1]}_padding_mask": torch.tensor([False, True])}

    images, masks = SmolVLAPolicy.prepare_images(policy, batch)

    assert len(images) == len(masks) == 3
    torch.testing.assert_close(images[1], torch.full((2, 3, 4, 6), 0.5))
    assert [mask.tolist() for mask in masks] == [[False, False], [False, True], [False, False]]


@pytest.mark.parametrize("channels_first", [False, True])
def test_pi05_preserves_video_history_and_frame_masks(channels_first: bool) -> None:
    policy = nn.Module()
    policy.register_parameter("device_anchor", nn.Parameter(torch.zeros(1)))
    policy.config = SimpleNamespace(image_features=dict.fromkeys(CAMERA_KEYS), image_resolution=(8, 8))
    frames = torch.full((2, 4, 3, 4, 6), 0.75)
    if not channels_first:
        frames = frames.permute(0, 1, 3, 4, 2)
    is_pad = torch.tensor([[True, False, False, False], [True, True, False, False]])

    images, masks = PI05Policy._preprocess_images(
        policy, {CAMERA_KEYS[1]: frames, f"{CAMERA_KEYS[1]}_is_pad": is_pad}
    )

    assert len(images) == len(masks) == 3
    expected_shape = (2, 4, 3, 8, 8) if channels_first else (2, 4, 8, 8, 3)
    for i, (image, mask) in enumerate(zip(images, masks, strict=True)):
        assert image.shape == expected_shape
        if i == 1:
            torch.testing.assert_close(mask, ~is_pad)
            assert image.max().item() == pytest.approx(0.5)
        else:
            torch.testing.assert_close(image, torch.full_like(image, -1))
            torch.testing.assert_close(mask, torch.zeros_like(is_pad))
