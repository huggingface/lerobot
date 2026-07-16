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
from PIL import Image

pytest.importorskip("peft")
pytest.importorskip("qwen_vl_utils")
pytest.importorskip("torchdiffeq")
pytest.importorskip("transformers")

from transformers.models.qwen2_vl.image_processing_qwen2_vl import (  # noqa: E402
    Qwen2VLImageProcessor,
)

from lerobot.policies.wall_x.constant import MAX_PIXELS, MIN_PIXELS  # noqa: E402
from lerobot.policies.wall_x.modeling_wall_x import (  # noqa: E402
    _prepare_wall_x_image_inputs,
    _wall_x_resize_dimensions,
)


@pytest.fixture
def image_processor() -> Qwen2VLImageProcessor:
    return Qwen2VLImageProcessor(
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
    )


def _legacy_pil_inputs(batch: dict[str, torch.Tensor], img_keys: list[str]) -> list[list[Image.Image]]:
    """Reproduce the previous per-sample PIL resize path for parity checks."""
    image_inputs = []
    for sample_index in range(next(iter(batch.values())).shape[0]):
        sample_images = []
        for key in img_keys:
            image = batch[key][sample_index].permute(1, 2, 0)
            image = Image.fromarray((image * 255).to(torch.uint8).numpy())
            intermediate_height, intermediate_width, resized_height, resized_width = (
                _wall_x_resize_dimensions(image.height, image.width)
            )
            image = image.resize((intermediate_width, intermediate_height))
            image = image.resize((resized_width, resized_height))
            sample_images.append(image)
        image_inputs.append(sample_images)
    return image_inputs


def test_batched_preprocessing_preserves_device_shape_and_sample_camera_order(image_processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    batch = {
        "cam_a": torch.rand(2, 3, 180, 320, device=device),
        "cam_b": torch.rand(2, 3, 240, 200, device=device),
    }
    img_keys = ["cam_a", "cam_b"]

    image_inputs, dimensions = _prepare_wall_x_image_inputs(batch, img_keys)

    assert dimensions == {
        "cam_a": (180, 320, 140, 252),
        "cam_b": (240, 200, 252, 224),
    }
    assert [[image.device for image in sample] for sample in image_inputs] == [
        [device, device],
        [device, device],
    ]
    assert [[image.shape for image in sample] for sample in image_inputs] == [
        [torch.Size((3, 140, 252)), torch.Size((3, 252, 224))],
        [torch.Size((3, 140, 252)), torch.Size((3, 252, 224))],
    ]

    processed = image_processor(images=image_inputs, return_tensors="pt", device=device)
    separately_processed = [
        image_processor(images=[image], return_tensors="pt", device=device)
        for sample in image_inputs
        for image in sample
    ]

    torch.testing.assert_close(
        processed.pixel_values,
        torch.cat([item.pixel_values for item in separately_processed]),
    )
    torch.testing.assert_close(
        processed.image_grid_thw,
        torch.cat([item.image_grid_thw for item in separately_processed]),
    )


def test_tensor_preprocessing_matches_pil_grid_and_is_numerically_close(image_processor):
    torch.manual_seed(0)
    batch = {
        "cam_a": torch.rand(2, 3, 180, 320),
        "cam_b": torch.rand(2, 3, 240, 200),
    }
    img_keys = ["cam_a", "cam_b"]

    tensor_inputs, _ = _prepare_wall_x_image_inputs(batch, img_keys)
    pil_inputs = _legacy_pil_inputs(batch, img_keys)
    tensor_output = image_processor(images=tensor_inputs, return_tensors="pt", device="cpu")
    pil_output = image_processor(images=pil_inputs, return_tensors="pt", device="cpu")

    torch.testing.assert_close(tensor_output.image_grid_thw, pil_output.image_grid_thw)
    # PIL and torchvision use slightly different antialiased bicubic kernels.
    torch.testing.assert_close(
        tensor_output.pixel_values,
        pil_output.pixel_values,
        rtol=0,
        atol=0.04,
    )
