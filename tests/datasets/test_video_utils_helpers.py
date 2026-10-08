# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.video_utils import (  # noqa: E402
    apply_rgb_transforms,
    convert_image_depth_units,
    normalize_rgb_frames,
)


def test_normalize_rgb_frames():
    frames = torch.tensor([[0, 255]], dtype=torch.uint8)
    assert normalize_rgb_frames(frames, return_uint8=True) is frames
    normalized = normalize_rgb_frames(frames, return_uint8=False)
    assert normalized.dtype == torch.float32
    assert normalized.tolist() == [[0.0, 1.0]]


def test_apply_rgb_transforms_skips_depth_and_missing_transforms():
    item = {"rgb": torch.ones(1), "depth": torch.ones(1)}
    apply_rgb_transforms(item, None, ["rgb", "depth"], ["depth"])
    assert item["rgb"].item() == 1
    apply_rgb_transforms(item, lambda image: image * 2, ["rgb", "depth"], ["depth"])
    assert item["rgb"].item() == 2
    assert item["depth"].item() == 1


def test_convert_image_depth_units():
    item = {
        "metres": torch.tensor([1.5]),
        "millimetres": torch.tensor([1500.0]),
        "unknown": torch.tensor([7.0]),
    }
    convert_image_depth_units(item, {"metres": "m", "millimetres": "mm", "unknown": None}, output_unit="mm")
    assert item["metres"].item() == 1500.0
    assert item["millimetres"].item() == 1500.0
    assert item["unknown"].item() == 7.0
