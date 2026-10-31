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
"""A tiny random Qwen3-VL config and image prompts shared by the Qwen3-VL fixed-grid tests."""

import torch
from transformers import Qwen3VLConfig

IMAGE_TOKEN_ID = 7
# The only value-dependent shapes allowed in an exported program: the stock text model, which the fixed-grid path
# keeps, reads and writes the DeepStack features at the image tokens through a boolean mask, once per DeepStack layer.
STOCK_DEEPSTACK_NODES = [
    "aten.index.Tensor with a boolean mask",
    "aten.index_put_.default with a boolean mask",
]
# Still images, as the image processors produce them (t == 1), of two shapes.
IMAGE_GRIDS = [[1, 4, 4], [1, 4, 6]]


def tiny_qwen3_vl_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=IMAGE_TOKEN_ID + 1,
    )


def image_prompts(config: Qwen3VLConfig, suffix: tuple[int, ...] = (4, 5)) -> dict[str, torch.Tensor]:
    """Two left-padded prompts of different lengths: text, image, text, image, text, then `suffix`."""
    merge = config.vision_config.spatial_merge_size
    rows = []
    for prefix in ([1, 2], [1, 2, 3, 4, 5]):
        ids = list(prefix)
        for t, h, w in IMAGE_GRIDS:
            ids += [IMAGE_TOKEN_ID] * (t * h * w // merge**2) + [3]
        rows.append(ids + list(suffix))
    length = max(len(r) for r in rows)
    input_ids = torch.tensor([[0] * (length - len(r)) + r for r in rows])
    grid_thw = torch.tensor(IMAGE_GRIDS * len(rows))
    vision = config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": input_ids,
        "attention_mask": torch.tensor([[0] * (length - len(r)) + [1] * len(r) for r in rows]),
        "mm_token_type_ids": (input_ids == IMAGE_TOKEN_ID).int(),
        "pixel_values": torch.randn(int(grid_thw.prod(-1).sum()), patch_dim),
        "image_grid_thw": grid_thw,
    }
