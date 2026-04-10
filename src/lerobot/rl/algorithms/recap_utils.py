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

"""Shared utilities for RECAP value network training and inference."""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES

PI05_VLM_KEY_PREFIX = "paligemma_with_expert.paligemma."


def collect_images(batch: dict[str, Any], image_size: int) -> Tensor:
    """Extract camera images from batch, handling both per-camera and combined formats."""
    image_keys = sorted(k for k in batch if k.startswith(f"{OBS_IMAGES}."))
    if image_keys:
        img_list = []
        for key in image_keys:
            img = batch[key]
            if img.ndim == 5:
                img = img[:, -1]
            img_list.append(img)
        images = torch.stack(img_list, dim=1)
    elif "observation.images" in batch:
        images = batch["observation.images"]
        if images.ndim == 4:
            images = images.unsqueeze(1)
    else:
        raise ValueError("No image keys found in batch")

    batch_size, n_cams, C, H, W = images.shape
    target_h, target_w = image_size, image_size
    if target_h != H or target_w != W:
        flat = images.reshape(batch_size * n_cams, C, H, W)
        flat = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        images = flat.reshape(batch_size, n_cams, C, target_h, target_w)
    return images


def load_pretrained_vlm_weights(model: nn.Module, pretrained_path: str) -> None:
    """Load VLM weights from a pretrained pi0.5 checkpoint into the PaliGemma backbone."""
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    logging.info(f"Loading pretrained VLM weights from {pretrained_path}")

    resolved_file = cached_file(pretrained_path, "model.safetensors")
    if resolved_file is None:
        raise FileNotFoundError(f"Could not find model.safetensors in {pretrained_path}")
    full_state_dict = load_file(resolved_file)

    vlm_state_dict: dict[str, Tensor] = {}
    for key, value in full_state_dict.items():
        if not key.startswith(PI05_VLM_KEY_PREFIX):
            continue
        new_key = key[len("paligemma_with_expert."):]
        vlm_state_dict[new_key] = value

    lm_head_key = "paligemma.lm_head.weight"
    embed_key = "paligemma.model.language_model.embed_tokens.weight"
    if lm_head_key in vlm_state_dict and embed_key not in vlm_state_dict:
        vlm_state_dict[embed_key] = vlm_state_dict[lm_head_key].clone()

    missing, unexpected = model.load_state_dict(vlm_state_dict, strict=False)

    expected_missing = [
        k for k in missing
        if k.startswith(("value_head.", "value_bin_support"))
    ]
    truly_missing = [k for k in missing if k not in expected_missing]

    loaded_count = len(vlm_state_dict) - len(unexpected)
    logging.info(
        f"Pretrained VLM weights: loaded {loaded_count} tensors, "
        f"{len(expected_missing)} expected-missing (value head), "
        f"{len(truly_missing)} unexpectedly missing, "
        f"{len(unexpected)} unexpected."
    )
    if truly_missing:
        logging.warning(f"Unexpectedly missing keys: {truly_missing[:10]}")
    if unexpected:
        logging.warning(f"Unexpected keys (not loaded): {unexpected[:10]}")
