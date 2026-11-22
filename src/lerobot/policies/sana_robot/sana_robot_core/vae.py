# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/utils/vae.py. Import paths are rewritten to the LeRobot package;
# the block-wise encoder of a stream; the compiled single-pass encoder of training is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.policies.sana_robot.sana_robot_core.causal_vae import LTX2VideoEncoder3d


def grouping_slots(encoder: LTX2VideoEncoder3d) -> list[int]:
    """Return the slots of the encoder's cache that hold the frame grouping state of a temporal downsampler instead of cached frames."""
    (slots, slot) = ([], 1)
    for block in encoder.down_blocks:
        slot += 2 * len(block.resnets)
        for downsampler in block.downsamplers:
            if downsampler.stride[0] > 1:
                slots.append(slot)
                slot += 1
            slot += 1
    return slots


def load_video_encoder(model_path, device, dtype=torch.bfloat16):
    """Load the encoder of the LTX-2 causal video VAE.

    A stream is encoded block by block on the cache of the encoder's layers: the first frame alone, then 8 frames
    at a time, every block after the cache the one before left. The encoder is causal in time, so the blocks give
    the latents of the whole video up to the rounding of the precision.

    Args:
        model_path: The directory of the VAE weights in the diffusers layout, with the ``vae`` subfolder.
        device: The device to encode on.
        dtype: The precision of the weights and of the latent normalization.

    Returns:
        ``encode(frames, stream)``: frames [B, 3, T, H, W] in [-1, 1] to normalized float32 latents
        [B, 128, T', H / 32, W / 32]. ``stream`` is the state of one video, the frames the encoder's layers cache,
        as a list of tensors that starts empty and is updated in place: its first call takes the first frame and
        each later call the next 8 * n frames.
    """
    folder = Path(model_path, "vae")
    config = json.loads((folder / "config.json").read_text())
    weights = load_file(folder / "diffusion_pytorch_model.safetensors")
    encoder = LTX2VideoEncoder3d(
        in_channels=config["in_channels"],
        out_channels=config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        down_block_types=tuple(config["down_block_types"]),
        spatio_temporal_scaling=tuple(config["spatio_temporal_scaling"]),
        layers_per_block=tuple(config["layers_per_block"]),
        downsample_type=tuple(config["downsample_type"]),
        patch_size=config["patch_size"],
        patch_size_t=config["patch_size_t"],
        resnet_norm_eps=config["resnet_norm_eps"],
        is_causal=config["encoder_causal"],
        spatial_padding_mode=config["encoder_spatial_padding_mode"],
    )
    encoder.load_state_dict(
        {name[len("encoder.") :]: value for (name, value) in weights.items() if name.startswith("encoder.")},
        strict=True,
    )
    encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    mean = weights["latents_mean"].view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    std = weights["latents_std"].view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    scaling_factor = config["scaling_factor"]
    block = config["temporal_compression_ratio"]
    slots = grouping_slots(encoder)

    @torch.no_grad()
    def encode(frames: torch.Tensor, stream: list) -> torch.Tensor:
        frames = frames.to(device=device, dtype=dtype)
        cache = list(stream)
        if cache:
            for slot in slots:
                cache.insert(slot, {"prefixed": True, "pending": None})
        (start, step, hidden) = (0, frames.shape[2] % block or block, [])
        while start < frames.shape[2]:
            hidden.append(
                encoder(frames[:, :, start : start + step, :, :], causal=True, feat_cache=cache, feat_idx=[0])
            )
            (start, step) = (start + step, block)
        hidden = hidden[0] if len(hidden) == 1 else torch.cat(hidden, dim=2)
        latents = torch.chunk(hidden, 2, dim=1)[0]
        stream[:] = [entry for entry in cache if torch.is_tensor(entry)]
        return ((latents - mean) * scaling_factor / std).float()

    return encode
