# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Generate synthetic sample inputs for ONNX tracing and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


def make_sample_inputs(
    policy: PreTrainedPolicy,
    cfg,
    mode: str = "act",
) -> tuple[Tensor, ...]:
    """Return a tuple of zero-initialized sample tensors for ONNX tracing.

    Args:
        policy: The policy whose config is used to determine shapes.
        cfg: ``ExportConfig`` instance (used for ``batch_size`` and ``device``).
        mode: One of ``"act"``, ``"diffusion-unet"``, ``"diffusion-ddim"``, ``"generic"``.

    Returns:
        A tuple of ``torch.Tensor`` objects suitable for passing to ``torch.onnx.export``.
    """
    config = policy.config
    device = torch.device(getattr(cfg, "device", "cpu"))
    batch_size: int = getattr(cfg, "batch_size", 1)

    if mode == "act":
        return _make_act_inputs(config, batch_size, device)
    elif mode == "diffusion-unet":
        return _make_diffusion_unet_inputs(policy, batch_size, device)
    elif mode == "diffusion-ddim":
        return _make_diffusion_ddim_inputs(policy, batch_size, device)
    elif mode == "generic":
        return _make_generic_inputs(config, batch_size, device)
    else:
        raise ValueError(f"Unknown sample_inputs mode '{mode}'")


# ──────────────────────────────────────────────────────────────────────────────
# ACT
# ──────────────────────────────────────────────────────────────────────────────


def _make_act_inputs(config, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Sample inputs matching ``ACTInferenceWrapper.forward`` signature."""
    inputs: list[Tensor] = []

    # robot_state or env_state
    if config.robot_state_feature is not None:
        state_dim = config.robot_state_feature.shape[0]
        inputs.append(torch.zeros(batch_size, state_dim, device=device))
    elif config.env_state_feature is not None:
        env_dim = config.env_state_feature.shape[0]
        inputs.append(torch.zeros(batch_size, env_dim, device=device))
    else:
        raise ValueError("ACT policy must have at least robot_state_feature or env_state_feature.")

    # Per-camera images
    for _key, ft in (config.image_features or {}).items():
        # ft.shape is (C, H, W)
        inputs.append(torch.zeros(batch_size, *ft.shape, device=device))

    return tuple(inputs)


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion
# ──────────────────────────────────────────────────────────────────────────────


def _get_diffusion_global_cond_dim(policy) -> int:
    """Compute the global_cond_dim expected by the Diffusion UNet."""
    config = policy.config
    n_obs_steps = config.n_obs_steps

    # Build a dummy batch and pass through _prepare_global_conditioning
    device = next(policy.parameters()).device
    state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0

    dummy_batch = {}
    if config.robot_state_feature:
        # (B, n_obs_steps, state_dim)
        dummy_batch[OBS_STATE] = torch.zeros(1, n_obs_steps, state_dim, device=device)

    if config.image_features:
        # (B, n_obs_steps, n_cameras, C, H, W)
        first_feat = next(iter(config.image_features.values()))
        n_cameras = len(config.image_features)
        c, h, w = first_feat.shape
        dummy_batch[OBS_IMAGES] = torch.zeros(1, n_obs_steps, n_cameras, c, h, w, device=device)

    if config.env_state_feature and not config.robot_state_feature:
        env_dim = config.env_state_feature.shape[0]
        dummy_batch[OBS_STATE] = torch.zeros(1, n_obs_steps, env_dim, device=device)

    with torch.no_grad():
        global_cond = policy.diffusion._prepare_global_conditioning(dummy_batch)
    return global_cond.shape[-1]


def _make_diffusion_unet_inputs(policy, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Sample inputs for ``DiffusionUNetWrapper.forward``."""
    config = policy.config
    action_dim = config.action_feature.shape[0]
    horizon = config.horizon
    global_cond_dim = _get_diffusion_global_cond_dim(policy)

    sample = torch.zeros(batch_size, horizon, action_dim, device=device)
    timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
    global_cond = torch.zeros(batch_size, global_cond_dim, device=device)
    return (sample, timestep, global_cond)


def _make_diffusion_ddim_inputs(policy, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Sample inputs for ``DiffusionDDIMWrapper.forward``."""
    config = policy.config
    action_dim = config.action_feature.shape[0]
    horizon = config.horizon
    global_cond_dim = _get_diffusion_global_cond_dim(policy)

    noise = torch.zeros(batch_size, horizon, action_dim, device=device)
    global_cond = torch.zeros(batch_size, global_cond_dim, device=device)
    return (noise, global_cond)


# ──────────────────────────────────────────────────────────────────────────────
# Generic
# ──────────────────────────────────────────────────────────────────────────────


def _make_generic_inputs(config, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Generic sample inputs derived from ``config.input_features``."""
    inputs: list[Tensor] = []
    n_obs_steps = getattr(config, "n_obs_steps", 1)

    # Non-image features first
    for key, ft in (config.input_features or {}).items():
        if "image" in key.lower():
            continue
        if n_obs_steps > 1:
            inputs.append(torch.zeros(batch_size, n_obs_steps, *ft.shape, device=device))
        else:
            inputs.append(torch.zeros(batch_size, *ft.shape, device=device))

    # Image features
    for key, ft in (config.input_features or {}).items():
        if "image" not in key.lower():
            continue
        if n_obs_steps > 1:
            # (B, n_obs_steps, C, H, W) — cameras will be stacked inside GenericWrapper
            inputs.append(torch.zeros(batch_size, n_obs_steps, *ft.shape, device=device))
        else:
            inputs.append(torch.zeros(batch_size, *ft.shape, device=device))

    return tuple(inputs)
