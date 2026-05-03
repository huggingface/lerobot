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
"""ONNX export adapter for Diffusion Policy.

Two modes are available, selected by the per-policy CLI knob
``--policy-options.mode=...``:

- ``"unet-only"`` (default): Exports a single denoising step of
  ``DiffusionConditionalUnet1d``. Caller runs the scheduler loop in Python.
- ``"ddim-N"``: Bakes a deterministic N-step DDIM loop into the ONNX graph.

Auto-discovered by ``lerobot.export.core.make_export_wrapper`` via the
naming convention ``policies/<type>/export_<type>.py:make_<type>_export_wrapper``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.export.adapters import IterativeDenoisingAdapter
from lerobot.export.core import ExportSpec, make_batch_dynamic_axes_and_shapes
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

if TYPE_CHECKING:
    from .modeling_diffusion import (
        DiffusionConditionalUnet1d,
        DiffusionModel,
        DiffusionPolicy,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────


class DiffusionUNetWrapper(nn.Module):
    """ONNX-compatible wrapper for a single Diffusion UNet denoising step.

    Only the UNet is exported. The caller is responsible for:

    - Running the denoising loop (N steps with a DDPM/DDIM scheduler).
    - Computing ``global_cond`` via ``policy.diffusion._prepare_global_conditioning``.

    Supports dynamic batch_size.
    """

    def __init__(self, unet: DiffusionConditionalUnet1d) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        """
        Args:
            sample:      ``(B, horizon, action_dim)`` noisy action trajectory.
            timestep:    ``(B,)`` int64 diffusion timestep indices.
            global_cond: ``(B, global_cond_dim)`` conditioning vector.

        Returns:
            ``(B, horizon, action_dim)`` model prediction (noise or original sample,
            depending on ``prediction_type``).
        """
        return self.unet(sample, timestep, global_cond=global_cond)


class DiffusionDDIMWrapper(IterativeDenoisingAdapter):
    """N-step deterministic DDIM denoising loop, unrolled into a single ONNX graph.

    Subclass of :class:`IterativeDenoisingAdapter`; the per-step model call
    invokes the UNet, and ``_step`` implements the DDIM (eta=0) update rule.

    Restrictions:

    - Deterministic DDIM only (eta=0, no added noise).
    - The number of denoising steps is fixed at export time.
    - ``prediction_type`` (`epsilon` / `v_prediction` / `sample`) is read from
      the policy's diffusion config and baked in.

    Supports dynamic batch_size at runtime.
    """

    def __init__(self, diffusion_model: DiffusionModel, num_ddim_steps: int) -> None:
        super().__init__(num_steps=num_ddim_steps)
        self.unet = diffusion_model.unet
        self.prediction_type: str = diffusion_model.config.prediction_type
        self.clip_sample: bool = diffusion_model.config.clip_sample
        self.clip_sample_range: float = diffusion_model.config.clip_sample_range

        # Pre-compute DDIM schedule and register as buffers so they are exported
        # as constants in the ONNX graph.
        scheduler = diffusion_model.noise_scheduler
        scheduler.set_timesteps(num_ddim_steps)
        timesteps = scheduler.timesteps  # (num_ddim_steps,) long tensor

        alphas_cumprod = scheduler.alphas_cumprod  # (num_train_timesteps,)
        alpha_t = alphas_cumprod[timesteps]  # (num_ddim_steps,)
        # prev_timesteps: shift by one step in the schedule.
        step_ratio = scheduler.config.num_train_timesteps // num_ddim_steps
        prev_timesteps = (timesteps - step_ratio).clamp(min=0)
        alpha_t_prev = alphas_cumprod[prev_timesteps]  # (num_ddim_steps,)

        self.register_buffer("timesteps_buf", timesteps)
        self.register_buffer("alpha_t_buf", alpha_t.float())
        self.register_buffer("alpha_t_prev_buf", alpha_t_prev.float())

    def _call_model(self, sample: Tensor, step_idx: int, *cond: Tensor) -> Tensor:
        """One UNet forward at the given step.

        ``cond`` is expected to be ``(global_cond,)`` — a single (B, D) tensor.
        """
        (global_cond,) = cond
        t = self.timesteps_buf[step_idx : step_idx + 1].expand(sample.shape[0])
        return self.unet(sample, t, global_cond=global_cond)

    def _step(self, sample: Tensor, model_output: Tensor, step_idx: int) -> Tensor:
        """One deterministic DDIM step (eta=0), implemented as pure tensor math."""
        alpha_t = self.alpha_t_buf[step_idx]
        alpha_t_prev = self.alpha_t_prev_buf[step_idx]
        beta_t = 1.0 - alpha_t

        if self.prediction_type == "epsilon":
            pred_original = (sample - beta_t**0.5 * model_output) / alpha_t**0.5
        elif self.prediction_type == "v_prediction":
            pred_original = alpha_t**0.5 * sample - beta_t**0.5 * model_output
        elif self.prediction_type == "sample":
            pred_original = model_output
        else:
            raise ValueError(f"Unsupported prediction_type '{self.prediction_type}'")

        if self.clip_sample:
            pred_original = pred_original.clamp(-self.clip_sample_range, self.clip_sample_range)

        if self.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.prediction_type == "v_prediction":
            pred_epsilon = alpha_t**0.5 * model_output + beta_t**0.5 * sample
        else:  # "sample"
            pred_epsilon = (sample - alpha_t**0.5 * pred_original) / beta_t**0.5

        pred_sample_direction = (1.0 - alpha_t_prev) ** 0.5 * pred_epsilon
        return alpha_t_prev**0.5 * pred_original + pred_sample_direction


# ──────────────────────────────────────────────────────────────────────────────
# Sample input helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_global_cond_dim(policy: DiffusionPolicy) -> int:
    """Compute the global_cond_dim expected by the Diffusion UNet.

    Builds a tiny dummy batch and runs it through ``_prepare_global_conditioning``.
    """
    config = policy.config
    n_obs_steps = config.n_obs_steps
    device = next(policy.parameters()).device

    dummy_batch: dict[str, Tensor] = {}
    if config.robot_state_feature:
        state_dim = config.robot_state_feature.shape[0]
        dummy_batch[OBS_STATE] = torch.zeros(1, n_obs_steps, state_dim, device=device)

    if config.image_features:
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


def _make_unet_sample_inputs(
    policy: DiffusionPolicy, batch_size: int, device: torch.device
) -> tuple[Tensor, ...]:
    """Sample inputs for ``DiffusionUNetWrapper.forward``."""
    config = policy.config
    action_dim = config.action_feature.shape[0]
    horizon = config.horizon
    global_cond_dim = _get_global_cond_dim(policy)

    sample = torch.zeros(batch_size, horizon, action_dim, device=device)
    timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
    global_cond = torch.zeros(batch_size, global_cond_dim, device=device)
    return (sample, timestep, global_cond)


def _make_ddim_sample_inputs(
    policy: DiffusionPolicy, batch_size: int, device: torch.device
) -> tuple[Tensor, ...]:
    """Sample inputs for ``DiffusionDDIMWrapper.forward``."""
    config = policy.config
    action_dim = config.action_feature.shape[0]
    horizon = config.horizon
    global_cond_dim = _get_global_cond_dim(policy)

    noise = torch.zeros(batch_size, horizon, action_dim, device=device)
    global_cond = torch.zeros(batch_size, global_cond_dim, device=device)
    return (noise, global_cond)


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────


def make_diffusion_export_wrapper(policy: DiffusionPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Build (wrapper, ExportSpec) for Diffusion Policy export.

    Mode is selected via ``cfg.policy_options["mode"]`` (CLI:
    ``--policy-options.mode=...``):

    - ``"unet-only"`` (default): single denoising step.
    - ``"ddim-N"``: full N-step deterministic DDIM loop, unrolled into the graph.

    Auto-discovered by ``lerobot.export.core.make_export_wrapper``.
    """
    diffusion_model = policy.diffusion
    # Unwrap torch.compile if present.
    raw_unet = getattr(diffusion_model.unet, "_orig_mod", diffusion_model.unet)
    device = torch.device(getattr(cfg, "device", "cpu"))
    batch_size = getattr(cfg, "batch_size", 1)

    policy_options = getattr(cfg, "policy_options", {}) or {}
    diffusion_mode: str = policy_options.get("mode", "unet-only")

    if diffusion_mode == "unet-only":
        wrapper = DiffusionUNetWrapper(raw_unet)
        wrapper.eval()
        sample_inputs = _make_unet_sample_inputs(policy, batch_size, device)
        input_names = ["sample", "timestep", "global_cond"]
        output_names = ["denoised"]
        dynamic_axes, dynamic_shapes = make_batch_dynamic_axes_and_shapes(
            input_names=input_names,
            sample_inputs=sample_inputs,
            output_names=output_names,
        )
        note = (
            "Exports the UNet only (single denoising step). "
            "Run the DDPM/DDIM scheduler loop in Python and call this model at each step. "
            "Compute global_cond via policy.diffusion._prepare_global_conditioning before the loop."
        )
        return wrapper, ExportSpec(
            input_names=input_names,
            output_names=output_names,
            sample_inputs=sample_inputs,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
            policy_note=note,
        )

    # Full DDIM mode: "ddim-N" where N is the number of steps.
    if not diffusion_mode.startswith("ddim-"):
        raise ValueError(
            f"Invalid policy_options.mode='{diffusion_mode}' for diffusion. "
            "Use 'unet-only' or 'ddim-N' where N is the number of DDIM steps (e.g. 'ddim-10')."
        )
    if diffusion_model.config.noise_scheduler_type != "DDIM":
        raise ValueError(
            f"Full DDIM export requires noise_scheduler_type='DDIM', "
            f"but found '{diffusion_model.config.noise_scheduler_type}'. "
            "Use --policy-options.mode=unet-only for DDPM policies."
        )
    num_steps = int(diffusion_mode.split("-")[1])
    wrapper = DiffusionDDIMWrapper(diffusion_model, num_steps)
    wrapper.eval()
    sample_inputs = _make_ddim_sample_inputs(policy, batch_size, device)
    input_names = ["noise", "global_cond"]
    output_names = ["action_chunk"]
    dynamic_axes, dynamic_shapes = make_batch_dynamic_axes_and_shapes(
        input_names=input_names,
        sample_inputs=sample_inputs,
        output_names=output_names,
    )
    note = (
        f"Full {num_steps}-step DDIM loop baked into the ONNX graph. "
        "Number of steps is fixed at export time. "
        "Compute global_cond via policy.diffusion._prepare_global_conditioning before calling."
    )
    return wrapper, ExportSpec(
        input_names=input_names,
        output_names=output_names,
        sample_inputs=sample_inputs,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        policy_note=note,
    )
