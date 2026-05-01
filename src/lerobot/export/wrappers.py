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
"""ONNX-compatible wrapper modules for LeRobot policies.

Each wrapper accepts flat tensor arguments (no dicts, no Python lists) so that
``torch.onnx.export`` can trace them cleanly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACT
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionModel
    from lerobot.policies.pretrained import PreTrainedPolicy

from .core import ExportSpec

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _key_to_input_name(key: str) -> str:
    """Convert a feature key like 'observation.images.cam_high' to 'observation_images_cam_high'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", key)


# ──────────────────────────────────────────────────────────────────────────────
# ACT
# ──────────────────────────────────────────────────────────────────────────────


class ACTInferenceWrapper(nn.Module):
    """ONNX-compatible wrapper for ACT inference.

    Exports the backbone + transformer encoder/decoder + action head.
    Excluded: VAE encoder (latent is zeros at inference), action queue, temporal ensembler.

    Note: Exported with batch_size=1 fixed. Inside ACT.forward, a
    ``torch.zeros([batch_size, latent_dim])`` call is traced as a constant, so the
    resulting ONNX model is only valid for batch_size=1. This matches the typical
    real-time robot inference use case.
    """

    def __init__(self, act_model: ACT, config) -> None:
        super().__init__()
        self.model = act_model
        self.config = config
        self._has_robot_state = config.robot_state_feature is not None
        self._has_env_state = config.env_state_feature is not None
        self._image_keys: list[str] = list(config.image_features.keys()) if config.image_features else []

    def forward(self, robot_state: Tensor, *camera_images: Tensor) -> Tensor:
        """
        Args:
            robot_state: ``(1, state_dim)`` robot joint state. If the policy has no
                ``robot_state_feature`` and uses ``env_state_feature`` instead, pass
                the environment state here with the same shape contract.
            *camera_images: One ``(1, C, H, W)`` tensor per camera, in the order
                defined by ``config.image_features``.

        Returns:
            ``(1, chunk_size, action_dim)`` predicted action chunk.
        """
        batch: dict[str, Tensor] = {}

        if self._has_robot_state:
            batch[OBS_STATE] = robot_state
        elif self._has_env_state:
            batch[OBS_ENV_STATE] = robot_state  # env_state passed in robot_state slot

        if self._image_keys:
            # ACT.forward expects OBS_IMAGES as a Python list of per-camera tensors.
            batch[OBS_IMAGES] = list(camera_images)

        actions, _ = self.model(batch)  # VAE encoder skipped (eval mode, latent = zeros)
        return actions  # (1, chunk_size, action_dim)


def _make_act_wrapper(policy: PreTrainedPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Build an ACTInferenceWrapper and the corresponding ExportSpec."""
    from torch.export import Dim

    from .sample_inputs import make_sample_inputs

    act_model = policy.model
    config = policy.config

    act_model.eval()
    wrapper = ACTInferenceWrapper(act_model, config)
    wrapper.eval()

    # Input names: robot_state (or env_state), then each camera
    input_names: list[str] = []
    if config.robot_state_feature:
        input_names.append("observation_state")
    elif config.env_state_feature:
        input_names.append("observation_env_state")

    for key in (config.image_features or {}):
        input_names.append(_key_to_input_name(key))

    sample_inputs = make_sample_inputs(policy, cfg, mode="act")

    # Legacy tracing bakes batch_size=1 because of `torch.zeros([batch_size, latent_dim])`
    # inside ACT.forward; the dynamo path (torch.export) symbolicizes it via Dim.
    # Use tuple form (matching positional sample_inputs) so it works with both
    # the regular and *args-style wrapper signatures.
    batch_dim = Dim("batch_size", min=1, max=64)
    dynamic_shapes = tuple({0: batch_dim} for _ in sample_inputs)

    return wrapper, ExportSpec(
        input_names=input_names,
        output_names=["action_chunk"],
        sample_inputs=sample_inputs,
        dynamic_axes=None,  # batch_size=1 fixed under legacy tracing
        dynamic_shapes=dynamic_shapes,  # batch_size dynamic under dynamo exporter
        policy_note=(
            "Exports ACT backbone + Transformer encoder/decoder + action head. "
            "Action queue and temporal ensembler must be managed in Python. "
            "batch_size is fixed=1 under exporter='legacy', dynamic under exporter='dynamo'."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion — UNet only
# ──────────────────────────────────────────────────────────────────────────────


class DiffusionUNetWrapper(nn.Module):
    """ONNX-compatible wrapper for a single Diffusion UNet denoising step.

    Only the UNet is exported. The caller is responsible for:
    - Running the denoising loop (N steps with a DDPM/DDIM scheduler)
    - Computing ``global_cond`` via ``policy.diffusion._prepare_global_conditioning``

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


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion — Full DDIM loop
# ──────────────────────────────────────────────────────────────────────────────


class DiffusionDDIMWrapper(nn.Module):
    """ONNX-compatible wrapper that bakes a full N-step DDIM denoising loop.

    The loop is unrolled at trace time so that both the UNet calls and the
    scheduler arithmetic end up as a single static ONNX graph.

    Restrictions:
    - Deterministic DDIM only (eta=0, no added noise).
    - The number of denoising steps (``num_ddim_steps``) is fixed at export time.
    - ``prediction_type`` is read from ``config`` and baked in.

    Supports dynamic batch_size.
    """

    def __init__(self, diffusion_model: DiffusionModel, num_ddim_steps: int) -> None:
        super().__init__()
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
        # Gather alpha_prod_t and alpha_prod_t_prev for each DDIM step
        alpha_t = alphas_cumprod[timesteps]  # (num_ddim_steps,)
        # prev_timesteps: shift by one step in the schedule
        step_ratio = scheduler.config.num_train_timesteps // num_ddim_steps
        prev_timesteps = (timesteps - step_ratio).clamp(min=0)
        alpha_t_prev = alphas_cumprod[prev_timesteps]  # (num_ddim_steps,)

        self.register_buffer("timesteps_buf", timesteps)
        self.register_buffer("alpha_t_buf", alpha_t.float())
        self.register_buffer("alpha_t_prev_buf", alpha_t_prev.float())
        self.num_ddim_steps = num_ddim_steps

    def _ddim_step(self, sample: Tensor, model_output: Tensor, step_idx: int) -> Tensor:
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

        # Compute noise prediction (epsilon) for direction
        if self.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.prediction_type == "v_prediction":
            pred_epsilon = alpha_t**0.5 * model_output + beta_t**0.5 * sample
        else:  # "sample"
            pred_epsilon = (sample - alpha_t**0.5 * pred_original) / beta_t**0.5

        pred_sample_direction = (1.0 - alpha_t_prev) ** 0.5 * pred_epsilon
        return alpha_t_prev**0.5 * pred_original + pred_sample_direction

    def forward(self, noise: Tensor, global_cond: Tensor) -> Tensor:
        """
        Args:
            noise:       ``(B, horizon, action_dim)`` Gaussian noise.
            global_cond: ``(B, global_cond_dim)`` conditioning vector.

        Returns:
            ``(B, horizon, action_dim)`` denoised action trajectory (all steps baked in).
        """
        sample = noise
        for i in range(self.num_ddim_steps):
            t = self.timesteps_buf[i : i + 1].expand(sample.shape[0])
            model_output = self.unet(sample, t, global_cond=global_cond)
            sample = self._ddim_step(sample, model_output, i)
        return sample


def _make_diffusion_wrapper(
    policy: PreTrainedPolicy, cfg
) -> tuple[nn.Module, ExportSpec]:
    """Build a Diffusion wrapper according to ``cfg.diffusion_mode``."""
    from torch.export import Dim

    from .sample_inputs import make_sample_inputs

    diffusion_model = policy.diffusion
    # Unwrap torch.compile if present
    raw_unet = getattr(diffusion_model.unet, "_orig_mod", diffusion_model.unet)

    diffusion_mode: str = getattr(cfg, "diffusion_mode", "unet-only")
    batch_dim = Dim("batch_size", min=1, max=64)

    if diffusion_mode == "unet-only":
        wrapper = DiffusionUNetWrapper(raw_unet)
        wrapper.eval()
        sample_inputs = make_sample_inputs(policy, cfg, mode="diffusion-unet")
        dynamic_axes = {
            "sample": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "global_cond": {0: "batch_size"},
            "denoised": {0: "batch_size"},
        }
        # Tuple form matching positional sample_inputs (sample, timestep, global_cond).
        dynamic_shapes = tuple({0: batch_dim} for _ in sample_inputs)
        note = (
            "Exports the UNet only (single denoising step). "
            "Run the DDPM/DDIM scheduler loop in Python and call this model at each step. "
            "Compute global_cond via policy.diffusion._prepare_global_conditioning before the loop."
        )
        return wrapper, ExportSpec(
            input_names=["sample", "timestep", "global_cond"],
            output_names=["denoised"],
            sample_inputs=sample_inputs,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
            policy_note=note,
        )

    # Full DDIM mode: "ddim-N" where N is the number of steps.
    if not diffusion_mode.startswith("ddim-"):
        raise ValueError(
            f"Invalid diffusion_mode '{diffusion_mode}'. "
            "Use 'unet-only' or 'ddim-N' where N is the number of DDIM steps (e.g. 'ddim-10')."
        )
    if diffusion_model.config.noise_scheduler_type != "DDIM":
        raise ValueError(
            f"Full DDIM export requires noise_scheduler_type='DDIM', "
            f"but found '{diffusion_model.config.noise_scheduler_type}'. "
            "Use --diffusion-mode=unet-only for DDPM policies."
        )
    num_steps = int(diffusion_mode.split("-")[1])
    wrapper = DiffusionDDIMWrapper(diffusion_model, num_steps)
    wrapper.eval()
    sample_inputs = make_sample_inputs(policy, cfg, mode="diffusion-ddim")
    dynamic_axes = {
        "noise": {0: "batch_size"},
        "global_cond": {0: "batch_size"},
        "action_chunk": {0: "batch_size"},
    }
    # Tuple form matching positional sample_inputs (noise, global_cond).
    dynamic_shapes = tuple({0: batch_dim} for _ in sample_inputs)
    note = (
        f"Full {num_steps}-step DDIM loop baked into the ONNX graph. "
        "Number of steps is fixed at export time. "
        "Compute global_cond via policy.diffusion._prepare_global_conditioning before calling."
    )
    return wrapper, ExportSpec(
        input_names=["noise", "global_cond"],
        output_names=["action_chunk"],
        sample_inputs=sample_inputs,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        policy_note=note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Generic fallback
# ──────────────────────────────────────────────────────────────────────────────


# Policy types that have non-standard forward signatures or stateful internals
# incompatible with the generic ACT-clone wrapper. These require a custom factory
# registered via lerobot.export.register_export_wrapper().
_UNSUPPORTED_GENERIC_POLICY_TYPES: frozenset[str] = frozenset(
    {"sac", "vqbet", "tdmpc", "pi0", "pi0fast", "smolvla"}
)


class GenericPolicyWrapper(nn.Module):
    """Stateless fallback wrapper for ACT-clone style policies (n_obs_steps=1).

    Bypasses internal observation queues and calls ``policy.model`` directly.
    Only supports single-step policies whose ``policy.model`` accepts a dict
    with the ACT image convention (``OBS_IMAGES`` as a list of per-camera tensors).

    For policies with non-standard forward signatures (SAC, VQBET, TDMPC, PI0,
    SmoLVLA), register a custom factory:

        from lerobot.export import register_export_wrapper
        @register_export_wrapper("my_policy_type")
        def my_factory(policy, cfg): ...
    """

    def __init__(self, policy_model: nn.Module, input_feature_keys: list[str]) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.input_feature_keys = input_feature_keys
        self.image_keys = [k for k in input_feature_keys if "image" in k.lower()]
        self.non_image_keys = [k for k in input_feature_keys if "image" not in k.lower()]

    def forward(self, *args: Tensor) -> Tensor:
        batch: dict[str, Tensor | list[Tensor]] = {}
        for key, tensor in zip(self.input_feature_keys, args, strict=False):
            batch[key] = tensor

        if self.image_keys:
            image_tensors = [batch.pop(k) for k in self.image_keys]
            # ACT convention: list of (B, C, H, W) per camera.
            batch[OBS_IMAGES] = image_tensors

        result = self.policy_model(batch)
        if isinstance(result, tuple):
            return result[0]
        return result


def _make_generic_wrapper(policy: PreTrainedPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Build a GenericPolicyWrapper for ACT-clone style policies only.

    Raises ``NotImplementedError`` for policies known to be incompatible with the
    generic wrapper, with a pointer to the plugin registration API.
    """
    from .sample_inputs import make_sample_inputs

    config = policy.config
    n_obs_steps: int = getattr(config, "n_obs_steps", 1)
    policy_type = config.type

    # Hard-blocked policy types — explicit error pointing to the registration API.
    if policy_type in _UNSUPPORTED_GENERIC_POLICY_TYPES:
        raise NotImplementedError(
            f"Policy type '{policy_type}' has a non-standard forward signature and cannot "
            "use the generic export wrapper. Register a custom factory:\n"
            "  from lerobot.export import register_export_wrapper\n"
            f"  @register_export_wrapper('{policy_type}')\n"
            "  def my_factory(policy, cfg):\n"
            "      ...  # build (wrapper, ExportSpec)"
        )

    # Soft requirements: must look like an ACT-clone (has policy.model + n_obs_steps == 1).
    if not (hasattr(policy, "model") and callable(getattr(policy, "model", None))):
        raise NotImplementedError(
            f"Policy '{policy_type}' has no callable '.model' attribute; cannot use the "
            "generic wrapper. Register a custom factory via "
            f"register_export_wrapper('{policy_type}', my_factory)."
        )
    if n_obs_steps != 1:
        raise NotImplementedError(
            f"Policy '{policy_type}' has n_obs_steps={n_obs_steps}; the generic wrapper only "
            "supports single-step policies. Register a custom factory via "
            f"register_export_wrapper('{policy_type}', my_factory)."
        )

    # Collect ordered input feature keys (non-image first, then images).
    non_image_keys = [k for k in config.input_features if "image" not in k.lower()]
    image_keys = [k for k in config.input_features if "image" in k.lower()]
    ordered_keys = non_image_keys + image_keys

    wrapper = GenericPolicyWrapper(policy.model, ordered_keys)
    wrapper.eval()

    input_names = [_key_to_input_name(k) for k in ordered_keys]
    sample_inputs = make_sample_inputs(policy, cfg, mode="generic")

    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes["action_chunk"] = {0: "batch_size"}

    return wrapper, ExportSpec(
        input_names=input_names,
        output_names=["action_chunk"],
        sample_inputs=sample_inputs,
        dynamic_axes=dynamic_axes,
        policy_note=(
            f"Generic ACT-clone wrapper for policy type '{policy_type}'. "
            "Calls policy.model directly, bypassing observation queues. "
            "Only validated for ACT-style architectures."
        ),
    )
