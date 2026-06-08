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

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .configuration_cosmos3 import Cosmos3Config
from .processor_cosmos3 import (
    COSMOS3_ACTION_CONDITION,
    COSMOS3_ACTION_CONDITION_MASK,
    COSMOS3_ACTION_DOMAIN_ID,
    COSMOS3_CLEAN_ACTION,
    COSMOS3_COND_INPUT_IDS,
    COSMOS3_CONDITIONING_FPS,
    COSMOS3_RAW_ACTION_DIM,
    COSMOS3_TRAINING_SIGMA,
    COSMOS3_UNCOND_INPUT_IDS,
    COSMOS3_VIDEO,
    classify_cosmos3_action_size,
)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported Cosmos3 dtype={dtype_name!r}")


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _retrieve_latents(encoder_output: Any, *, sample_mode: str = "argmax") -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        latent_dist = encoder_output.latent_dist
    elif isinstance(encoder_output, tuple):
        latent_dist = encoder_output[0]
    else:
        raise TypeError(f"Unexpected VAE encoder output type: {type(encoder_output)!r}")

    if sample_mode == "argmax":
        return latent_dist.mode()
    if sample_mode == "sample":
        return latent_dist.sample()
    raise ValueError(f"Unsupported VAE latent sample_mode={sample_mode!r}.")


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)

    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    next_temporal_offset = temporal_offset + num_tokens
    return mrope_ids, next_temporal_offset


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_temporal_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_temporal_offset


def _seeded_standard_normal(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    random_array = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


def preprocess_action_video_batch(
    videos: Tensor,
    *,
    resolution_tier: int,
    num_frames: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, int, int]:
    if videos.dtype != torch.uint8:
        raise ValueError(f"Cosmos3 action video input must be uint8, got dtype={videos.dtype}.")
    if videos.ndim != 5:
        raise ValueError(f"Expected Cosmos3 action video shape [B,C,T,H,W], got shape={tuple(videos.shape)}.")

    frames = videos.detach().to(device=device)
    batch_size, channels, _time, source_h, source_w = frames.shape
    target_h, target_w, content_h, content_w = classify_cosmos3_action_size(
        source_h,
        source_w,
        resolution_tier=resolution_tier,
    )

    if frames.shape[2] < num_frames:
        frames = torch.cat(
            [frames, frames[:, :, -1:].expand(-1, -1, num_frames - frames.shape[2], -1, -1)],
            dim=2,
        )
    else:
        frames = frames[:, :, :num_frames]

    frames_t = frames.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, source_h, source_w)
    frames_t = frames_t.to(dtype=torch.float32)
    if content_h != source_h or content_w != source_w:
        frames_t = F.interpolate(
            frames_t,
            size=(content_h, content_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    pad_right = target_w - content_w
    pad_bottom = target_h - content_h
    if pad_right or pad_bottom:
        pad_mode = "replicate" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
        frames_t = F.pad(frames_t, (0, pad_right, 0, pad_bottom), mode=pad_mode)

    frames = frames_t.reshape(batch_size, num_frames, channels, target_h, target_w)
    frames = frames.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype) / 127.5 - 1.0
    image_size = torch.tensor([target_h, target_w, content_h, content_w], device=device, dtype=torch.float32)
    return frames, image_size, target_h, target_w


class Cosmos3Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for Cosmos3 DROID action generation."""

    config_class = Cosmos3Config
    name = "cosmos3"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, *args, config=None, **kwargs):
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        if not isinstance(config, Cosmos3Config):
            raise TypeError(f"Expected Cosmos3Config, got {type(config)!r}.")
        config.pretrained_path = Path(pretrained_name_or_path)
        policy = super().from_pretrained(pretrained_name_or_path, *args, config=config, **kwargs)
        policy.model.ensure_runtime_dtypes()
        return policy

    def __init__(self, config: Cosmos3Config, **kwargs):
        require_package("diffusers", extra="cosmos3")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = Cosmos3ActionModel(
            config,
            transformer=kwargs.pop("transformer", None),
            vae=kwargs.pop("vae", None),
            scheduler=kwargs.pop("scheduler", None),
        )
        self.model.ensure_runtime_dtypes()
        self.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.model.reset_generation()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.model(batch)

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        return self.model.sample_actions(batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        actions = self.sample_actions(batch, **kwargs).to(torch.float32)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_future_video(self, batch: dict[str, Tensor], **kwargs) -> Tensor | None:
        return self.model.predict_future_video(batch, **kwargs)

    def get_optim_params(self) -> dict:
        return self.parameters()


class Cosmos3ActionModel(nn.Module):
    """Cosmos3 action model built from public Diffusers model-level components."""

    def __init__(
        self,
        config: Cosmos3Config,
        *,
        transformer: nn.Module | None = None,
        vae: nn.Module | None = None,
        scheduler: Any | None = None,
    ):
        super().__init__()
        self.config = config
        self.transformer = transformer if transformer is not None else self._build_transformer()
        self.vae = vae if vae is not None else self._build_vae()
        self.scheduler = scheduler if scheduler is not None else self._build_scheduler()
        if self.config.freeze_vae:
            self.vae.eval().requires_grad_(False)
        self.reset_generation()

    def ensure_runtime_dtypes(self) -> None:
        for module_name in getattr(self.transformer, "_keep_in_fp32_modules", []) or []:
            module = getattr(self.transformer, module_name, None)
            if module is not None:
                module.float()
        rotary_emb = getattr(self.transformer, "rotary_emb", None)
        inv_freq = getattr(rotary_emb, "inv_freq", None)
        if inv_freq is not None and inv_freq.device.type != "meta" and inv_freq.dtype != torch.float32:
            rotary_emb.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def reset_generation(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)

    def _next_seed(self) -> int:
        if self.config.deterministic_seed:
            return int(self.config.seed)
        return int(self._rng.integers(0, 2**31))

    def _build_transformer(self) -> nn.Module:
        from diffusers import Cosmos3OmniTransformer

        torch_dtype = _torch_dtype(self.config.dtype)
        transformer = Cosmos3OmniTransformer(**self.config.transformer_backbone_config)
        return transformer.to(dtype=torch_dtype)

    def _build_vae(self) -> nn.Module:
        from diffusers import AutoencoderKLWan

        torch_dtype = _torch_dtype(self.config.dtype)
        if self.config.vae_config is not None:
            return AutoencoderKLWan(**self.config.vae_config).to(dtype=torch_dtype)
        raise ValueError(
            "Cosmos3Config.vae_config is required. "
            "Load a converted LeRobot Cosmos3 checkpoint or provide the serialized VAE config."
        )

    def _build_scheduler(self) -> Any:
        from diffusers import UniPCMultistepScheduler

        if self.config.scheduler_config is None:
            raise ValueError(
                "Cosmos3Config.scheduler_config is required. "
                "Load a converted LeRobot Cosmos3 checkpoint or provide the serialized scheduler config."
            )
        return UniPCMultistepScheduler.from_config(self.config.scheduler_config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        required = [
            COSMOS3_VIDEO,
            COSMOS3_ACTION_CONDITION,
            COSMOS3_ACTION_CONDITION_MASK,
            COSMOS3_ACTION_DOMAIN_ID,
            COSMOS3_CONDITIONING_FPS,
            COSMOS3_RAW_ACTION_DIM,
            COSMOS3_COND_INPUT_IDS,
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Cosmos3 training batch is missing required model inputs: {missing}")

        videos = batch[COSMOS3_VIDEO]
        if videos.ndim == 4:
            videos = videos.unsqueeze(0)
        action_conditions = batch[COSMOS3_ACTION_CONDITION]
        if action_conditions.ndim == 2:
            action_conditions = action_conditions.unsqueeze(0)
        action_condition_masks = batch[COSMOS3_ACTION_CONDITION_MASK]
        if action_condition_masks.ndim == 2:
            action_condition_masks = action_condition_masks.unsqueeze(0)

        batch_size = videos.shape[0]
        clean_actions = self._prepare_clean_action_tokens(batch, action_conditions)
        sigmas = self._get_training_sigmas(batch, batch_size=batch_size)

        cond_input_ids = [
            self._get_ids_for_batch(batch[COSMOS3_COND_INPUT_IDS], batch_idx)
            for batch_idx in range(batch_size)
        ]
        losses = self._compute_training_loss(
            cond_input_ids=cond_input_ids,
            videos=videos,
            clean_action=clean_actions,
            action_condition_mask=action_condition_masks,
            domain_id=batch[COSMOS3_ACTION_DOMAIN_ID],
            conditioning_fps=batch[COSMOS3_CONDITIONING_FPS],
            raw_action_dim=batch[COSMOS3_RAW_ACTION_DIM],
            sigma=sigmas,
        )

        loss = losses["loss"]
        vision_loss = losses["flow_matching_loss_vision"]
        action_loss = losses["flow_matching_loss_action"]
        metrics = {
            "loss": float(loss.detach().cpu()),
            "flow_matching_loss_vision": float(vision_loss.detach().cpu()),
            "flow_matching_loss_action": float(action_loss.detach().cpu()),
        }
        return loss, metrics

    def _get_ids_for_batch(self, input_ids: Any, batch_idx: int) -> Tensor:
        if isinstance(input_ids, Tensor):
            if input_ids.ndim == 1:
                return input_ids
            return input_ids[batch_idx]
        return input_ids[batch_idx]

    def _prepare_clean_action_tokens(self, batch: dict[str, Tensor], action_conditions: Tensor) -> Tensor:
        if COSMOS3_CLEAN_ACTION in batch:
            clean_action = batch[COSMOS3_CLEAN_ACTION]
            if clean_action.ndim == 2:
                clean_action = clean_action.unsqueeze(0)
            if clean_action.shape[-1] < self.config.max_action_dim:
                clean_action = F.pad(clean_action, (0, self.config.max_action_dim - clean_action.shape[-1]))
            return clean_action[:, :, : self.config.max_action_dim].to(dtype=torch.float32)

        if ACTION not in batch:
            raise ValueError(
                f"Cosmos3 training requires {COSMOS3_CLEAN_ACTION!r} or {ACTION!r} action labels."
            )
        action = batch[ACTION]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Cosmos3 action labels must have shape [B,T,D], got {tuple(action.shape)}.")

        batch_size, action_len, _ = action_conditions.shape
        clean_action = torch.zeros(
            batch_size,
            action_len,
            self.config.max_action_dim,
            dtype=torch.float32,
            device=action_conditions.device,
        )
        clean_action[:, :, : self.config.raw_action_dim] = action_conditions[
            :, :, : self.config.raw_action_dim
        ].to(dtype=torch.float32)
        action = (
            action[:, : self.config.chunk_size, : self.config.raw_action_dim].to(dtype=torch.float32).clone()
        )
        if self.config.invert_gripper:
            action[:, :, -1] = 1.0 - action[:, :, -1]
        future_start = int(self.config.use_state)
        clean_action[:, future_start : future_start + action.shape[1], : self.config.raw_action_dim] = action
        return clean_action

    def _get_training_sigmas(self, batch: dict[str, Tensor], *, batch_size: int) -> Tensor:
        device = _module_device(self.transformer)
        if COSMOS3_TRAINING_SIGMA in batch:
            sigmas = torch.as_tensor(batch[COSMOS3_TRAINING_SIGMA], device=device, dtype=torch.float32)
            if sigmas.ndim == 0:
                sigmas = sigmas.expand(batch_size, 1)
            elif sigmas.ndim == 1:
                sigmas = sigmas.view(batch_size, 1)
            elif sigmas.ndim != 2:
                raise ValueError(
                    f"{COSMOS3_TRAINING_SIGMA} must be scalar, [B], or [B,1], got {sigmas.shape}."
                )
            return sigmas

        if self.config.train_time_video_distribution == "uniform":
            t_raw = torch.rand((batch_size, 1), device=device, dtype=torch.float32)
        elif self.config.train_time_video_distribution == "logitnormal":
            t_raw = torch.sigmoid(torch.randn((batch_size, 1), device=device, dtype=torch.float32))
        elif self.config.train_time_video_distribution == "waver":
            u = torch.rand((batch_size, 1), device=device, dtype=torch.float32)
            t_raw = 1.0 - u - 1.29 * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
        else:
            raise ValueError(
                f"Unsupported Cosmos3 train_time_video_distribution={self.config.train_time_video_distribution!r}."
            )

        tau = 1.0 - t_raw
        shift = float(self.config.shift)
        return shift * tau / (1.0 + (shift - 1.0) * tau)

    def _compute_training_loss(
        self,
        *,
        cond_input_ids: list[Tensor],
        videos: Tensor,
        clean_action: Tensor,
        action_condition_mask: Tensor,
        domain_id: Tensor,
        conditioning_fps: Tensor,
        raw_action_dim: Tensor,
        sigma: Tensor,
    ) -> dict[str, Tensor]:
        device = _module_device(self.transformer)
        dtype = _module_dtype(self.transformer)
        batch_size = videos.shape[0]
        raw_action_dims = torch.as_tensor(raw_action_dim, device=device, dtype=torch.long).view(batch_size)
        conditioning_fps = torch.as_tensor(conditioning_fps, device=device, dtype=torch.float32).view(
            batch_size
        )
        domain_id = torch.as_tensor(domain_id, device=device, dtype=torch.long).view(batch_size)
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32).view(batch_size, 1)

        vision_tensor, action_image_size, _height, _width = preprocess_action_video_batch(
            videos,
            resolution_tier=self.config.resolution_tier,
            num_frames=self.config.chunk_size + 1,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            clean_vision = self._encode_video(vision_tensor).contiguous().float()
            clean_vision = self._remove_action_video_padding_from_latent(clean_vision, action_image_size)

        vision_condition_mask = torch.zeros(
            (1, 1, clean_vision.shape[2], 1, 1),
            device=device,
            dtype=torch.float32,
        )
        vision_condition_mask[:, :, 0] = 1.0
        vision_noisy_mask = 1.0 - vision_condition_mask
        vision_sigma = sigma.view(batch_size, 1, 1, 1, 1) * vision_noisy_mask
        epsilon_vision = torch.randn(clean_vision.shape, device=device, dtype=torch.float32)
        noised_vision = epsilon_vision * vision_sigma + clean_vision * (1.0 - vision_sigma)
        target_vision = epsilon_vision - clean_vision

        action_dim = int(self.transformer.config.action_dim)
        clean_action = clean_action.to(device=device, dtype=torch.float32)
        if clean_action.shape[-1] < action_dim:
            clean_action = F.pad(clean_action, (0, action_dim - clean_action.shape[-1]))
        clean_action = clean_action[:, :, :action_dim]
        action_condition_mask = action_condition_mask.to(device=device, dtype=torch.float32)
        sigma_action = sigma.view(batch_size, 1, 1) * (1.0 - action_condition_mask)
        epsilon_action = torch.randn(clean_action.shape, device=device, dtype=torch.float32)
        noised_action = epsilon_action * sigma_action + clean_action * (1.0 - sigma_action)
        target_action = epsilon_action - clean_action
        action_dim_indexes = torch.arange(action_dim, device=device).view(1, 1, action_dim)
        raw_action_mask = action_dim_indexes < raw_action_dims.view(batch_size, 1, 1)
        noised_action = noised_action.masked_fill(~raw_action_mask, 0)

        max_timestep = float(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        timesteps = sigma.flatten() * max_timestep
        packed_samples = self._pack_batch_static(
            cond_input_ids=cond_input_ids,
            vision_tokens=noised_vision.to(dtype=dtype),
            action_tokens=noised_action.to(dtype=dtype),
            conditioning_fps=conditioning_fps,
        )
        vision_timesteps = [
            torch.full(
                (sample["num_noisy_vision_tokens"],),
                float(timesteps[batch_idx].item()),
                device=device,
                dtype=torch.float32,
            )
            for batch_idx, sample in enumerate(packed_samples)
        ]
        action_timesteps = [
            torch.full(
                (sample["num_noisy_action_tokens"],),
                float(timesteps[batch_idx].item()),
                device=device,
                dtype=torch.float32,
            )
            for batch_idx, sample in enumerate(packed_samples)
        ]

        pred_vision, pred_action = self._predict_velocity_batch(
            packed_samples=packed_samples,
            vision_tokens=noised_vision.to(dtype=dtype),
            action_tokens=noised_action.to(dtype=dtype),
            vision_timesteps=vision_timesteps,
            action_timesteps=action_timesteps,
            action_domain_ids=domain_id,
            vision_condition_mask=vision_condition_mask.to(dtype=dtype),
            action_condition_mask=action_condition_mask.to(dtype=dtype),
            raw_action_dims=raw_action_dims,
        )

        vision_loss = self._masked_flow_matching_mse_by_sample(
            pred_vision.to(dtype=torch.float32),
            target_vision.to(device=pred_vision.device, dtype=torch.float32),
            vision_noisy_mask.to(device=pred_vision.device, dtype=torch.float32),
        )

        action_noisy_mask = (1.0 - action_condition_mask) * raw_action_mask
        action_loss = self._masked_flow_matching_mse_by_sample(
            pred_action.to(dtype=torch.float32),
            target_action.to(device=pred_action.device, dtype=torch.float32),
            action_noisy_mask.to(device=pred_action.device, dtype=torch.float32),
        )
        total_loss = (
            self.config.video_loss_weight * vision_loss + self.config.action_loss_weight * action_loss
        )
        return {
            "loss": total_loss,
            "flow_matching_loss_vision": vision_loss,
            "flow_matching_loss_action": action_loss,
        }

    def _masked_flow_matching_mse_by_sample(self, pred: Tensor, target: Tensor, noisy_mask: Tensor) -> Tensor:
        noisy_mask = noisy_mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
        sqerr = (pred - target) ** 2 * noisy_mask
        if not self.config.normalize_loss_by_active:
            return sqerr.flatten(1).mean(dim=1).mean()

        active_count = noisy_mask.flatten(1).sum(dim=1)
        return (sqerr.flatten(1).sum(dim=1) / active_count.clamp_min(1.0)).mean()

    def _encode_video(self, video: Tensor) -> Tensor:
        vae_dtype = _module_dtype(self.vae)
        encoded = _retrieve_latents(self.vae.encode(video.to(vae_dtype)), sample_mode="argmax")
        mean = torch.tensor(self.vae.config.latents_mean, device=encoded.device, dtype=encoded.dtype)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=encoded.device, dtype=encoded.dtype)
        return ((encoded - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(video.dtype)

    def _remove_action_video_padding_from_latent(self, latents: Tensor, image_size: Tensor) -> Tensor:
        spatial_factor = int(getattr(self.vae.config, "scale_factor_spatial", 16))
        content_h = int(image_size[2].item())
        content_w = int(image_size[3].item())
        content_h_latent = max(content_h // spatial_factor, 1)
        content_w_latent = max(content_w // spatial_factor, 1)
        return latents[:, :, :, :content_h_latent, :content_w_latent].contiguous()

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        *,
        seed: int | list[int] | tuple[int, ...] | Tensor | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        required = [
            COSMOS3_VIDEO,
            COSMOS3_ACTION_CONDITION,
            COSMOS3_ACTION_CONDITION_MASK,
            COSMOS3_ACTION_DOMAIN_ID,
            COSMOS3_CONDITIONING_FPS,
            COSMOS3_RAW_ACTION_DIM,
            COSMOS3_COND_INPUT_IDS,
            COSMOS3_UNCOND_INPUT_IDS,
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Cosmos3 batch is missing required model inputs: {missing}")

        videos = batch[COSMOS3_VIDEO]
        if videos.ndim == 4:
            videos = videos.unsqueeze(0)
        batch_size = videos.shape[0]
        action_condition = batch[COSMOS3_ACTION_CONDITION]
        if action_condition.ndim == 2:
            action_condition = action_condition.unsqueeze(0)
        action_condition_mask = batch[COSMOS3_ACTION_CONDITION_MASK]
        if action_condition_mask.ndim == 2:
            action_condition_mask = action_condition_mask.unsqueeze(0)
        seeds = self._normalise_sample_seeds(seed, batch_size)
        return self._sample_batch(
            cond_input_ids=[
                self._get_ids_for_batch(batch[COSMOS3_COND_INPUT_IDS], batch_idx)
                for batch_idx in range(batch_size)
            ],
            uncond_input_ids=[
                self._get_ids_for_batch(batch[COSMOS3_UNCOND_INPUT_IDS], batch_idx)
                for batch_idx in range(batch_size)
            ],
            videos=videos,
            action_condition=action_condition,
            action_condition_mask=action_condition_mask,
            domain_id=batch[COSMOS3_ACTION_DOMAIN_ID],
            conditioning_fps=batch[COSMOS3_CONDITIONING_FPS],
            raw_action_dim=batch[COSMOS3_RAW_ACTION_DIM],
            seeds=seeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def predict_future_video(self, batch: dict[str, Tensor], **kwargs) -> Tensor | None:
        if not self.config.generate_video:
            return None
        raise NotImplementedError(
            "Cosmos3 future-video decoding is reserved for a follow-up integration step."
        )

    def _normalise_sample_seeds(self, seed: Any, batch_size: int) -> list[int]:
        if seed is None:
            return [self._next_seed() for _ in range(batch_size)]
        if isinstance(seed, Tensor):
            seed = seed.detach().cpu().flatten().tolist()
        if isinstance(seed, list | tuple):
            if len(seed) != batch_size:
                raise ValueError(f"Expected {batch_size} Cosmos3 seeds, got {len(seed)}.")
            return [int(item) for item in seed]
        return [int(seed)] * batch_size

    def _sample_batch(
        self,
        *,
        cond_input_ids: list[Tensor],
        uncond_input_ids: list[Tensor],
        videos: Tensor,
        action_condition: Tensor,
        action_condition_mask: Tensor,
        domain_id: Tensor,
        conditioning_fps: Tensor,
        raw_action_dim: Tensor,
        seeds: list[int],
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        device = _module_device(self.transformer)
        dtype = _module_dtype(self.transformer)
        batch_size = videos.shape[0]
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        raw_action_dims = torch.as_tensor(raw_action_dim, device=device, dtype=torch.long).view(batch_size)
        conditioning_fps = torch.as_tensor(conditioning_fps, device=device, dtype=torch.float32).view(
            batch_size
        )
        domain_id = torch.as_tensor(domain_id, device=device, dtype=torch.long).view(batch_size)

        vision_tensor, action_image_size, _height, _width = preprocess_action_video_batch(
            videos,
            resolution_tier=self.config.resolution_tier,
            num_frames=self.config.chunk_size + 1,
            device=device,
            dtype=dtype,
        )
        x0_tokens_vision = self._encode_video(vision_tensor).contiguous().float()
        x0_tokens_vision = self._remove_action_video_padding_from_latent(x0_tokens_vision, action_image_size)

        vision_condition_mask = torch.zeros(
            (1, 1, x0_tokens_vision.shape[2], 1, 1),
            device=device,
            dtype=dtype,
        )
        vision_condition_mask[:, :, 0] = 1.0
        pure_noise = torch.cat(
            [
                _seeded_standard_normal(
                    tuple(x0_tokens_vision[batch_idx : batch_idx + 1].shape),
                    dtype=dtype,
                    device=device,
                    seed=seeds[batch_idx],
                )
                for batch_idx in range(batch_size)
            ],
            dim=0,
        )
        latents = (
            vision_condition_mask * x0_tokens_vision.to(dtype=dtype)
            + (1.0 - vision_condition_mask) * pure_noise
        )

        action_dim = int(self.transformer.config.action_dim)
        action_condition = action_condition.to(device=device, dtype=dtype)
        if action_condition.shape[-1] < action_dim:
            action_condition = F.pad(action_condition, (0, action_dim - action_condition.shape[-1]))
        action_condition = action_condition[:, :, :action_dim]
        action_condition_mask = action_condition_mask.to(device=device, dtype=dtype)
        pure_action_noise = torch.stack(
            [
                _seeded_standard_normal(
                    tuple(action_condition[batch_idx].shape),
                    dtype=dtype,
                    device=device,
                    seed=seeds[batch_idx],
                )
                for batch_idx in range(batch_size)
            ],
            dim=0,
        )
        action_latents = (
            action_condition_mask * action_condition + (1.0 - action_condition_mask) * pure_action_noise
        )
        action_dim_indexes = torch.arange(action_dim, device=device).view(1, 1, action_dim)
        raw_action_mask = action_dim_indexes < raw_action_dims.view(batch_size, 1, 1)
        action_latents = action_latents.masked_fill(~raw_action_mask, 0)

        cond_packed_static = self._pack_batch_static(
            cond_input_ids=cond_input_ids,
            vision_tokens=latents,
            action_tokens=action_latents,
            conditioning_fps=conditioning_fps,
        )
        uncond_packed_static = self._pack_batch_static(
            cond_input_ids=uncond_input_ids,
            vision_tokens=latents,
            action_tokens=action_latents,
            conditioning_fps=conditioning_fps,
        )

        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        vision_shape = tuple(latents.shape[1:])
        action_shape = tuple(action_latents.shape[1:])
        vision_size = latents[0].numel()

        def pack_latents(vision: Tensor, action: Tensor) -> Tensor:
            return torch.cat([vision.reshape(batch_size, -1), action.reshape(batch_size, -1)], dim=1)

        def unpack_latents(flat_latents: Tensor) -> tuple[Tensor, Tensor]:
            vision = flat_latents[:, :vision_size].reshape(batch_size, *vision_shape)
            action = flat_latents[:, vision_size:].reshape(batch_size, *action_shape)
            return vision, action

        flat_latents = pack_latents(latents, action_latents)
        for timestep_tensor in timesteps:
            timestep = float(timestep_tensor.item())
            latents, action_latents = unpack_latents(flat_latents)
            vision_tokens = latents.to(device=device, dtype=dtype)
            action_tokens = action_latents.to(device=device, dtype=dtype)
            vision_timesteps = [
                torch.full((sample["num_noisy_vision_tokens"],), timestep, device=device)
                for sample in cond_packed_static
            ]
            action_timesteps = [
                torch.full((sample["num_noisy_action_tokens"],), timestep, device=device)
                for sample in cond_packed_static
            ]

            cond_v_vision, cond_v_action = self._predict_velocity_batch(
                packed_samples=cond_packed_static,
                vision_tokens=vision_tokens,
                action_tokens=action_tokens,
                vision_timesteps=vision_timesteps,
                action_timesteps=action_timesteps,
                action_domain_ids=domain_id,
                vision_condition_mask=vision_condition_mask,
                action_condition_mask=action_condition_mask,
                raw_action_dims=raw_action_dims,
            )
            if guidance_scale != 1.0:
                uncond_v_vision, uncond_v_action = self._predict_velocity_batch(
                    packed_samples=uncond_packed_static,
                    vision_tokens=vision_tokens,
                    action_tokens=action_tokens,
                    vision_timesteps=vision_timesteps,
                    action_timesteps=action_timesteps,
                    action_domain_ids=domain_id,
                    vision_condition_mask=vision_condition_mask,
                    action_condition_mask=action_condition_mask,
                    raw_action_dims=raw_action_dims,
                )
                velocity_vision = uncond_v_vision + guidance_scale * (cond_v_vision - uncond_v_vision)
                velocity_action = uncond_v_action + guidance_scale * (cond_v_action - uncond_v_action)
            else:
                velocity_vision = cond_v_vision
                velocity_action = cond_v_action

            velocity = pack_latents(velocity_vision, velocity_action)
            flat_latents = scheduler.step(velocity, timestep_tensor, flat_latents, return_dict=False)[0]
            latents, action_latents = unpack_latents(flat_latents)
            action_latents = action_latents.masked_fill(~raw_action_mask, 0)
            flat_latents = pack_latents(latents, action_latents)

        output_action_dim = int(raw_action_dims.max().item())
        actions = action_latents[:, :, :output_action_dim].detach().cpu().to(torch.float32)
        if self.config.history_length:
            actions = actions[:, self.config.history_length :]
        if self.config.invert_gripper:
            for batch_idx, raw_dim in enumerate(raw_action_dims.detach().cpu().tolist()):
                actions[batch_idx, :, raw_dim - 1] = 1.0 - actions[batch_idx, :, raw_dim - 1]
        return actions[:, : self.config.chunk_size]

    def _prepare_text_segment(self, input_ids: Tensor, device: torch.device | str) -> dict[str, Any]:
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        input_ids = torch.cat(
            [
                input_ids,
                input_ids.new_tensor([self.config.eos_token_id, self.config.start_of_generation_token_id]),
            ],
            dim=0,
        )
        config = self.transformer.config
        und_len = int(input_ids.numel())
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=bool(config.enable_fps_modulation),
        )
        return {
            "input_ids": input_ids,
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset
            + config.unified_3d_mrope_temporal_modality_margin,
        }

    def _pack_static_segments(
        self,
        *,
        text_segment: dict[str, Any],
        latents: Tensor,
        action_latents: Tensor,
        vision_condition_indexes: list[int],
        fps_vision: float,
        action_start_frame_offset: int,
    ) -> dict[str, Any]:
        device = latents.device
        vision_segment = self._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=True,
            mrope_offset=text_segment["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=text_segment["und_len"],
            device=device,
            condition_frame_indexes=vision_condition_indexes,
        )
        action_segment = self._prepare_action_segment(
            input_action_tokens=action_latents,
            condition_frame_indexes=[0] if self.config.use_state else [],
            mrope_offset=text_segment["vision_start_temporal_offset"],
            action_fps=fps_vision,
            curr=text_segment["und_len"] + vision_segment["num_vision_tokens"],
            device=device,
            start_frame_offset=action_start_frame_offset,
        )
        position_ids = torch.cat(
            [
                text_segment["text_mrope_ids"],
                vision_segment["vision_mrope_ids"],
                action_segment["action_mrope_ids"],
            ],
            dim=1,
        )
        return {
            **text_segment,
            **vision_segment,
            **action_segment,
            "position_ids": position_ids,
            "sequence_length": text_segment["und_len"]
            + vision_segment["num_vision_tokens"]
            + action_segment["action_len"],
        }

    def _pack_batch_static(
        self,
        *,
        cond_input_ids: list[Tensor],
        vision_tokens: Tensor,
        action_tokens: Tensor,
        conditioning_fps: Tensor,
    ) -> list[dict[str, Any]]:
        device = vision_tokens.device
        packed_samples = []
        for batch_idx, input_ids in enumerate(cond_input_ids):
            text_segment = self._prepare_text_segment(input_ids, device=device)
            packed_samples.append(
                self._pack_static_segments(
                    text_segment=text_segment,
                    latents=vision_tokens[batch_idx : batch_idx + 1],
                    action_latents=action_tokens[batch_idx],
                    vision_condition_indexes=[0],
                    fps_vision=float(conditioning_fps[batch_idx].item()),
                    action_start_frame_offset=0 if self.config.use_state else 1,
                )
            )
        return packed_samples

    def _prepare_vision_segment(
        self,
        *,
        input_vision_tokens: Tensor,
        has_image_condition: bool,
        mrope_offset: int | float,
        vision_fps: float | None,
        curr: int,
        device: torch.device | str,
        condition_frame_indexes: list[int] | None = None,
    ) -> dict[str, Any]:
        config = self.transformer.config
        latent_patch_size = int(config.latent_patch_size)
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w

        if condition_frame_indexes is None:
            condition_frame_indexes = [0] if has_image_condition else []
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < latent_t}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(latent_t) if idx not in cond_frames], device=device, dtype=torch.long
        )

        frame_token_stride = patch_h * patch_w
        mse_loss_indexes: list[int] = []
        for frame_idx in noisy_frame_indexes.tolist():
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

        effective_fps = vision_fps if config.enable_fps_modulation else None
        temporal_compression_factor = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=temporal_compression_factor,
        )

        return {
            "vision_token_shapes": [(latent_t, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(
                curr, curr + num_vision_tokens, dtype=torch.long, device=device
            ),
            "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            "vision_mrope_ids": vision_mrope_ids.to(device),
            "num_vision_tokens": num_vision_tokens,
            "num_noisy_vision_tokens": len(noisy_frame_indexes) * frame_token_stride,
        }

    def _prepare_action_segment(
        self,
        *,
        input_action_tokens: Tensor,
        condition_frame_indexes: list[int],
        mrope_offset: int | float,
        action_fps: float | None,
        curr: int,
        device: torch.device | str,
        start_frame_offset: int,
    ) -> dict[str, Any]:
        config = self.transformer.config
        action_len = input_action_tokens.shape[0]
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < action_len}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(action_len) if idx not in cond_frames], device=device, dtype=torch.long
        )

        effective_fps = action_fps if config.enable_fps_modulation else None
        base_tcf = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
            base_temporal_compression_factor=base_tcf,
            start_frame_offset=start_frame_offset,
        )
        sequence_indexes = torch.arange(curr, curr + action_len, dtype=torch.long, device=device)
        return {
            "action_token_shapes": [(action_len, 1, 1)],
            "action_sequence_indexes": sequence_indexes,
            "action_mse_loss_indexes": sequence_indexes[noisy_frame_indexes],
            "action_noisy_frame_indexes": [noisy_frame_indexes],
            "action_mrope_ids": action_mrope_ids.to(device),
            "action_len": action_len,
            "num_noisy_action_tokens": len(noisy_frame_indexes),
        }

    def _predict_velocity_batch(
        self,
        *,
        packed_samples: list[dict[str, Any]],
        vision_tokens: Tensor,
        action_tokens: Tensor,
        vision_timesteps: list[Tensor],
        action_timesteps: list[Tensor],
        action_domain_ids: Tensor,
        vision_condition_mask: Tensor,
        action_condition_mask: Tensor,
        raw_action_dims: Tensor,
    ) -> tuple[Tensor, Tensor]:
        transformer = self.transformer
        batch_size = len(packed_samples)
        device = vision_tokens.device
        hidden_size = int(transformer.config.hidden_size)
        max_und_len = max(sample["und_len"] for sample in packed_samples)
        gen_lengths = [sample["sequence_length"] - sample["und_len"] for sample in packed_samples]
        max_gen_len = max(gen_lengths)

        all_input_ids = torch.cat([sample["input_ids"] for sample in packed_samples], dim=0)
        all_text_embeddings = transformer.embed_tokens(all_input_ids)
        target_dtype = all_text_embeddings.dtype
        und_seq = all_text_embeddings.new_zeros((batch_size, max_und_len, hidden_size))
        gen_seq = all_text_embeddings.new_zeros((batch_size, max_gen_len, hidden_size))
        und_valid_mask = torch.zeros((batch_size, max_und_len), device=device, dtype=torch.bool)
        gen_valid_mask = torch.zeros((batch_size, max_gen_len), device=device, dtype=torch.bool)

        position_dtype = packed_samples[0]["position_ids"].dtype
        position_ids_und = torch.zeros((3, batch_size, max_und_len), device=device, dtype=position_dtype)
        position_ids_gen = torch.zeros((3, batch_size, max_gen_len), device=device, dtype=position_dtype)

        text_offset = 0
        for batch_idx, sample in enumerate(packed_samples):
            und_len = sample["und_len"]
            gen_len = gen_lengths[batch_idx]
            und_seq[batch_idx, :und_len] = all_text_embeddings[text_offset : text_offset + und_len]
            text_offset += und_len
            und_valid_mask[batch_idx, :und_len] = True
            gen_valid_mask[batch_idx, :gen_len] = True
            position_ids = sample["position_ids"]
            position_ids_und[:, batch_idx, :und_len] = position_ids[:, :und_len]
            position_ids_gen[:, batch_idx, :gen_len] = position_ids[:, und_len : und_len + gen_len]

        vision_token_list = [vision_tokens[batch_idx : batch_idx + 1] for batch_idx in range(batch_size)]
        vision_token_shapes = [sample["vision_token_shapes"][0] for sample in packed_samples]
        vision_noisy_frame_indexes = [sample["vision_noisy_frame_indexes"][0] for sample in packed_samples]
        packed_vision_tokens, original_latent_shapes = transformer._patchify_and_pack_latents(
            vision_token_list
        )
        packed_vision_tokens = transformer.proj_in(packed_vision_tokens)
        packed_vision_timestep_embeds = transformer.time_embedder(
            transformer.time_proj(torch.cat(vision_timesteps, dim=0) * transformer.config.timestep_scale)
        ).to(target_dtype)
        packed_vision_tokens = transformer._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_vision_tokens,
            packed_timestep_embeds=packed_vision_timestep_embeds,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )

        vision_token_offset = 0
        for batch_idx, sample in enumerate(packed_samples):
            token_count = sample["num_vision_tokens"]
            gen_indexes = sample["vision_sequence_indexes"] - sample["und_len"]
            gen_seq[batch_idx, gen_indexes] = packed_vision_tokens[
                vision_token_offset : vision_token_offset + token_count
            ]
            vision_token_offset += token_count

        action_token_list = [action_tokens[batch_idx] for batch_idx in range(batch_size)]
        action_token_shapes = [sample["action_token_shapes"][0] for sample in packed_samples]
        action_noisy_frame_indexes = [sample["action_noisy_frame_indexes"][0] for sample in packed_samples]
        action_domain_id_list = [action_domain_ids[batch_idx].view(1) for batch_idx in range(batch_size)]
        packed_action_tokens, per_token_domain_ids = transformer._pack_action_latents(
            action_token_list, action_token_shapes, action_domain_id_list
        )
        packed_action_tokens = packed_action_tokens.to(target_dtype)
        per_token_domain_ids = per_token_domain_ids.to(device=device)
        packed_action_tokens = transformer.action_proj_in(packed_action_tokens, per_token_domain_ids)
        packed_action_tokens = packed_action_tokens + transformer.action_modality_embed
        packed_action_timestep_embeds = transformer.time_embedder(
            transformer.time_proj(torch.cat(action_timesteps, dim=0) * transformer.config.timestep_scale)
        ).to(target_dtype)
        packed_action_tokens = transformer._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_action_tokens,
            packed_timestep_embeds=packed_action_timestep_embeds,
            noisy_frame_indexes=action_noisy_frame_indexes,
            token_shapes=action_token_shapes,
        )

        action_token_offset = 0
        for batch_idx, sample in enumerate(packed_samples):
            token_count = sample["action_len"]
            gen_indexes = sample["action_sequence_indexes"] - sample["und_len"]
            gen_seq[batch_idx, gen_indexes] = packed_action_tokens[
                action_token_offset : action_token_offset + token_count
            ]
            action_token_offset += token_count

        cos_und, sin_und = transformer.rotary_emb(
            position_ids=position_ids_und,
            device=device,
            dtype=target_dtype,
        )
        cos_gen, sin_gen = transformer.rotary_emb(
            position_ids=position_ids_gen,
            device=device,
            dtype=target_dtype,
        )
        rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)
        und_seq = und_seq * und_valid_mask.unsqueeze(-1)
        gen_seq = gen_seq * gen_valid_mask.unsqueeze(-1)

        for decoder_layer in transformer.layers:
            und_seq, gen_seq = self._batched_decoder_layer(
                decoder_layer,
                und_seq,
                gen_seq,
                rotary_emb,
                und_valid_mask,
                gen_valid_mask,
            )

        gen_out = transformer.norm_moe_gen(gen_seq) * gen_valid_mask.unsqueeze(-1)

        vision_hidden_states = []
        for batch_idx, sample in enumerate(packed_samples):
            gen_indexes = sample["vision_mse_loss_indexes"] - sample["und_len"]
            vision_hidden_states.append(gen_out[batch_idx, gen_indexes])
        preds_vision_packed = transformer.proj_out(torch.cat(vision_hidden_states, dim=0))
        preds_vision = transformer._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )
        pred_vision = torch.cat(preds_vision, dim=0)

        action_hidden_states = []
        action_noisy_domain_ids = []
        for batch_idx, sample in enumerate(packed_samples):
            gen_indexes = sample["action_mse_loss_indexes"] - sample["und_len"]
            action_hidden_states.append(gen_out[batch_idx, gen_indexes])
            action_noisy_domain_ids.append(action_domain_ids[batch_idx].view(1).expand(len(gen_indexes)))
        preds_action_packed = transformer.action_proj_out(
            torch.cat(action_hidden_states, dim=0),
            torch.cat(action_noisy_domain_ids, dim=0).to(device=device),
        )
        preds_action = transformer._unpack_action_latents(
            preds_action_packed,
            action_token_shapes,
            action_noisy_frame_indexes,
        )
        pred_action = torch.stack(preds_action, dim=0)

        pred_vision = pred_vision * (1.0 - vision_condition_mask).to(device=pred_vision.device)
        pred_action = pred_action * (1.0 - action_condition_mask).to(device=pred_action.device)
        action_dim_indexes = torch.arange(pred_action.shape[-1], device=pred_action.device).view(1, 1, -1)
        raw_action_mask = action_dim_indexes < raw_action_dims.to(device=pred_action.device).view(
            batch_size, 1, 1
        )
        pred_action = pred_action.masked_fill(~raw_action_mask, 0)
        return pred_vision, pred_action

    def _batched_decoder_layer(
        self,
        decoder_layer: nn.Module,
        und_seq: Tensor,
        gen_seq: Tensor,
        rotary_emb: tuple[Tensor, Tensor, Tensor, Tensor],
        und_valid_mask: Tensor,
        gen_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        und_norm = decoder_layer.input_layernorm(und_seq)
        gen_norm = decoder_layer.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self._batched_mot_attention(
            decoder_layer.self_attn,
            und_norm,
            gen_norm,
            rotary_emb,
            und_valid_mask,
            gen_valid_mask,
        )
        residual_und = (und_seq + und_attn_out) * und_valid_mask.unsqueeze(-1)
        residual_gen = (gen_seq + gen_attn_out) * gen_valid_mask.unsqueeze(-1)

        mlp_out_und = decoder_layer.mlp(decoder_layer.post_attention_layernorm(residual_und))
        mlp_out_gen = decoder_layer.mlp_moe_gen(decoder_layer.post_attention_layernorm_moe_gen(residual_gen))
        und_out = (residual_und + mlp_out_und) * und_valid_mask.unsqueeze(-1)
        gen_out = (residual_gen + mlp_out_gen) * gen_valid_mask.unsqueeze(-1)
        return und_out, gen_out

    def _batched_mot_attention(
        self,
        attn: nn.Module,
        und_seq: Tensor,
        gen_seq: Tensor,
        rotary_emb: tuple[Tensor, Tensor, Tensor, Tensor],
        und_valid_mask: Tensor,
        gen_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, und_len, _hidden_size = und_seq.shape
        gen_len = gen_seq.shape[1]

        q_und = attn.to_q(und_seq).view(batch_size, und_len, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(batch_size, und_len, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(batch_size, und_len, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(batch_size, gen_len, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(batch_size, gen_len, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(batch_size, gen_len, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        q_und = q_und * cos_und.unsqueeze(2) + self._rotate_half(q_und) * sin_und.unsqueeze(2)
        k_und = k_und * cos_und.unsqueeze(2) + self._rotate_half(k_und) * sin_und.unsqueeze(2)
        q_gen = q_gen * cos_gen.unsqueeze(2) + self._rotate_half(q_gen) * sin_gen.unsqueeze(2)
        k_gen = k_gen * cos_gen.unsqueeze(2) + self._rotate_half(k_gen) * sin_gen.unsqueeze(2)

        q_und = q_und.transpose(1, 2)
        k_und = self._repeat_kv_heads(k_und, attn.num_key_value_groups).transpose(1, 2)
        v_und = self._repeat_kv_heads(v_und, attn.num_key_value_groups).transpose(1, 2)
        q_gen = q_gen.transpose(1, 2)
        k_gen = self._repeat_kv_heads(k_gen, attn.num_key_value_groups).transpose(1, 2)
        v_gen = self._repeat_kv_heads(v_gen, attn.num_key_value_groups).transpose(1, 2)

        causal_mask = torch.ones((und_len, und_len), device=und_seq.device, dtype=torch.bool).tril()
        und_attn_mask = causal_mask.view(1, 1, und_len, und_len) & und_valid_mask.view(
            batch_size, 1, 1, und_len
        )
        causal_out = F.scaled_dot_product_attention(
            q_und,
            k_und,
            v_und,
            attn_mask=und_attn_mask,
            dropout_p=0.0,
        )
        causal_out = causal_out.transpose(1, 2).flatten(-2, -1)

        all_k = torch.cat([k_und, k_gen], dim=2)
        all_v = torch.cat([v_und, v_gen], dim=2)
        all_valid_mask = torch.cat([und_valid_mask, gen_valid_mask], dim=1)
        gen_attn_mask = all_valid_mask.view(batch_size, 1, 1, und_len + gen_len).expand(
            batch_size, 1, gen_len, und_len + gen_len
        )
        full_out = F.scaled_dot_product_attention(
            q_gen,
            all_k,
            all_v,
            attn_mask=gen_attn_mask,
            dropout_p=0.0,
        )
        full_out = full_out.transpose(1, 2).flatten(-2, -1)

        und_out = attn.to_out(causal_out) * und_valid_mask.unsqueeze(-1)
        gen_out = attn.to_add_out(full_out) * gen_valid_mask.unsqueeze(-1)
        return und_out, gen_out

    def _repeat_kv_heads(self, hidden_states: Tensor, num_key_value_groups: int) -> Tensor:
        if num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(num_key_value_groups, dim=2)

    def _rotate_half(self, hidden_states: Tensor) -> Tensor:
        half = hidden_states.shape[-1] // 2
        return torch.cat((-hidden_states[..., half:], hidden_states[..., :half]), dim=-1)
