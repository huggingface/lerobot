# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 The FastWAM Authors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Long-WAM flow matching, adapted from NVlabs/LongLive (1b9e348).

Originally derived from FastWAM. Upstream license notice for the adapted code:

MIT License

Copyright (c) 2026 The FastWAM Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import deque
from typing import TYPE_CHECKING, Any, Self

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _diffusers_available, _transformers_available, require_package

from .configuration_longwam import LongWAMConfig
from .mot_longwam import LongWAMDiT

if TYPE_CHECKING or _diffusers_available:
    from diffusers import AutoencoderKLWan
if TYPE_CHECKING or _transformers_available:
    from transformers import UMT5Config, UMT5EncoderModel


class LongWAMPolicy(PreTrainedPolicy):
    config_class = LongWAMConfig
    name = "longwam"
    config: LongWAMConfig

    def __init__(self, config: LongWAMConfig, **kwargs: Any) -> None:
        require_package("diffusers", extra="longwam")
        require_package("transformers", extra="longwam")
        super().__init__(config)
        if not all((config.video_config, config.vae_config, config.text_encoder_config)):
            raise ValueError("Load a native Long-WAM checkpoint containing the three backbone configs")
        if config.video_config.get("patch_size", (1, 2, 2))[0] != 1:
            raise ValueError("Long-WAM requires temporal patch size 1")
        if config.vae_config.get("scale_factor_temporal", 4) != 4:
            raise ValueError("Long-WAM requires VAE temporal compression 4")
        self.dit = LongWAMDiT(config)
        self.vae = AutoencoderKLWan.from_config(config.vae_config).requires_grad_(False).eval()
        self.text_encoder = UMT5EncoderModel(UMT5Config(**config.text_encoder_config))
        self.text_encoder.requires_grad_(False).eval()
        self.proprio_encoder = nn.Linear(config.proprio_dim, config.video_config["text_dim"])
        self.to(dtype=config.dtype)
        self.reset()

    def tie_weights(self) -> None:
        """Restore backbone sharing after meta placement or assigned state loading."""
        self.text_encoder.tie_weights()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.vae.eval()
        self.text_encoder.eval()
        if mode:
            self.reset()
        return self

    def get_optim_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self) -> None:
        self._frames: deque[Tensor] = deque(maxlen=self.config.past_obs_steps + 1)
        self._action_queue: deque[Tensor] = deque()
        self._language_tokens: Tensor | None = None
        self._language_mask: Tensor | None = None
        self._text_context: Tensor | None = None

    def drop_queued_actions(self) -> None:
        # Interactive rollout can change instructions without ending the environment episode.
        self.reset()

    def _images(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.cat([batch[key] for key in self.config.camera_keys], dim=-1)

    def _observe(self, batch: dict[str, Tensor]) -> None:
        tokens, mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        if self._language_tokens is not None and (
            not torch.equal(tokens, self._language_tokens) or not torch.equal(mask, self._language_mask)
        ):
            self.reset()
        if self._language_tokens is None:
            self._language_tokens, self._language_mask = tokens.clone(), mask.clone()
        image = self._images(batch)
        if image.ndim != 4:
            raise ValueError("Synchronous inference expects the current frame [B,3,H,W]")
        if self._frames and self._frames[-1].shape != image.shape:
            raise ValueError("Reset before changing inference batch size or camera shape")
        self._frames.append(image.detach().clone())

    def _history(self) -> Tensor:
        frames = list(self._frames)
        frames = [frames[0]] * (self.config.past_obs_steps + 1 - len(frames)) + frames
        return torch.stack(frames[:: self.config.video_stride], dim=2)

    def _context(self, batch: dict[str, Tensor], state: Tensor, *, cache: bool = False) -> Tensor:
        if not cache or self._text_context is None:
            with torch.no_grad():
                text = self.text_encoder(
                    input_ids=batch[OBS_LANGUAGE_TOKENS],
                    attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
                ).last_hidden_state
                # Released Wan conditioning keeps zero-padded tokens visible to cross-attention.
                text = text * batch[OBS_LANGUAGE_ATTENTION_MASK][..., None]
            if cache:
                self._text_context = text
        else:
            text = self._text_context
        proprio = self.proprio_encoder(state.to(text.dtype)).unsqueeze(1)
        return torch.cat((text, proprio), dim=1)

    @torch.no_grad()
    def _encode_video(self, video: Tensor) -> Tensor:
        latents = self.vae.encode(video.to(next(self.vae.parameters()).dtype)).latent_dist.mode()
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inverse_std = torch.tensor(self.vae.config.latents_std, device=latents.device).reciprocal()
        return (latents - mean) * inverse_std.to(latents).view(1, -1, 1, 1, 1)

    def _schedule(self, steps: int, reference: Tensor, target: float = 0) -> Tensor:
        u = torch.linspace(1, 0, steps + 1, device=reference.device)
        sigma = self.config.flow_shift * u / (1 + (self.config.flow_shift - 1) * u)
        return target + (1 - target) * sigma

    def _sample_time(self, reference: Tensor) -> Tensor:
        u = torch.rand(reference.shape[0], device=reference.device)
        return (1000 * self.config.flow_shift * u / (1 + (self.config.flow_shift - 1) * u)).to(
            reference.dtype
        )

    @staticmethod
    def _add_noise(data: Tensor, noise: Tensor, time: Tensor) -> Tensor:
        sigma = (time / 1000).reshape(-1, *([1] * (data.ndim - 1)))
        return (1 - sigma) * data + sigma * noise

    def _flow_loss(self, prediction: Tensor, target: Tensor, pad: Tensor, time: Tensor) -> Tensor:
        error = (prediction.float() - target.float()).square()
        error = error.mean((1, 3, 4)) if error.ndim == 5 else error.mean(-1)
        valid = (~pad).to(error.dtype)
        per_sample = (error * valid).sum(1) / valid.sum(1).clamp_min(1)
        grid = self._schedule(1000, time)[:-1].double()
        baseline = torch.exp(torch.tensor(-0.5, device=time.device, dtype=torch.float64))
        normalizer = (torch.exp(-2 * (grid - 0.5).square()) - baseline).mean()
        weight = (torch.exp(-2 * (time.float() / 1000 - 0.5).square()) - baseline) / normalizer
        return per_sample * weight.float()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        cfg = self.config
        images = self._images(batch)
        if images.ndim != 5 or images.shape[1] != len(cfg.observation_delta_indices):
            raise ValueError("Training needs the strided history/current/future window")
        context = self._context(batch, batch["observation.state"][:, cfg.past_obs_steps // cfg.video_stride])
        latents = self._encode_video(images.transpose(1, 2))
        clean_frames = cfg.past_obs_steps // 16 + 1
        action = batch["action"].to(latents.dtype)
        if action.shape[1:] != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("Training action shape does not match the configured horizon")
        video_noise, action_noise = torch.randn_like(latents), torch.randn_like(action)
        video_time, action_time = self._sample_time(latents), self._sample_time(action)
        noisy_video = self._add_noise(latents, video_noise, video_time)
        noisy_video[:, :, :clean_frames] = latents[:, :, :clean_frames]
        noisy_action = self._add_noise(action, action_noise, action_time)
        if cfg.denoising == "cod":
            video_prediction, action_prediction, _ = self.dit(
                noisy_video, noisy_action, video_time, action_time, context
            )
        else:
            video_prediction, _, _ = self.dit(noisy_video, None, video_time, None, context)
            # Action loss must not backpropagate through the imagined video expert.
            with torch.no_grad():
                imagine_time = torch.full_like(video_time, cfg.imagine_sigma * 1000)
                imagined = self._add_noise(latents, torch.randn_like(latents), imagine_time)
                imagined[:, :, :clean_frames] = latents[:, :, :clean_frames]
                _, _, video_cache = self.dit(imagined, None, imagine_time, None, context, return_cache=True)
            _, action_prediction, _ = self.dit(
                None, noisy_action, None, action_time, context, video_cache=video_cache
            )
        image_pad = torch.zeros(images.shape[:2], dtype=torch.bool, device=images.device)
        for key in cfg.camera_keys:
            image_pad = image_pad | batch.get(f"{key}_is_pad", torch.zeros_like(image_pad))
        video_pad = image_pad[:, cfg.past_obs_steps // cfg.video_stride + 1 :].unflatten(1, (-1, 4)).all(-1)
        action_pad = batch.get(
            "action_is_pad", torch.zeros(action.shape[:2], dtype=torch.bool, device=action.device)
        )
        loss_video = self._flow_loss(
            video_prediction[:, :, clean_frames:],
            (video_noise - latents)[:, :, clean_frames:],
            video_pad,
            video_time,
        )
        loss_action = self._flow_loss(action_prediction, action_noise - action, action_pad, action_time)
        loss = cfg.loss_lambda_video * loss_video + cfg.loss_lambda_action * loss_action
        if reduction == "mean":
            loss = loss.mean()
        elif reduction != "none":
            raise ValueError("reduction must be mean or none")
        return loss, {
            "loss_video": loss_video.detach().mean().item(),
            "loss_action": loss_action.detach().mean().item(),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        self._observe(batch)
        return self._predict(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        # Record every tick, including those that execute a previously predicted action.
        self._observe(batch)
        if not self._action_queue:
            actions = self._predict(batch)
            self._action_queue.extend(actions[:, : self.config.n_action_steps].transpose(0, 1))
        return self._action_queue.popleft()

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        cfg = self.config
        clean = self._encode_video(self._history())
        context = self._context(batch, batch["observation.state"], cache=True)

        def noise(shape):
            generator = (
                None if cfg.inference_seed is None else torch.Generator().manual_seed(cfg.inference_seed)
            )
            return torch.randn(shape, generator=generator, device="cpu").to(clean)

        b, z, m, h, w = clean.shape
        future = noise((b, z, cfg.future_video_steps // 16, h, w))
        action = noise((b, cfg.prediction_steps, cfg.action_dim))
        schedule = self._schedule(cfg.num_inference_steps, clean)
        if cfg.denoising == "idm":
            video_schedule = self._schedule(cfg.imagine_steps, clean, cfg.imagine_sigma)
            for current, following in zip(video_schedule[:-1], video_schedule[1:], strict=True):
                time = (current * 1000).expand(b).to(clean.dtype)
                prediction, _, _ = self.dit(torch.cat((clean, future), dim=2), None, time, None, context)
                future = future + (following - current).to(future.dtype) * prediction[:, :, m:]
            time = torch.full((b,), cfg.imagine_sigma * 1000, device=clean.device, dtype=clean.dtype)
            _, _, video_cache = self.dit(
                torch.cat((clean, future), dim=2), None, time, None, context, return_cache=True
            )
        for current, following in zip(schedule[:-1], schedule[1:], strict=True):
            time = (current * 1000).expand(b).to(clean.dtype)
            if cfg.denoising == "cod":
                video_prediction, prediction, _ = self.dit(
                    torch.cat((clean, future), dim=2), action, time, time, context
                )
                future = future + (following - current).to(future.dtype) * video_prediction[:, :, m:]
            else:
                _, prediction, _ = self.dit(None, action, None, time, context, video_cache=video_cache)
            action = action + (following - current).to(action.dtype) * prediction
        return action.float()
