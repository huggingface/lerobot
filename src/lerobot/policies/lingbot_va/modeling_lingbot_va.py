# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
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

"""LingBot-VA policy: an autoregressive video-action world model on the Wan2.2 stack.

The sampling loop is a faithful re-implementation of the upstream streaming server
(``wan_va/wan_va_server.py``) and LIBERO client (``evaluation/libero/client.py``), adapted
to LeRobot's ``select_action`` interface:

  * the trainable dual-stream transformer is owned as a sub-module and round-trips in the
    single ``model.safetensors`` checkpoint;
  * the frozen Wan VAE + UMT5 text encoder + tokenizer are *lazily pulled* from
    ``config.wan_pretrained_path`` (not bundled), so the LeRobot checkpoint stays small;
  * ``predict_action_chunk`` runs one autoregressive chunk (video stream then action
    stream, each with CFG and its own flow-matching scheduler) and updates the KV cache;
  * ``select_action`` drains a per-step action queue and records the real observed
    keyframes that are fed back into the KV cache when the queue is refilled.

NOTE: matching the upstream LIBERO success rate is the Phase-5 correctness gate and must be
validated on a CUDA GPU with the converted checkpoint (tensor-diff against upstream on
identical inputs). The streaming path is written for single-environment eval
(``--eval.batch_size=1``).
"""

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.import_utils import require_package

from .configuration_lingbot_va import LingBotVAConfig
from .schedulers import FlowMatchScheduler
from .wan_transformer import WanTransformer3DModel
from .wan_utils import data_seq_to_patch, get_mesh_id
from .wan_vae import WanVAEStreamingWrapper, denormalize_latents, load_text_encoder, load_tokenizer, load_vae


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class LingBotVAPolicy(PreTrainedPolicy):
    """LeRobot wrapper for the LingBot-VA autoregressive video-action world model."""

    config_class = LingBotVAConfig
    name = "lingbot_va"

    def __init__(self, config: LingBotVAConfig, **kwargs):
        require_package("diffusers", extra="lingbot_va")
        require_package("transformers", extra="lingbot_va")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.dtype = _torch_dtype(config.dtype)

        # Trainable dual-stream transformer (the only sub-module saved in the LeRobot checkpoint).
        self.transformer = WanTransformer3DModel(
            patch_size=tuple(config.patch_size),
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            action_dim=config.action_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            rope_max_seq_len=config.rope_max_seq_len,
            attn_mode=config.attn_mode,
        )

        # Frozen modules are stored OUTSIDE the nn.Module registry (plain dict) so they are
        # neither saved into model.safetensors nor moved by ``.to()``. They are lazily loaded
        # from ``config.wan_pretrained_path`` the first time inference runs.
        self._frozen: dict = {}

        self.last_predicted_frames: Tensor | None = None
        self.reset()

    # ------------------------------------------------------------------
    # Frozen-module lazy loading (VAE + UMT5 + tokenizer)
    # ------------------------------------------------------------------
    def _ensure_frozen_modules(self):
        if self._frozen:
            return
        import os

        path = self.config.wan_pretrained_path
        device = self.config.device

        # Support both local diffusers-style dirs (with vae/ text_encoder/ tokenizer/ sub-folders)
        # and HF repo ids (loaders accept a subfolder kwarg, omitted here = repo root layout).
        if os.path.isdir(path):
            vae_path, te_path, tok_path = (
                os.path.join(path, n) for n in ("vae", "text_encoder", "tokenizer")
            )
        else:
            vae_path = te_path = tok_path = path

        vae = load_vae(vae_path, torch_dtype=self.dtype, torch_device=device)
        text_encoder = load_text_encoder(te_path, torch_dtype=self.dtype, torch_device=device)
        tokenizer = load_tokenizer(tok_path)
        self._frozen = {
            "vae": vae.eval(),
            "streaming_vae": WanVAEStreamingWrapper(vae),
            "text_encoder": text_encoder.eval(),
            "tokenizer": tokenizer,
        }

    @property
    def _vae(self):
        return self._frozen["vae"]

    @property
    def _streaming_vae(self):
        return self._frozen["streaming_vae"]

    # ------------------------------------------------------------------
    # PreTrainedPolicy API
    # ------------------------------------------------------------------
    def get_optim_params(self) -> dict:
        # Only the transformer is trainable; the VAE / text encoder stay frozen.
        return self.transformer.parameters()

    def reset(self):
        """Reset all per-episode streaming state (KV cache, queues, frame counter)."""
        cfg = self.config
        self._action_queue: deque = deque(maxlen=cfg.n_action_steps)
        self._obs_buffer: list = []  # keyframe camera tensors observed during the current chunk
        self._executed_actions: Tensor | None = (
            None  # last chunk's actions (model-normalized) for KV feedback
        )
        self._steps_since_refill = 0
        self._frame_st_id = 0
        self._first_chunk = True
        self._prompt: str | None = None
        self._prompt_embeds = None
        self._negative_prompt_embeds = None
        self.last_predicted_frames = None
        self._use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)
        # Two independent flow-matching schedulers (video latent + action streams).
        self._scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
        self._action_scheduler = FlowMatchScheduler(
            shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self._scheduler.set_timesteps(1000, training=True)
        self._action_scheduler.set_timesteps(1000, training=True)
        self._cache_initialised = False
        # Clear KV cache on the (already-built) transformer, if present.
        if hasattr(self, "transformer"):
            self.transformer.clear_cache("pos")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Training loss. Implemented in the LingBot-VA training PR (Phase 7).

        The flow-matching dual-stream loss needs the pre-extracted latent dataset
        (see ``LatentLeRobotDataset`` upstream) and ``attn_mode='flex'``; it is intentionally
        not part of this inference-focused integration.
        """
        raise NotImplementedError(
            "LingBot-VA training (flow-matching dual-stream loss) is part of the training port "
            "(Phase 7 / PR #2) and is not implemented in this inference integration. "
            "Use this policy for evaluation / inference."
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action, refilling the chunk (and feeding back observed keyframes) as needed."""
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        # Record the current observation as a keyframe at every frame boundary so that, when the
        # queue empties, ``predict_action_chunk`` can feed the real observed frames back into the
        # KV cache (mirroring the upstream ``compute_kv_cache`` call in the LIBERO client loop).
        # We skip ``steps_since_refill == 0`` (the obs that conditioned the current chunk): only
        # frames observed *after* executing each frame's actions are fed back.
        if self._steps_since_refill > 0 and self._steps_since_refill % self.config.action_per_frame == 0:
            self._obs_buffer.append(self._encode_obs(batch))

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)  # [B, chunk_size, n_used]
            # queue holds per-step actions: shape [chunk_size, B, n_used]
            self._action_queue.extend(actions.transpose(0, 1))
            self._steps_since_refill = 0

        self._steps_since_refill += 1
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Run one autoregressive chunk and return actions ``[B, chunk_size, n_used]`` (normalized)."""
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        is_first = self._first_chunk
        if is_first:
            init_latent = self._encode_obs(batch)
            self._init_latent = init_latent
            self._init_streaming_cache(init_latent)
            self._obs_buffer = []  # frame 0 (the init obs) conditions the chunk; it is not fed back
            actions, latents = self._infer(init_latent, frame_st_id=0)
            self._first_chunk = False
        else:
            # Feed the real observed keyframes + the executed actions back into the KV cache.
            self._compute_kv_cache(self._obs_buffer, self._executed_actions)
            self._obs_buffer = []
            actions, latents = self._infer(None, frame_st_id=self._frame_st_id)

        # actions: [B, action_dim, F, action_per_frame, 1] (model-normalized). Keep for KV feedback.
        self._executed_actions = actions

        if self.config.save_predicted_video:
            self.last_predicted_frames = self._decode_predicted_video(latents)

        # On the first chunk, frame 0 is the conditioning frame (already "known"): the upstream
        # LIBERO client skips it (start_idx=1), so we drop the first frame's actions here.
        used = self.config.used_action_channel_ids
        a = actions[:, used]  # [B, n_used, F, action_per_frame, 1]
        if is_first:
            a = a[:, :, 1:]  # drop frame 0 -> (F-1) frames of actions
        a = a.squeeze(-1).flatten(2)  # [B, n_used, n_steps]
        a = a.transpose(1, 2).contiguous()  # [B, n_steps, n_used]
        return a.to(torch.float32)

    # ------------------------------------------------------------------
    # Prompt / text encoding
    # ------------------------------------------------------------------
    def _maybe_init_prompt(self, batch):
        if self._prompt_embeds is not None:
            return
        task = batch.get("task")
        prompt = task[0] if isinstance(task, list | tuple) else task
        self._prompt = prompt or ""
        self._prompt_embeds, self._negative_prompt_embeds = self._encode_prompt(self._prompt)

    def _get_t5_prompt_embeds(self, prompt, max_sequence_length):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        tokenizer = self._frozen["tokenizer"]
        text_encoder = self._frozen["text_encoder"]
        device = self.config.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        te_device = next(text_encoder.parameters()).device
        prompt_embeds = text_encoder(text_input_ids.to(te_device), mask.to(te_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
        return prompt_embeds.to(device)

    def _encode_prompt(self, prompt):
        max_len = self.config.max_sequence_length
        prompt_embeds = self._get_t5_prompt_embeds(prompt, max_len)
        negative_prompt_embeds = None
        if self._use_cfg:
            negative_prompt_embeds = self._get_t5_prompt_embeds("", max_len)
        return prompt_embeds, negative_prompt_embeds

    # ------------------------------------------------------------------
    # Observation (image) encoding -> normalized video latents
    # ------------------------------------------------------------------
    def _camera_tensor(self, batch, key):
        """Return a single-frame camera tensor [B, C, 1, H, W] resized + scaled to [-1, 1]."""
        img = batch[key]
        if img.dim() == 3:  # [C, H, W]
            img = img.unsqueeze(0)
        # LeRobot images arrive as float in [0, 1], shape [B, C, H, W].
        img = img.to(self.config.device, torch.float32)
        img = F.interpolate(
            img, size=(self.config.height, self.config.width), mode="bilinear", align_corners=False
        )
        img = img * 2.0 - 1.0
        return img.unsqueeze(2).to(self.dtype)  # [B, C, F=1, H, W]

    @torch.no_grad()
    def _encode_obs(self, batch) -> Tensor:
        """VAE-encode all configured cameras of the current obs and concat latents on width."""
        videos = [self._camera_tensor(batch, k) for k in self.config.obs_cam_keys]
        videos = torch.cat(videos, dim=0)  # [num_cam, C, F, H, W]
        vae_device = next(self._vae.parameters()).device
        enc_out = self._streaming_vae.encode_chunk(videos.to(vae_device).to(self.dtype))
        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self._vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self._vae.config.latents_std).to(mu.device)
        # Note: upstream passes 1/std so the op is (x - mean) * (1/std).
        mean = latents_mean.view(1, -1, 1, 1, 1)
        inv_std = (1.0 / latents_std).view(1, -1, 1, 1, 1)
        mu_norm = ((mu.float() - mean) * inv_std).to(mu)
        # Concatenate the per-camera latents along width.
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.config.device)

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------
    @property
    def _latent_hw(self):
        h = self.config.height // 16
        w = (self.config.width // 16) * len(self.config.obs_cam_keys)
        return h, w

    def _init_streaming_cache(self, init_latent):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        p = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * latent_h * latent_w) // (p[0] * p[1] * p[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        self.transformer.create_empty_cache(
            "pos",
            cfg.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.config.device,
            dtype=self.dtype,
            batch_size=2 if self._use_cfg else 1,
        )
        self._cache_initialised = True

    def _repeat_input_for_cfg(self, input_dict):
        if self._use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat(
                [
                    self._prompt_embeds.to(self.dtype).clone(),
                    self._negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_latent_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ):
        cfg = self.config
        device = self.config.device
        p = cfg.patch_size
        out = {}
        if latent_model_input is not None:
            out["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=device)
                * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // p[0],
                    latent_model_input.shape[-2] // p[1],
                    latent_model_input.shape[-1] // p[2],
                    0,
                    1,
                    frame_st_id,
                ).to(device),
                "text_emb": self._prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                out["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                out["latent_res_lst"]["timesteps"][0:1] *= 0
        if action_model_input is not None:
            out["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=device)
                * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(device),
                "text_emb": self._prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                out["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                out["action_res_lst"]["timesteps"][0:1] *= 0
            out["action_res_lst"]["noisy_latents"][:, ~self._action_mask] *= 0
        return out

    @property
    def _action_mask(self):
        mask = torch.zeros([self.config.action_dim], dtype=torch.bool)
        mask[self.config.used_action_channel_ids] = True
        return mask

    # ------------------------------------------------------------------
    # Action conditioning (executed action history) (de)normalization
    # ------------------------------------------------------------------
    def _preprocess_action_state(self, action_norm: Tensor) -> Tensor:
        """Build the action-conditioning tensor from the already-normalized executed actions.

        ``action_norm`` is the model-space action chunk ``[B, action_dim, F, action_per_frame, 1]``.
        Upstream re-derives the conditioning from the raw executed action via quantile norm; here
        the executed actions are already in the model-normalized space, so we pass them through.
        """
        return action_norm.to(self.config.device, self.dtype)

    def _compute_kv_cache(self, obs_buffer, executed_actions):
        """Feed real observed keyframes + executed actions back into the KV cache."""
        if not obs_buffer or executed_actions is None:
            return
        self.transformer.clear_pred_cache("pos")
        # Concatenate the observed keyframe latents along the frame axis.
        latent_model_input = torch.cat(obs_buffer, dim=2)
        # On the first feedback, prepend the init latent so the latent/action frame counts align
        # (upstream prepends ``init_latent`` to the observed keyframes when frame_st_id == 0).
        if self._frame_st_id == 0 and getattr(self, "_init_latent", None) is not None:
            latent_model_input = torch.cat([self._init_latent, latent_model_input], dim=2)
        action_model_input = self._preprocess_action_state(executed_actions)
        action_model_input = action_model_input.to(latent_model_input)
        input_dict = self._prepare_latent_input(
            latent_model_input, action_model_input, frame_st_id=self._frame_st_id
        )
        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name="pos",
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name="pos",
                action_mode=True,
            )
        self._frame_st_id += latent_model_input.shape[2]

    # ------------------------------------------------------------------
    # The core dual-stream denoising loop (one chunk)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer(self, init_latent, frame_st_id=0):
        cfg = self.config
        device = self.config.device
        latent_h, latent_w = self._latent_hw
        frame_chunk_size = cfg.frame_chunk_size

        latents = torch.randn(1, 48, frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype)
        actions = torch.randn(
            1, cfg.action_dim, frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype
        )

        self._scheduler.set_timesteps(cfg.num_inference_steps)
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(self._scheduler.timesteps, (0, 1), mode="constant", value=0)
        if cfg.video_exec_step != -1:
            timesteps = timesteps[: cfg.video_exec_step]
        action_timesteps = F.pad(self._action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        # 1. Video-latent denoising loop
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype)
                if frame_st_id == 0 and init_latent is not None
                else None
            )
            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
            )
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=False,
            )
            if not last_step or cfg.video_exec_step != -1:
                video_noise_pred = data_seq_to_patch(
                    cfg.patch_size,
                    video_noise_pred,
                    frame_chunk_size,
                    latent_h,
                    latent_w,
                    batch_size=2 if self._use_cfg else 1,
                )
                if cfg.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (
                        video_noise_pred[:1] - video_noise_pred[1:]
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self._scheduler.step(video_noise_pred, t, latents, return_dict=False)
            if frame_st_id == 0 and latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

        # 2. Action denoising loop
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros([1, cfg.action_dim, 1, cfg.action_per_frame, 1], device=device, dtype=self.dtype)
                if frame_st_id == 0
                else None
            )
            input_dict = self._prepare_latent_input(
                None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=True,
            )
            if not last_step:
                from einops import rearrange

                action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                if cfg.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:]
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self._action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
            if frame_st_id == 0 and action_cond is not None:
                actions[:, :, 0:1] = action_cond

        actions[:, ~self._action_mask] *= 0
        return actions, latents

    # ------------------------------------------------------------------
    # Predicted-video decoding (opt-in)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _decode_predicted_video(self, latents) -> Tensor:
        """VAE-decode predicted latents into a uint8 frame stack ``[T, H, W, 3]`` on CPU."""
        vae = self._vae
        z_dim = vae.config.z_dim
        latents = denormalize_latents(
            latents.to(vae.dtype), vae.config.latents_mean, vae.config.latents_std, z_dim
        )
        video = vae.decode(latents, return_dict=False)[0]  # [B, C, F, H, W] in [-1, 1]
        video = (video.float().clamp(-1, 1) + 1.0) / 2.0
        video = (video[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8)  # [F, H, W, C]
        return video.cpu()
