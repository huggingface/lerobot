#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
from collections import deque
from typing import Any

import torch
from torch import Tensor

# Set default path to Wan2.2 pretrained weights on shared cluster storage so users
# don't need to export DIFFSYNTH_MODEL_BASE_PATH manually.
os.environ.setdefault(
    "DIFFSYNTH_MODEL_BASE_PATH",
    "/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights",
)

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)

# Prompt template used during FastWAM LIBERO training (robot_video_dataset.py DEFAULT_PROMPT).
# The raw task description string MUST be wrapped in this template — the T5 encoder was trained
# on these templated strings and will produce wrong embeddings without the prefix.
PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _get_fastwam_cls(variant: str):
    """Import and return the correct FastWAM model class for the given variant."""
    if variant == "fastwam":
        from lerobot.policies.fastwam.wan22.fastwam import FastWAM
        return FastWAM
    elif variant == "fastwam_joint":
        from lerobot.policies.fastwam.wan22.fastwam_joint import FastWAMJoint
        return FastWAMJoint
    elif variant == "fastwam_idm":
        from lerobot.policies.fastwam.wan22.fastwam_idm import FastWAMIDM
        return FastWAMIDM
    else:
        raise ValueError(f"Unknown model_variant: {variant!r}")


class FastWAMPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for FastWAM.

    FastWAM is loaded in two stages:
      1. ``__init__`` calls ``from_wan22_pretrained(skip_dit_load_from_pretrain=True)``
         which loads the VAE and T5 text encoder from the Wan2.2 pretrained weights,
         while leaving the DiT (video + action experts) randomly initialised.
      2. ``from_pretrained`` then loads ``model.safetensors`` from the checkpoint directory
         (strict=False) which fills in the fine-tuned DiT weights and VAE weights.

    The text encoder is intentionally NOT stored in the checkpoint safetensors — it always
    comes from the Wan2.2 pretrained path. This means the text encoder is correctly loaded
    in every ``__init__`` call regardless of whether we are starting fresh or restoring a
    checkpoint.
    """

    config_class = FastWAMConfig
    name = "fastwam"

    def __init__(self, config: FastWAMConfig, **kwargs):
        super().__init__(config)

        dtype = _DTYPE_MAP[config.dtype]
        cls = _get_fastwam_cls(config.model_variant)

        logger.info(
            "Building FastWAM model: loading VAE + text encoder from wan22 pretrained; "
            "DiT weights will be filled by from_pretrained / checkpoint loading."
        )

        # Load VAE + text encoder from wan22 pretrained.
        # skip_dit_load_from_pretrain=True: build random DiT architecture only.
        # The fine-tuned DiT weights are provided by from_pretrained (model.safetensors).
        self.model = cls.from_wan22_pretrained(
            device="cpu",
            torch_dtype=dtype,
            model_id=config.wan22_pretrained_path,
            tokenizer_model_id=config.tokenizer_model_id,
            tokenizer_max_len=config.tokenizer_max_length,
            load_text_encoder=config.load_text_encoder,
            redirect_common_files=config.redirect_common_files,
            video_dit_config=config.get_video_dit_config(),
            action_dit_config=config.get_action_dit_config(),
            skip_dit_load_from_pretrain=True,
            proprio_dim=config.state_dim,
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
            video_train_shift=config.video_train_shift,
            video_infer_shift=config.video_infer_shift,
            video_num_train_timesteps=config.video_num_train_timesteps,
            action_train_shift=config.action_train_shift,
            action_infer_shift=config.action_infer_shift,
            action_num_train_timesteps=config.action_num_train_timesteps,
            loss_lambda_video=config.loss_lambda_video,
            loss_lambda_action=config.loss_lambda_action,
        )

        if config.freeze_vae:
            for p in self.model.vae.parameters():
                p.requires_grad_(False)
        if config.freeze_text_encoder and self.model.text_encoder is not None:
            for p in self.model.text_encoder.parameters():
                p.requires_grad_(False)

        self.reset()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Sync FastWAM's plain `device` attribute (not updated by nn.Module.to()).
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = torch.device(args[0])
        elif "device" in kwargs:
            device = torch.device(kwargs["device"])
        if device is not None:
            self.model.device = device
        return result

    # ------------------------------------------------------------------
    # PreTrainedPolicy interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._queue: deque[Tensor] = deque()

    def get_optim_params(self) -> dict[str, Any]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        self.eval()
        # _run_inference returns (chunk_size, action_dim) for B=1; add batch dim
        chunk = self._run_inference(batch)   # (chunk_size, action_dim) on CPU
        return chunk.unsqueeze(0)            # (1, chunk_size, action_dim)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the next action (B=1, action_dim) from the action queue.

        When the queue is empty, run a new inference pass to fill it with
        ``n_action_steps`` actions from the current observation.
        """
        if not self._queue:
            chunk = self.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
            for a in chunk[0, : self.config.n_action_steps]:
                self._queue.append(a)
        return self._queue.popleft().unsqueeze(0)  # (1, action_dim)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass — delegates to FastWAM.training_loss."""
        video = self._prepare_video_for_training(batch)
        context, context_mask = self._encode_text(batch)
        proprio = self._get_proprio(batch)
        # build_inputs expects proprio as (B, T, D); add T dim if (B, D)
        if proprio is not None and proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        action = batch[ACTION].to(device=self.model.device, dtype=self.model.torch_dtype)
        action_is_pad = batch.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.model.device, dtype=torch.bool)

        sample = {
            "video": video,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "action": action,
            "action_is_pad": action_is_pad,
        }
        loss, loss_dict = self.model.training_loss(sample)
        return loss, {k: v.item() if isinstance(v, Tensor) else v for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, batch: dict[str, Tensor]) -> Tensor:
        """Run one inference pass; return action chunk (chunk_size, action_dim) on CPU."""
        image = self._prepare_image(batch)
        proprio = self._get_proprio(batch)

        task_list = batch.get("task", None)
        task = task_list[0] if (task_list is not None and task_list) else ""
        prompt = PROMPT_TEMPLATE.format(task=task)

        logger.debug("_run_inference: prompt=%r", prompt)
        logger.debug(
            "_run_inference: image shape=%s range=[%.3f, %.3f]",
            tuple(image.shape), image.min().item(), image.max().item(),
        )

        with torch.no_grad():
            result = self.model.infer_action(
                prompt=prompt,
                input_image=image,
                action_horizon=self.config.chunk_size,
                proprio=proprio,
                num_inference_steps=self.config.num_inference_steps,
            )

        actions = result["action"]  # (chunk_size, action_dim) float32 on CPU
        logger.debug(
            "_run_inference: actions mean=%.4f std=%.4f gripper[0]=%.4f",
            actions.mean().item(), actions.std().item(), actions[0, -1].item(),
        )
        return actions

    def _prepare_image(self, batch: dict[str, Tensor]) -> Tensor:
        """Collect camera tensors, concatenate horizontally, and normalise to [-1, 1].

        LiberoProcessorStep already flips images 180° and delivers float32 [0, 1].
        Here we convert [0, 1] → [-1, 1] (what the VAE expects) and concat cameras.
        """
        imgs = []
        for i in range(self.config.num_cameras):
            key = f"{OBS_IMAGES}.image" if i == 0 else f"{OBS_IMAGES}.image{i + 1}"
            img = batch[key]          # (B, [T,] C, H, W) float in [0, 1]
            if img.dim() == 5:
                img = img[:, 0]       # take first temporal step → (B, C, H, W)
            imgs.append(img)

        image = torch.cat(imgs, dim=-1)      # concat along W: (B, C, H, W*N_cam)
        image = image * 2.0 - 1.0           # [0, 1] → [-1, 1]
        return image.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _get_proprio(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Return normalised proprio (1, state_dim) on model device, or None."""
        if OBS_STATE not in batch:
            return None
        s = batch[OBS_STATE]
        if s.dim() == 3:
            s = s[:, 0]  # (B, T, D) → (B, D)
        return s.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _encode_text(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode task descriptions to context embeddings for training."""
        task_list = batch.get("task", None)
        if task_list is None:
            raise ValueError("FastWAM training requires batch['task'] (list of task strings).")
        templated = [PROMPT_TEMPLATE.format(task=t) for t in task_list]
        return self.model.encode_prompt(templated)

    def _prepare_video_for_training(self, batch: dict[str, Tensor]) -> Tensor:
        """Stack multi-camera video tensors into (B, C, T, H, W*N_cam) in [-1, 1].

        Accepts either:
          - 5D (B, T, C, H, W): multi-frame dataset (n_obs_steps > 1)
          - 4D (B, C, H, W):    single-frame dataset (n_obs_steps == 1) — tiled to T=5
        The tiled T=5 case produces a static video; the action loss still trains the policy.
        """
        cam_videos = []
        for i in range(self.config.num_cameras):
            key = f"{OBS_IMAGES}.image" if i == 0 else f"{OBS_IMAGES}.image{i + 1}"
            v = batch[key]  # (B, T, C, H, W) or (B, C, H, W)
            if v.dim() == 4:
                # Single-frame: tile to T=5 (minimum T satisfying T%4==1, T>1)
                v = v.unsqueeze(1).expand(-1, 5, -1, -1, -1)
            cam_videos.append(v)

        # Concat along W, then permute to (B, C, T, H, W*N)
        video = torch.cat(cam_videos, dim=-1)   # (B, T, C, H, W*N)
        video = video.permute(0, 2, 1, 3, 4)   # (B, C, T, H, W*N)
        video = video * 2.0 - 1.0              # [0, 1] → [-1, 1]
        return video.to(device=self.model.device, dtype=self.model.torch_dtype)
