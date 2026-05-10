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

import builtins
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.evo1_model import EVO1
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class EVO1Policy(PreTrainedPolicy):
    config_class = Evo1Config
    name = "evo1"

    def __init__(self, config: Evo1Config, **kwargs):
        super().__init__(config)
        config.validate_features()

        if len(config.image_features) > config.max_views:
            raise ValueError(
                f"EVO1 supports at most {config.max_views} camera streams, got {len(config.image_features)}"
            )

        self.config = config
        self.model = EVO1(self._build_model_config(config))
        self.model.set_finetune_flags()
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool | None = None,
        **kwargs,
    ) -> T:
        if strict is None:
            strict = not (config is not None and getattr(config, "training_stage", None) == "stage2")
        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    @staticmethod
    def _build_model_config(config: Evo1Config) -> dict:
        return {
            "device": config.device,
            "return_cls_only": config.return_cls_only,
            "vlm_name": config.vlm_model_name,
            "vlm_num_layers": config.vlm_num_layers,
            "vlm_dtype": config.vlm_dtype,
            "use_flash_attn": config.use_flash_attn,
            "action_head": config.action_head,
            "action_horizon": config.chunk_size,
            "per_action_dim": config.max_action_dim,
            "state_dim": config.max_state_dim,
            "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim,
            "state_hidden_dim": config.state_hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "num_inference_timesteps": config.num_inference_timesteps,
            "num_categories": config.num_categories,
            "enable_gradient_checkpointing": config.enable_gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": config.gradient_checkpointing_use_reentrant,
            "finetune_vlm": config.finetune_vlm,
            "finetune_language_model": config.finetune_language_model,
            "finetune_vision_model": config.finetune_vision_model,
            "finetune_action_head": config.finetune_action_head,
        }

    @property
    def _camera_keys(self) -> list[str]:
        return list(self.config.image_features)

    @property
    def _env_action_dim(self) -> int:
        action_feature = self.config.action_feature
        if action_feature is None:
            return self.config.max_action_dim
        return int(action_feature.shape[0])

    @property
    def _compute_dtype(self) -> torch.dtype:
        return next(self.model.action_head.parameters()).dtype

    @property
    def _training_compute_dtype(self) -> torch.dtype:
        if str(self.config.device).startswith("cuda"):
            return torch.bfloat16
        return self._compute_dtype

    @property
    def _inference_compute_dtype(self) -> torch.dtype:
        if str(self.config.device).startswith("cuda") and self.config.use_amp:
            return torch.bfloat16
        return self._compute_dtype

    def get_optim_params(self) -> list[dict]:
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_bias = name.endswith("bias") or ".bias" in name
            is_norm = param.dim() == 1 or "norm" in name.lower()
            if is_bias or is_norm:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": self.config.optimizer_weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _normalize_task_batch(self, batch: dict[str, Tensor | list[str] | str]) -> list[str]:
        prompts = batch.get(self.config.task_field)
        if prompts is None and self.config.task_field != "task":
            prompts = batch.get("task")
        if prompts is None:
            raise ValueError(f"EVO1 expects a '{self.config.task_field}' text field in the batch.")
        if isinstance(prompts, str):
            return [prompts]
        if isinstance(prompts, (list, tuple)):
            return [str(prompt) for prompt in prompts]
        raise TypeError(f"Unsupported prompt batch type: {type(prompts)}")

    def _prepare_state(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if OBS_STATE not in batch:
            raise ValueError(f"EVO1 requires '{OBS_STATE}' in the batch.")
        state = batch[OBS_STATE]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state[:, -1]
        elif state.dim() != 2:
            raise ValueError(f"Unsupported state tensor shape for EVO1: {tuple(state.shape)}")
        batch_size, state_dim = state.shape
        if state_dim > self.config.max_state_dim:
            raise ValueError(
                f"State dim {state_dim} exceeds configured max_state_dim {self.config.max_state_dim}"
            )
        explicit_mask = batch.get("state_mask")
        if explicit_mask is not None:
            if explicit_mask.dim() == 1:
                explicit_mask = explicit_mask.unsqueeze(0)
            elif explicit_mask.dim() == 3:
                explicit_mask = explicit_mask[:, -1]
            elif explicit_mask.dim() != 2:
                raise ValueError(
                    f"Unsupported state_mask tensor shape for EVO1: {tuple(explicit_mask.shape)}"
                )
            if explicit_mask.shape != (batch_size, state_dim):
                raise ValueError(
                    f"state_mask shape {tuple(explicit_mask.shape)} does not match state shape {(batch_size, state_dim)}"
                )
        padded = torch.zeros(
            batch_size,
            self.config.max_state_dim,
            dtype=state.dtype,
            device=self.config.device,
        )
        padded[:, :state_dim] = state.to(device=self.config.device)
        mask = torch.zeros(
            batch_size,
            self.config.max_state_dim,
            dtype=torch.bool,
            device=self.config.device,
        )
        if explicit_mask is None:
            mask[:, :state_dim] = True
        else:
            mask[:, :state_dim] = explicit_mask.to(device=self.config.device, dtype=torch.bool)
        return padded.to(dtype=self._compute_dtype), mask

    def _prepare_actions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if ACTION not in batch:
            raise ValueError(f"EVO1 requires '{ACTION}' in the batch for training.")
        action = batch[ACTION]
        if action.dim() == 2:
            action = action.unsqueeze(1)
        batch_size, horizon, action_dim = action.shape
        if horizon != self.config.chunk_size:
            raise ValueError(
                f"EVO1 expects chunk_size={self.config.chunk_size}, got action horizon {horizon}"
            )
        if action_dim > self.config.max_action_dim:
            raise ValueError(
                f"Action dim {action_dim} exceeds configured max_action_dim {self.config.max_action_dim}"
            )
        explicit_mask = batch.get("action_mask")
        if explicit_mask is not None:
            if explicit_mask.dim() == 2:
                if horizon == 1:
                    explicit_mask = explicit_mask.unsqueeze(1)
                else:
                    raise ValueError(
                        f"2D action_mask is only supported when chunk_size=1, got action horizon {horizon}"
                    )
            elif explicit_mask.dim() != 3:
                raise ValueError(
                    f"Unsupported action_mask tensor shape for EVO1: {tuple(explicit_mask.shape)}"
                )
            if explicit_mask.shape != (batch_size, horizon, action_dim):
                raise ValueError(
                    "action_mask shape "
                    f"{tuple(explicit_mask.shape)} does not match action shape {(batch_size, horizon, action_dim)}"
                )
        padded = torch.zeros(
            batch_size,
            horizon,
            self.config.max_action_dim,
            dtype=action.dtype,
            device=self.config.device,
        )
        padded[:, :, :action_dim] = action.to(device=self.config.device)
        mask = torch.zeros(
            batch_size,
            horizon,
            self.config.max_action_dim,
            dtype=torch.bool,
            device=self.config.device,
        )
        if explicit_mask is None:
            mask[:, :, :action_dim] = True
        else:
            mask[:, :, :action_dim] = explicit_mask.to(device=self.config.device, dtype=torch.bool)
        return padded.to(dtype=self._compute_dtype), mask

    def _prepare_inference_action_mask(self, batch_size: int) -> Tensor:
        mask = torch.zeros(
            batch_size,
            self.config.max_action_dim,
            dtype=torch.bool,
            device=self.config.device,
        )
        mask[:, : self._env_action_dim] = True
        return mask

    def _get_embodiment_ids(self, batch: dict[str, Tensor], batch_size: int) -> Tensor:
        embodiment_ids = batch.get("embodiment_id")
        if embodiment_ids is None and self.config.embodiment_id_field:
            embodiment_ids = batch.get(self.config.embodiment_id_field)
        if embodiment_ids is None:
            return torch.full(
                (batch_size,),
                self.config.default_embodiment_id,
                dtype=torch.long,
                device=self.config.device,
            )
        if embodiment_ids.dim() == 0:
            embodiment_ids = embodiment_ids.unsqueeze(0)
        elif embodiment_ids.dim() > 1:
            embodiment_ids = embodiment_ids[:, -1]
        return embodiment_ids.to(device=self.config.device, dtype=torch.long)

    def _collect_image_batches(self, batch: dict[str, Tensor]) -> tuple[list[list[Tensor]], Tensor]:
        camera_keys = self._camera_keys or sorted(key for key in batch if key.startswith(f"{OBS_IMAGES}."))
        if not camera_keys:
            raise ValueError("EVO1 requires at least one visual observation feature.")

        # Normalize each camera tensor to (B, C, H, W) up-front so that batch_size is read
        # from a real batch dim and not from C in the unbatched (C, H, W) case.
        normalized: dict[str, Tensor] = {}
        for camera_key in camera_keys[: self.config.max_views]:
            image = batch[camera_key]
            if image.dim() == 3:
                image = image.unsqueeze(0)
            elif image.dim() == 5:
                image = image[:, -1]
            elif image.dim() != 4:
                raise ValueError(
                    f"Unsupported image tensor shape for EVO1: key={camera_key} shape={tuple(image.shape)}"
                )
            normalized[camera_key] = image

        batch_size = normalized[camera_keys[0]].shape[0]
        image_batches: list[list[Tensor]] = []
        image_masks = torch.zeros(batch_size, self.config.max_views, dtype=torch.bool)

        for batch_index in range(batch_size):
            sample_images: list[Tensor] = []
            for camera_key in camera_keys[: self.config.max_views]:
                sample_images.append(normalized[camera_key][batch_index].detach().cpu())
            if not sample_images:
                raise ValueError("EVO1 received a batch without any image tensor.")
            while len(sample_images) < self.config.max_views:
                sample_images.append(torch.zeros_like(sample_images[0]))
            image_batches.append(sample_images[: self.config.max_views])
            image_masks[batch_index, : min(len(camera_keys), self.config.max_views)] = True

        return image_batches, image_masks

    def _compute_fused_tokens(
        self,
        prompts: list[str],
        image_batches: list[list[Tensor]],
        image_masks: Tensor,
    ) -> Tensor:
        fused_tokens = self.model.get_vl_embeddings(
            images=image_batches,
            image_mask=image_masks,
            prompt=prompts,
            return_cls_only=self.config.return_cls_only,
        )
        return fused_tokens.to(device=self.config.device, dtype=self._compute_dtype)

    def _compute_masked_loss(
        self,
        pred_velocity: Tensor,
        target_velocity: Tensor,
        action_mask: Tensor,
        reduction: str,
    ) -> Tensor:
        flat_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
        sq_error = ((pred_velocity - target_velocity) * flat_mask).pow(2)
        active = flat_mask.sum(dim=1).clamp_min(1.0)
        per_sample_loss = sq_error.sum(dim=1) / active
        if reduction == "none":
            return per_sample_loss
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction '{reduction}'")
        return sq_error.sum() / active.sum()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        prompts = self._normalize_task_batch(batch)
        image_batches, image_masks = self._collect_image_batches(batch)
        states, _state_mask = self._prepare_state(batch)
        actions_gt, action_mask = self._prepare_actions(batch)
        fused_tokens = self._compute_fused_tokens(prompts, image_batches, image_masks)
        states = states.to(dtype=self._training_compute_dtype)
        actions_gt = actions_gt.to(dtype=self._training_compute_dtype)
        fused_tokens = fused_tokens.to(dtype=self._training_compute_dtype)
        embodiment_ids = self._get_embodiment_ids(batch, states.shape[0])

        pred_velocity, noise = self.model(
            fused_tokens,
            state=states,
            actions_gt=actions_gt,
            action_mask=action_mask.to(device=self.config.device, dtype=self._compute_dtype),
            embodiment_ids=embodiment_ids,
        )
        flat_action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=actions_gt.dtype)
        target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1) * flat_action_mask
        loss = self._compute_masked_loss(pred_velocity, target_velocity, action_mask, reduction)
        loss_mean = loss.mean().item() if loss.ndim > 0 else loss.item()
        return loss, {
            "loss": loss_mean,
            "active_action_dims": float(action_mask.sum(dim=(1, 2)).float().mean().item()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        prompts = self._normalize_task_batch(batch)
        image_batches, image_masks = self._collect_image_batches(batch)
        states, _state_mask = self._prepare_state(batch)
        fused_tokens = self._compute_fused_tokens(prompts, image_batches, image_masks)
        states = states.to(dtype=self._inference_compute_dtype)
        fused_tokens = fused_tokens.to(dtype=self._inference_compute_dtype)
        embodiment_ids = self._get_embodiment_ids(batch, states.shape[0])
        action_mask = self._prepare_inference_action_mask(states.shape[0])

        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.config.use_amp and str(self.config.device).startswith("cuda")
            else nullcontext()
        ):
            actions = self.model(
                fused_tokens,
                state=states,
                action_mask=action_mask,
                embodiment_ids=embodiment_ids,
            )
        actions = actions.view(states.shape[0], self.config.chunk_size, self.config.max_action_dim)
        return actions[:, :, : self._env_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(action_chunk.transpose(0, 1))
        return self._action_queue.popleft()
