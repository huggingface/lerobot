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
from typing import TypedDict, Unpack

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.modeling_rtc import RTCProcessor
from .configuration_evo1 import Evo1Config
from .evo1_model import Evo1Model


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


class Evo1Policy(PreTrainedPolicy):
    config_class = Evo1Config
    name = "evo1"

    def __init__(self, config: Evo1Config, *, vlm_hub_kwargs: dict | None = None, **kwargs):
        super().__init__(config)
        config.validate_features()

        if len(config.image_features) > config.max_views:
            raise ValueError(
                f"EVO1 supports at most {config.max_views} camera streams, got {len(config.image_features)}"
            )

        self.config = config
        self.model = Evo1Model(config, vlm_hub_kwargs=vlm_hub_kwargs)
        self.model.set_finetune_flags()
        self._keep_frozen_embedder_eval()
        self.init_rtc_processor()
        self.reset()

    def init_rtc_processor(self):
        """Create the RTC processor when config.rtc_config is set.

        The RTC rollout backend assigns config.rtc_config after loading the policy and re-invokes
        this method.
        """
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
        model = getattr(self, "model", None)
        if model is not None:
            model.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

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
            strict = True
        vlm_hub_kwargs = kwargs.pop("vlm_hub_kwargs", None)
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if vlm_hub_kwargs is None:
            # Forward the hub download options to the base-VLM download as well; `revision` is not
            # forwarded because it identifies the policy repo, not the VLM repo.
            vlm_hub_kwargs = {
                key: value
                for key, value in (
                    ("token", token),
                    ("cache_dir", cache_dir),
                    ("local_files_only", local_files_only),
                    ("proxies", proxies),
                )
                if value not in (None, False)
            }
        kwargs["vlm_hub_kwargs"] = vlm_hub_kwargs
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
    def _device(self) -> torch.device:
        # The device the policy actually lives on. Derived from the parameters rather than
        # config.device so the policy keeps working after accelerate (or a plain .to()) moves it.
        return next(self.model.action_head.parameters()).device

    @property
    def _amp_enabled(self) -> bool:
        return bool(self.config.use_amp) and self._device.type == "cuda"

    def _maybe_autocast(self):
        # EVO1 manages its own mixed precision: an explicit bf16 autocast that also overrides any
        # outer autocast context (e.g. lerobot-eval's fp16 default), keeping train and eval
        # numerics identical.
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

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
        device = self._device
        padded = torch.zeros(
            batch_size,
            self.config.max_state_dim,
            dtype=state.dtype,
            device=device,
        )
        padded[:, :state_dim] = state.to(device=device)
        mask = torch.zeros(
            batch_size,
            self.config.max_state_dim,
            dtype=torch.bool,
            device=device,
        )
        if explicit_mask is None:
            mask[:, :state_dim] = True
        else:
            mask[:, :state_dim] = explicit_mask.to(device=device, dtype=torch.bool)
        # Zero out masked state dims so an explicit state_mask actually affects the model input
        # (the state encoder has no mask argument of its own).
        padded = padded * mask.to(dtype=padded.dtype)
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
        device = self._device
        padded = torch.zeros(
            batch_size,
            horizon,
            self.config.max_action_dim,
            dtype=action.dtype,
            device=device,
        )
        padded[:, :, :action_dim] = action.to(device=device)
        mask = torch.zeros(
            batch_size,
            horizon,
            self.config.max_action_dim,
            dtype=torch.bool,
            device=device,
        )
        if explicit_mask is None:
            mask[:, :, :action_dim] = True
        else:
            mask[:, :, :action_dim] = explicit_mask.to(device=device, dtype=torch.bool)

        # Timesteps beyond the episode end hold fabricated (repeated) actions; exclude them from
        # the loss like the other chunked policies do.
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.shape != (batch_size, horizon):
                raise ValueError(
                    f"action_is_pad shape {tuple(action_is_pad.shape)} does not match "
                    f"(batch_size, chunk_size)={(batch_size, horizon)}"
                )
            in_episode = ~action_is_pad.to(device=device, dtype=torch.bool)
            mask = mask & in_episode.unsqueeze(-1)
        return padded.to(dtype=self._compute_dtype), mask

    def _prepare_inference_action_mask(self, batch_size: int) -> Tensor:
        mask = torch.zeros(
            batch_size,
            self.config.max_action_dim,
            dtype=torch.bool,
            device=self._device,
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
                device=self._device,
            )
        if embodiment_ids.dim() == 0:
            embodiment_ids = embodiment_ids.unsqueeze(0)
        elif embodiment_ids.dim() > 1:
            embodiment_ids = embodiment_ids[:, -1]
        return embodiment_ids.to(device=self._device, dtype=torch.long)

    @property
    def _tracks_vlm_gradients(self) -> bool:
        return bool(
            self.config.finetune_vlm
            or self.config.finetune_language_model
            or self.config.finetune_vision_model
        )

    def _keep_frozen_embedder_eval(self) -> None:
        if self._tracks_vlm_gradients:
            return
        embedder = getattr(self.model, "embedder", None)
        if embedder is not None:
            embedder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._keep_frozen_embedder_eval()
        return self

    def _collect_image_batches(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], Tensor]:
        camera_keys = self._camera_keys or sorted(key for key in batch if key.startswith(f"{OBS_IMAGES}."))
        if not camera_keys:
            raise ValueError("EVO1 requires at least one visual observation feature.")
        camera_keys = list(camera_keys)[: self.config.max_views]

        # Configured cameras may be absent from the batch up to the empty_cameras budget (e.g. the
        # placeholder features added by validate_features); they become masked-out views that the
        # embedder zero-pads. Any other absent camera is an error.
        present_keys = [key for key in camera_keys if key in batch]
        missing_keys = [key for key in camera_keys if key not in batch]
        if len(missing_keys) > self.config.empty_cameras:
            raise ValueError(
                f"Missing camera features {missing_keys} in batch; at most "
                f"empty_cameras={self.config.empty_cameras} may be absent."
            )
        if not present_keys:
            raise ValueError("EVO1 requires at least one visual observation in the batch.")

        # Keep each present camera as a batched (B, C, H, W) tensor on its current (GPU) device.
        # Resizing/normalization and zero-padding of absent views happen batched inside the
        # embedder, so images never leave the device here.
        camera_images: list[Tensor] = []
        for camera_key in present_keys:
            image = batch[camera_key]
            if image.dim() == 3:
                # Promote an unbatched (C, H, W) frame so batch_size is read from a real batch dim.
                image = image.unsqueeze(0)
            elif image.dim() == 5:
                image = image[:, -1]
            elif image.dim() != 4:
                raise ValueError(
                    f"Unsupported image tensor shape for EVO1: key={camera_key} shape={tuple(image.shape)}"
                )
            camera_images.append(image)

        batch_size = camera_images[0].shape[0]
        n_present = len(camera_images)
        image_masks = torch.zeros(
            batch_size, self.config.max_views, dtype=torch.bool, device=camera_images[0].device
        )
        image_masks[:, :n_present] = True

        return camera_images, image_masks

    def _compute_fused_tokens(
        self,
        prompts: list[str],
        image_batches: list[Tensor],
        image_masks: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        track_vlm_gradients = self._tracks_vlm_gradients
        grad_context = nullcontext() if track_vlm_gradients else torch.no_grad()
        with grad_context:
            fused_tokens, context_mask = self.model.get_vl_embeddings(
                images=image_batches,
                image_mask=image_masks,
                prompt=prompts,
                return_cls_only=self.config.return_cls_only,
            )

        if not track_vlm_gradients:
            fused_tokens = fused_tokens.detach()
        fused_tokens = fused_tokens.to(device=self._device, dtype=self._compute_dtype)
        if context_mask is not None:
            context_mask = context_mask.to(device=self._device)
        return fused_tokens, context_mask

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
        embodiment_ids = self._get_embodiment_ids(batch, states.shape[0])

        with self._maybe_autocast():
            fused_tokens, context_mask = self._compute_fused_tokens(prompts, image_batches, image_masks)
            pred_velocity, noise = self.model(
                fused_tokens,
                state=states,
                actions_gt=actions_gt,
                action_mask=action_mask.to(device=self._device, dtype=self._compute_dtype),
                embodiment_ids=embodiment_ids,
                context_mask=context_mask,
            )

        # Compute the flow-matching regression loss in fp32, outside the autocast block.
        pred_velocity = pred_velocity.float()
        noise = noise.float()
        flat_action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=torch.float32)
        # Flow-matching velocity target. Padded (masked-out) action dims are already zero on both sides
        # here (`actions_gt` is zero-padded in `_prepare_actions`, and `noise` is masked inside the head),
        # and the whole difference is multiplied by `flat_action_mask`, so padded dims contribute nothing.
        target_velocity = (actions_gt.float() - noise).view(actions_gt.shape[0], -1) * flat_action_mask
        loss = self._compute_masked_loss(pred_velocity, target_velocity, action_mask, reduction)
        loss_mean = loss.mean().item() if loss.ndim > 0 else loss.item()
        return loss, {
            "loss": loss_mean,
            "active_action_dims": float(action_mask.sum(dim=(1, 2)).float().mean().item()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        inference_delay = kwargs.get("inference_delay")
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
        execution_horizon = kwargs.get("execution_horizon")
        if (inference_delay is not None or prev_chunk_left_over is not None) and not self._rtc_enabled():
            raise RuntimeError(
                "Received RTC arguments but RTC is not configured for this EVO1 policy: set "
                "config.rtc_config and call init_rtc_processor() (lerobot-rollout does this for "
                "--inference.type=rtc)."
            )
        self.eval()

        prompts = self._normalize_task_batch(batch)
        image_batches, image_masks = self._collect_image_batches(batch)
        states, _state_mask = self._prepare_state(batch)
        embodiment_ids = self._get_embodiment_ids(batch, states.shape[0])
        action_mask = self._prepare_inference_action_mask(states.shape[0])
        if prev_chunk_left_over is not None:
            prev_chunk_left_over = prev_chunk_left_over.to(device=self._device)

        with self._maybe_autocast():
            fused_tokens, context_mask = self._compute_fused_tokens(prompts, image_batches, image_masks)
            actions = self.model(
                fused_tokens,
                state=states,
                action_mask=action_mask,
                embodiment_ids=embodiment_ids,
                context_mask=context_mask,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
            )
        actions = actions.view(states.shape[0], self.config.chunk_size, self.config.max_action_dim)
        return actions.to(dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )
        self.eval()
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(action_chunk.transpose(0, 1))
        # Returns one step of shape (B, max_action_dim): actions are emitted at the padded max_action_dim
        # width and cropped to the real action dim downstream by the postprocessor (Evo1ActionProcessorStep).
        # Callers that bypass the postprocessor receive the padded width.
        return self._action_queue.popleft()
