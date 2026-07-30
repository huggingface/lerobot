# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot policy wrapper for Hy-Embodied-0.5-VLA."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_hy_vla import HyVLAConfig


def _resize_with_pad(image: Tensor, height: int, width: int, pad_value: float = 0) -> Tensor:
    if image.shape[-2:] == (height, width):
        return image
    ratio = max(image.shape[-2] / height, image.shape[-1] / width)
    resized_height = max(1, int(image.shape[-2] / ratio))
    resized_width = max(1, int(image.shape[-1] / ratio))
    resized = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height, pad_width = height - resized_height, width - resized_width
    top, left = pad_height // 2, pad_width // 2
    return F.pad(
        resized,
        (left, pad_width - left, top, pad_height - top),
        value=pad_value,
    )


def _pad_last(value: Tensor, dimension: int) -> Tensor:
    if value.shape[-1] > dimension:
        raise ValueError(f"Cannot pad dimension {value.shape[-1]} to {dimension}.")
    if value.shape[-1] == dimension:
        return value
    output = value.new_zeros(*value.shape[:-1], dimension)
    output[..., : value.shape[-1]] = value
    return output


class HyVLAPolicy(PreTrainedPolicy):
    """First-class LeRobot policy for the released Hy-VLA checkpoints."""

    config_class = HyVLAConfig
    name = "hy_vla"

    def __init__(self, config: HyVLAConfig, **_: Any):
        require_package("transformers", extra="hy_vla")
        super().__init__(config)
        config.validate_features()

        # Heavy imports stay behind the optional dependency checks.
        from transformers import AutoTokenizer

        from .modeling.modeling_hy_vla import HyVLAFlowMatching

        tokenizer_source = getattr(config, "_tokenizer_source", None) or config.vlm_model_path
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False,
            revision=getattr(config, "_tokenizer_revision", None),
            fix_mistral_regex=True,
        )
        self.model = HyVLAFlowMatching(config, self.language_tokenizer)
        # The action expert consumes injected action/state embeddings and never
        # produces vocabulary logits. The author checkpoints therefore omit
        # this otherwise randomly initialized, unused CausalLM output head.
        self.model.dual_tower.expert.lm_head = None
        # The released runtime trains and evaluates the complete policy in
        # BF16. LeRobot's generic factory moves policies to a device but does
        # not change dtype, so establish the checkpoint dtype here instead of
        # requiring a private training runner to call ``policy.to(bfloat16)``.
        self.model.to(dtype=torch.bfloat16)
        self.enable_video_encoder_if_needed()
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        revision: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> HyVLAPolicy:
        if config is None:
            config = HyVLAConfig.from_pretrained(
                pretrained_name_or_path,
                revision=revision,
                force_download=kwargs.get("force_download", False),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
                token=kwargs.get("token"),
            )
        if not isinstance(config, HyVLAConfig):
            raise TypeError(f"Expected HyVLAConfig, got {type(config)!r}.")
        # Transient fields are intentionally not dataclass fields, so absolute
        # local paths are never baked into config.json on the next save.
        config._tokenizer_source = str(pretrained_name_or_path)
        config._tokenizer_revision = revision
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        if not self.config.vlm_config_dict:
            self.config.vlm_config_dict = self.model.dual_tower.vlm.config.to_dict()
        super()._save_pretrained(save_directory, state_dict=state_dict)
        self.language_tokenizer.save_pretrained(str(save_directory))

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.execution_horizon)
        history_span = (self.config.img_history_size - 1) * self.config.img_history_interval + 1
        self._image_history: dict[str, deque[Tensor]] = {
            key: deque(maxlen=history_span) for key in self.config.image_features
        }

    def _append_inference_history(self, batch: dict[str, Any]) -> None:
        """Record every eval frame, including frames served from the action queue."""

        for key, history in self._image_history.items():
            image = batch.get(key)
            if not isinstance(image, Tensor) or image.ndim != 4:
                raise ValueError(f"MEM inference requires {key} as BCHW, got {getattr(image, 'shape', None)}")
            history.append(image.detach().to("cpu"))

    def _with_inference_history(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Build the author's deterministic K-frame, interval-S MEM stacks."""

        output = dict(batch)
        history_size = self.config.img_history_size
        interval = self.config.img_history_interval
        for key, history_queue in self._image_history.items():
            history = list(history_queue)
            step = len(history) - 1
            current = history[-1]
            frames: list[Tensor] = []
            for slot in range(history_size):
                unclipped = step - (history_size - 1 - slot) * interval
                frames.append(torch.zeros_like(current) if unclipped < 0 else history[unclipped])
            output[key] = torch.stack(frames, dim=1).to(batch[key])
        return output

    def enable_video_encoder_if_needed(self) -> None:
        if not self.config.use_video_encoder:
            return
        from .modeling.space_time_attention import apply_video_encoder_patch

        apply_video_encoder_patch(
            self.model.dual_tower.vlm.model.visual,
            spacetime_layer_stride=self.config.spacetime_layer_stride,
            past_drop_layer=self.config.past_drop_layer,
            max_num_frames=self.config.max_num_frames,
        )

    def get_optim_params(self):
        return self.parameters()

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.model.parameters())
        return parameter.device, parameter.dtype

    def prepare_images(self, batch: dict[str, Any]) -> tuple[list[Tensor], list[Tensor]]:
        model_device, model_dtype = self._model_device_dtype()
        expected_keys = list(self.config.image_features)
        present_keys = [key for key in expected_keys if key in batch]
        missing_keys = [key for key in expected_keys if key not in batch]
        if not present_keys:
            raise ValueError(f"All expected image features are missing: {expected_keys}")

        processed: dict[str, tuple[Tensor, Tensor]] = {}
        template: Tensor | None = None
        template_mask: Tensor | None = None
        for key in present_keys:
            image = batch[key]
            if not isinstance(image, Tensor) or image.ndim not in {4, 5}:
                raise ValueError(
                    f"{key} must be BCHW or BKCHW tensor, got {type(image)} {getattr(image, 'shape', None)}"
                )
            image = image.to(device=model_device, dtype=model_dtype)
            if image.ndim == 5 and not self.config.use_video_encoder:
                image = image[:, -1]
            elif image.ndim == 4 and self.config.use_video_encoder:
                image = image.unsqueeze(1)
            if self.config.resize_imgs_with_padding is not None:
                height, width = self.config.resize_imgs_with_padding
                if image.ndim == 5:
                    batch_size, history, channels, old_height, old_width = image.shape
                    image = _resize_with_pad(
                        image.reshape(batch_size * history, channels, old_height, old_width),
                        height,
                        width,
                    ).reshape(batch_size, history, channels, height, width)
                else:
                    image = _resize_with_pad(image, height, width)
            image = image * 2 - 1
            mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            processed[key] = (image, mask)
            template, template_mask = image, mask

        if len(missing_keys) > self.config.empty_cameras:
            raise ValueError(
                f"Missing {len(missing_keys)} camera(s), but empty_cameras={self.config.empty_cameras}: "
                f"{missing_keys}"
            )
        assert template is not None and template_mask is not None
        images: list[Tensor] = []
        masks: list[Tensor] = []
        for key in expected_keys:
            if key in processed:
                image, mask = processed[key]
            else:
                # Preserve the configured camera-slot order. Appending all
                # empty cameras at the end would silently move a later real
                # camera into the wrong visual segment.
                image = torch.full_like(template, -1)
                mask = torch.zeros_like(template_mask)
            images.append(image)
            masks.append(mask)
        return images, masks

    def _format_tasks(self, tasks: list[str]) -> list[str]:
        """Apply only model chat formatting; preserve every raw task byte."""

        self._last_raw_tasks = tuple(tasks)
        return [
            task if task.endswith(self.config.task_suffix) else task + self.config.task_suffix
            for task in tasks
        ]

    def prepare_language(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        device = next(
            value.device
            for key, value in batch.items()
            if key.startswith(OBS_IMAGES) and isinstance(value, Tensor)
        )
        raw_tasks = batch.get("task")
        if isinstance(raw_tasks, str):
            raw_tasks = [raw_tasks]
        if not isinstance(raw_tasks, list | tuple) or not all(isinstance(task, str) for task in raw_tasks):
            raise ValueError("Hy-VLA requires an already-selected raw LeRobot task string per sample.")
        tasks = self._format_tasks(list(raw_tasks))
        labels = batch.get("text_label")
        if labels is not None:
            labels = [label + self.language_tokenizer.eos_token for label in labels]
        tokenized = self.language_tokenizer(
            tasks,
            text_pair=labels,
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=True,
        )
        tokens = tokenized["input_ids"].to(device)
        masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
        token_types = tokenized.get("token_type_ids", torch.zeros_like(tokens)).to(device)
        return tokens, masks, token_types

    def prepare_state(self, batch: dict[str, Any]) -> Tensor:
        model_device, model_dtype = self._model_device_dtype()
        return _pad_last(batch[OBS_STATE], self.config.max_state_dim).to(
            device=model_device, dtype=model_dtype
        )

    def prepare_action(self, batch: dict[str, Any]) -> Tensor:
        model_device, model_dtype = self._model_device_dtype()
        return _pad_last(batch[ACTION], self.config.max_action_dim).to(device=model_device, dtype=model_dtype)

    def _prepare_model_inputs(self, batch: dict[str, Any]):
        images, image_masks = self.prepare_images(batch)
        tokens, language_masks, token_types = self.prepare_language(batch)
        return images, image_masks, tokens, language_masks, token_types

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        images, image_masks, tokens, language_masks, token_types = self._prepare_model_inputs(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        labels = None
        if batch.get("text_label") is not None:
            labels = tokens.masked_fill(
                token_types == self.model.dual_tower.vlm.config.pad_token_id,
                self.model.dual_tower.vlm.config.ignore_index,
            )
        flow_losses, language_losses = self.model(
            images,
            image_masks,
            tokens,
            language_masks,
            state,
            actions,
            noise,
            time,
            labels,
        )
        if flow_losses is None:
            raise RuntimeError("Hy-VLA training requires an action target.")
        flow_losses = flow_losses[..., : self.config.model_action_dim]
        action_mask = batch.get(f"{ACTION}.mask")
        if action_mask is not None:
            if not isinstance(action_mask, Tensor):
                raise ValueError(f"{ACTION}.mask must be a tensor.")
            action_mask = action_mask[..., : self.config.model_action_dim].to(
                device=flow_losses.device, dtype=flow_losses.dtype
            )
            if action_mask.shape != flow_losses.shape:
                action_mask = torch.broadcast_to(action_mask, flow_losses.shape)
            valid_per_sample = action_mask.sum(dim=(-2, -1)).clamp_min(1)
            flow_loss_per_sample = (flow_losses * action_mask).sum(dim=(-2, -1)) / valid_per_sample
        else:
            flow_loss_per_sample = flow_losses.mean(dim=(-2, -1))
        if reduction == "none":
            flow_loss = flow_loss_per_sample
        elif reduction == "mean":
            flow_loss = flow_loss_per_sample.mean()
        else:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        language_loss = language_losses.mean() if language_losses is not None else flow_loss.new_zeros(())
        loss = flow_loss + language_loss
        return loss, {
            "loss": loss,
            "flow_loss": flow_loss.detach(),
            "language_loss": language_loss.detach(),
        }

    def _pair_relative_absolute(self, actions: Tensor) -> Tensor:
        if self.config.action_representation == "relative":
            return actions
        horizon = self.config.physical_action_horizon
        if actions.shape[-2] != 2 * horizon:
            raise RuntimeError(f"Expected {2 * horizon} rel/abs tokens, got {actions.shape[-2]}.")
        return torch.cat((actions[:, :horizon], actions[:, horizon:]), dim=-1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        if self.config.use_video_encoder:
            self._append_inference_history(batch)
        if not self._action_queue:
            model_batch = self._with_inference_history(batch) if self.config.use_video_encoder else batch
            images, image_masks, tokens, language_masks, _ = self._prepare_model_inputs(model_batch)
            actions = self.model.sample_actions(
                images,
                image_masks,
                tokens,
                language_masks,
                self.prepare_state(model_batch),
                noise=noise,
                vis_attn=False,
            )
            actions = actions[..., : self.config.model_action_dim]
            actions = self._pair_relative_absolute(actions)
            self._action_queue.extend(actions[:, : self.config.execution_horizon].transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return the complete normalized chunk without mutating the action queue."""

        self.eval()
        images, image_masks, tokens, language_masks, _ = self._prepare_model_inputs(batch)
        actions = self.model.sample_actions(
            images,
            image_masks,
            tokens,
            language_masks,
            self.prepare_state(batch),
            noise=noise,
            vis_attn=False,
        )
        return self._pair_relative_absolute(actions[..., : self.config.model_action_dim])


__all__ = ["HyVLAPolicy"]
