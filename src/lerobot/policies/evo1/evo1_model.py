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

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from lerobot.policies.evo1.flow_matching import FlowmatchingActionHead
from lerobot.policies.evo1.internvl3_embedder import InternVL3Embedder


def _cfgget(config: Any, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._device = _cfgget(config, "device", "cuda")
        self.return_cls_only = _cfgget(config, "return_cls_only", False)
        vlm_name = _cfgget(config, "vlm_name", "OpenGVLab/InternVL3-1B")
        image_size = _cfgget(config, "image_size", 448)
        if image_size is None:
            image_resolution = _cfgget(config, "image_resolution", (448, 448))
            image_size = int(image_resolution[0])

        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            image_size=image_size,
            device=self._device,
            num_language_layers=_cfgget(config, "vlm_num_layers", 14),
            model_dtype=_cfgget(config, "vlm_dtype", "bfloat16"),
            use_flash_attn=_cfgget(config, "use_flash_attn", True),
            enable_gradient_checkpointing=_cfgget(config, "enable_gradient_checkpointing", True),
            gradient_checkpointing_use_reentrant=_cfgget(
                config, "gradient_checkpointing_use_reentrant", False
            ),
        )

        action_head_type = _cfgget(config, "action_head", "flowmatching").lower()
        if action_head_type != "flowmatching":
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        horizon = _cfgget(config, "action_horizon", _cfgget(config, "horizon", 16))
        per_action_dim = _cfgget(config, "per_action_dim", 7)
        action_dim = horizon * per_action_dim

        if isinstance(config, dict):
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim

        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_head = FlowmatchingActionHead(config=config).to(self._device)

    def _normalize_image_batches(
        self,
        images: Sequence[Image.Image | torch.Tensor] | Sequence[Sequence[Image.Image | torch.Tensor]],
        prompt: str | list[str] | None,
        image_mask: torch.Tensor,
    ) -> tuple[list[list[Image.Image | torch.Tensor]], list[str], torch.Tensor]:
        if not images:
            raise ValueError("EVO1 expects at least one image per sample.")

        first = images[0]
        if isinstance(first, (Image.Image, torch.Tensor)):
            image_batches = [list(images)]  # type: ignore[arg-type]
        else:
            image_batches = [list(sample) for sample in images]  # type: ignore[arg-type]

        batch_size = len(image_batches)
        if prompt is None:
            prompts = [""] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        else:
            prompts = [str(p) for p in prompt]
            if len(prompts) != batch_size:
                raise ValueError(
                    f"Prompt batch size {len(prompts)} does not match image batch size {batch_size}"
                )

        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
        if image_mask.shape[0] != batch_size:
            raise ValueError(
                f"image_mask batch size {image_mask.shape[0]} does not match image batch size {batch_size}"
            )

        return image_batches, prompts, image_mask

    def get_vl_embeddings(
        self,
        images: list[Image.Image | torch.Tensor] | list[list[Image.Image | torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str | list[str] | None = None,
        return_cls_only: bool | None = None,
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        image_batches, prompts, image_mask = self._normalize_image_batches(images, prompt, image_mask)
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors_batch=image_batches,
            image_masks=image_mask,
            text_prompts=prompts,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: list | torch.Tensor) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError(f"Unsupported state input type: {type(state_input)}")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
    ):
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
            )
        return self.action_head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_ids,
        )

    @torch.no_grad()
    def run_inference(
        self,
        images: list[Image.Image | torch.Tensor],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: list | torch.Tensor,
        return_cls_only: bool | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)

        fused_tokens = self.get_vl_embeddings(
            images=[images],
            image_mask=image_mask,
            prompt=[prompt],
            return_cls_only=return_cls_only,
        )
        state_tensor = self.prepare_state(state_input)
        action = self.predict_action(
            fused_tokens,
            state_tensor,
            action_mask=action_mask,
            embodiment_ids=embodiment_ids,
        )
        if isinstance(action, torch.Tensor) and action.dtype == torch.bfloat16:
            action = action.to(torch.float32)
        return action

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
    ):
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _set_module_trainable(self, module: nn.Module, trainable: bool):
        for param in module.parameters():
            param.requires_grad = trainable

    def set_finetune_flags(self):
        finetune_vlm = _cfgget(self.config, "finetune_vlm", False)
        finetune_language_model = _cfgget(self.config, "finetune_language_model", False)
        finetune_vision_model = _cfgget(self.config, "finetune_vision_model", False)
        has_explicit_branch_flags = any(
            flag is not None for flag in (finetune_language_model, finetune_vision_model)
        )
        finetune_language_model = bool(finetune_language_model)
        finetune_vision_model = bool(finetune_vision_model)
        finetune_vlm = bool(finetune_vlm)

        if has_explicit_branch_flags:
            self._set_module_trainable(self.embedder, False)
            if hasattr(self.embedder.model, "language_model"):
                self._set_module_trainable(self.embedder.model.language_model, finetune_language_model)
            if hasattr(self.embedder.model, "vision_model"):
                self._set_module_trainable(self.embedder.model.vision_model, finetune_vision_model)
            if hasattr(self.embedder.model, "mlp1"):
                self._set_module_trainable(self.embedder.model.mlp1, finetune_vision_model)
        elif not finetune_vlm:
            self._set_module_trainable(self.embedder, False)

        if not _cfgget(self.config, "finetune_action_head", False):
            self._set_module_trainable(self.action_head, False)
