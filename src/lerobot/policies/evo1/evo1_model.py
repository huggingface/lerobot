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

import torch
import torch.nn as nn

from .configuration_evo1 import Evo1Config
from .flow_matching import FlowmatchingActionHead
from .internvl3_embedder import InternVL3Embedder


class Evo1Model(nn.Module):
    def __init__(self, config: Evo1Config, vlm_hub_kwargs: dict | None = None):
        super().__init__()
        self.config = config
        self._device = config.device
        self.return_cls_only = config.return_cls_only
        # Set by Evo1Policy.init_rtc_processor() when config.rtc_config is provided.
        self.rtc_processor = None

        # Gradient checkpointing only pays off when the VLM is actually being trained; keep it off
        # whenever every VLM branch is frozen so the frozen forward stays cheap.
        tracks_vlm_gradients = bool(
            config.finetune_vlm or config.finetune_language_model or config.finetune_vision_model
        )
        enable_gradient_checkpointing = config.enable_gradient_checkpointing and tracks_vlm_gradients

        self.embedder = InternVL3Embedder(
            model_name=config.vlm_model_name,
            image_size=int(config.image_resolution[0]),
            device=self._device,
            num_language_layers=config.vlm_num_layers,
            model_dtype=config.vlm_dtype,
            use_flash_attn=config.use_flash_attn,
            max_text_length=config.max_text_length,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            gradient_checkpointing_use_reentrant=config.gradient_checkpointing_use_reentrant,
            hub_kwargs=vlm_hub_kwargs,
        )

        action_head_type = config.action_head.lower()
        if action_head_type != "flowmatching":
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        horizon = config.chunk_size
        per_action_dim = config.max_action_dim
        action_dim = horizon * per_action_dim

        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_head = FlowmatchingActionHead(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            action_dim=action_dim,
            horizon=horizon,
            per_action_dim=per_action_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_inference_timesteps=config.num_inference_timesteps,
            num_categories=config.num_categories,
            state_dim=config.max_state_dim,
            state_hidden_dim=config.state_hidden_dim,
        ).to(self._device)

    def get_vl_embeddings(
        self,
        images: list[torch.Tensor],
        image_mask: torch.Tensor,
        prompt: str | list[str] | None = None,
        return_cls_only: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fused VL embeddings from per-camera image batches.

        Args:
            images: list of per-camera tensors, each shaped ``(B, C, H, W)`` with values in ``[0, 1]``.
            image_mask: bool tensor ``(B, max_views)`` marking present views.

        Returns:
            ``(embeddings, valid_mask)``: the fused tokens and the bool mask of attendable context
            positions (None when a single pooled token is returned).
        """
        if return_cls_only is None:
            return_cls_only = self.return_cls_only
        if not images:
            raise ValueError("EVO1 expects at least one image per sample.")

        batch_size = images[0].shape[0]
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

        return self.embedder.get_fused_image_text_embedding_batched(
            camera_images=images,
            image_masks=image_mask,
            text_prompts=prompts,
            return_cls_only=return_cls_only,
        )

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        execution_horizon: int | None = None,
    ):
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
                context_mask=context_mask,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
                rtc_processor=self.rtc_processor,
            )
        return self.action_head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_ids,
            context_mask=context_mask,
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        execution_horizon: int | None = None,
    ):
        return self.predict_action(
            fused_tokens,
            state,
            actions_gt,
            action_mask,
            embodiment_ids,
            context_mask,
            inference_delay,
            prev_chunk_left_over,
            execution_horizon,
        )

    def _set_module_trainable(self, module: nn.Module, trainable: bool):
        for param in module.parameters():
            param.requires_grad = trainable

    def _vlm_submodule(self, name: str) -> nn.Module:
        module = getattr(self.embedder.model, name, None)
        if not isinstance(module, nn.Module):
            raise AttributeError(
                f"InternVL model {type(self.embedder.model).__name__} has no '{name}' submodule; "
                "the native HF InternVL layout (language_model / vision_tower / "
                "multi_modal_projector) is required to apply the EVO1 finetune flags."
            )
        return module

    def set_finetune_flags(self):
        # __post_init__ resolves every finetune flag to a concrete boolean, so branch-level flags
        # are authoritative here. Freeze everything first, then re-enable the requested branches.
        self._set_module_trainable(self.embedder, False)
        self._set_module_trainable(
            self._vlm_submodule("language_model"), bool(self.config.finetune_language_model)
        )
        finetune_vision = bool(self.config.finetune_vision_model)
        self._set_module_trainable(self._vlm_submodule("vision_tower"), finetune_vision)
        self._set_module_trainable(self._vlm_submodule("multi_modal_projector"), finetune_vision)

        if not self.config.finetune_action_head:
            self._set_module_trainable(self.action_head, False)
