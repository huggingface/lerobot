#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .utils import (
    build_meta,
    get_image_keys,
    normalize_task_batch,
    pad_action_chunk,
    pad_vector,
    tensor_to_pil,
)


class DM05LerobotBatchConverter:
    """Convert raw LeRobot batches into DM05 model inputs.

    The converter keeps the tokenization contract used by the DM05 policy while
    preserving device placement for numeric tensors when possible.
    """

    def __init__(self, config: Any, tokenization_cls: type, processor: Any):
        self.config = config
        self._tokenizer = tokenization_cls(
            processor=processor,
            n_bins=config.n_bins,
            max_length=config.tokenizer_max_length,
            add_state=config.add_state,
        )

    def convert_lerobot_batch(self, batch: dict[str, Any], include_actions: bool) -> dict[str, Any]:
        """Tokenize one LeRobot batch and attach DM05 state/action tensors."""
        if OBS_STATE not in batch:
            raise ValueError(f"DM05 raw LeRobot batch requires `{OBS_STATE}` or pre-tokenized `input_ids`.")
        state = batch[OBS_STATE]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = int(state.shape[0])
        image_keys = get_image_keys(batch, self.config.image_keys)
        if not image_keys:
            raise ValueError("DM05 raw LeRobot batch requires at least one visual observation key.")

        action_batch = batch.get(ACTION)
        if action_batch is not None:
            if action_batch.dim() == 1:
                action_batch = action_batch.unsqueeze(0).unsqueeze(1)
            elif action_batch.dim() == 2:
                action_batch = (
                    action_batch.unsqueeze(1)
                    if action_batch.shape[0] == batch_size
                    else action_batch.unsqueeze(0)
                )
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad_kwargs = {"dtype": torch.bool}
            if action_batch is not None:
                action_is_pad_kwargs["device"] = action_batch.device
            action_is_pad = torch.as_tensor(action_is_pad, **action_is_pad_kwargs)
            if action_is_pad.ndim == 1:
                if action_is_pad.shape[0] == batch_size:
                    action_is_pad = action_is_pad.unsqueeze(-1)
                elif batch_size == 1:
                    action_is_pad = action_is_pad.unsqueeze(0)
            if action_is_pad.shape[0] != batch_size:
                raise ValueError(
                    f"action_is_pad batch size {action_is_pad.shape[0]} does not match {batch_size}."
                )
        tasks = normalize_task_batch(batch.get("task"), batch_size, "Execute the robot action.")
        meta = build_meta(image_keys)
        action_dim, state_dim = int(self.config.max_action_dim), int(self.config.max_state_dim)
        state_for_text_batch = None
        if self.config.add_state:
            state_for_text_batch = (
                state.detach().to(dtype=torch.float32).cpu().numpy().astype("float32", copy=False)
            )
        samples = []
        states, actions, action_dim_masks, timestep_masks = [], [], [], []
        for idx in range(batch_size):
            model_state = state[idx].to(torch.float32)
            sample = {
                "prompt": tasks[idx],
                "images": [tensor_to_pil(batch[key][idx]) for key in image_keys],
                "state": None if state_for_text_batch is None else state_for_text_batch[idx],
                "meta_data": meta,
            }
            if include_actions:
                if action_batch is None:
                    raise ValueError("DM05 training requires an action batch.")
                model_action = action_batch[idx].to(torch.float32)
                if model_action.ndim == 1:
                    model_action = model_action.unsqueeze(0)
                actions.append(pad_action_chunk(model_action, self.config.chunk_size, action_dim))
                action_dim_mask = torch.zeros(action_dim, device=model_action.device, dtype=torch.bool)
                action_dim_mask[: min(int(model_action.shape[-1]), action_dim)] = True
                action_dim_masks.append(action_dim_mask)
                timestep_mask = torch.zeros(
                    self.config.chunk_size,
                    device=model_action.device,
                    dtype=torch.bool,
                )
                timestep_mask[min(model_action.shape[0], self.config.chunk_size) :] = True
                if action_is_pad is not None:
                    source_mask = action_is_pad[idx, : self.config.chunk_size].to(device=timestep_mask.device)
                    timestep_mask[: source_mask.numel()] = source_mask
                    timestep_mask[source_mask.numel() :] = True
                timestep_masks.append(timestep_mask)
            states.append(pad_vector(model_state, state_dim))
            samples.append(sample)

        tokenized = self._tokenizer.tokenize_robot_batch(samples)
        tokenized["states"] = torch.stack(states)
        if include_actions:
            tokenized["actions"] = torch.stack(actions)
            tokenized["action_dim_mask"] = torch.stack(action_dim_masks)
            tokenized["action_is_pad"] = torch.stack(timestep_masks)
            tokenized["has_actions"] = torch.ones(
                batch_size,
                device=tokenized["actions"].device,
                dtype=torch.bool,
            )
        return tokenized
